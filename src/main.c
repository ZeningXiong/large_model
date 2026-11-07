/* main.c - qms command line.
 *   qms info MODEL
 *   qms deq  MODEL TENSOR ROW0 NROWS OUT.bin       dequantize rows (flattened ne1*ne2) to f32
 *   qms run  MODEL --ids a,b,c [--gen N] [--top K] [--ctx N] [--mode inc|recompute]
 *            [--dump-prompt-logits OUT.bin]
 */
#include "model.h"
#include "tok.h"

int qms_cmd_chat(int argc, char **argv);
#include "qms.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static uint64_t peak_rss_bytes(void)
{
    PROCESS_MEMORY_COUNTERS pmc;
    return GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) ? (uint64_t)pmc.PeakWorkingSetSize : 0;
}
#else
#include <sys/resource.h>
static uint64_t peak_rss_bytes(void)
{
    struct rusage ru;
    return getrusage(RUSAGE_SELF, &ru) == 0 ? (uint64_t)ru.ru_maxrss * 1024 : 0;
}
#endif

static double now_s(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmd_info(const char *path)
{
    QmsGguf g;
    if (qms_gguf_open(&g, path) != 0) return 1;
    printf("%d shard(s), %d tensors, %d kv\n", g.n_shards, g.n_tensors, g.n_kv);
    for (int i = 0; i < g.n_kv; i++) {
        const QmsKV *k = &g.kv[i];
        if (k->vtype == 8) printf("  %s = %s\n", k->key, k->str);
        else if (k->vtype == 6 || k->vtype == 12) printf("  %s = %g\n", k->key, k->v.f);
        else printf("  %s = %lld\n", k->key, (long long)k->v.i);
    }
    uint64_t total = 0;
    for (int i = 0; i < g.n_tensors; i++) total += g.tensors[i].nbytes;
    printf("tensor bytes: %.2f GB\n", total / 1e9);
    qms_gguf_close(&g);
    return 0;
}

static int cmd_deq(const char *path, const char *name, long long r0, long long nr, const char *out)
{
    QmsGguf g;
    if (qms_gguf_open(&g, path) != 0) return 1;
    const QmsTensor *t = qms_gguf_find(&g, name);
    if (!t) { fprintf(stderr, "no tensor %s\n", name); return 1; }
    const uint64_t rb = qms_row_bytes(t->type, t->ne[0]);
    const long long rows = t->ne[1] * t->ne[2] * t->ne[3];
    if (!rb || r0 < 0 || r0 + nr > rows) { fprintf(stderr, "bad row range\n"); return 1; }
    unsigned char *buf = (unsigned char *)malloc(rb);
    float *f = (float *)malloc((size_t)t->ne[0] * sizeof(float));
    FILE *fo = fopen(out, "wb");
    if (!buf || !f || !fo) return 1;
    for (long long r = r0; r < r0 + nr; r++) {
        if (qms_gguf_read(&g, t, (uint64_t)r * rb, rb, buf) != 0) return 1;
        if (qms_dequant_row(t->type, buf, f, t->ne[0]) != 0) { fprintf(stderr, "unsupported type\n"); return 1; }
        fwrite(f, sizeof(float), (size_t)t->ne[0], fo);
    }
    fclose(fo);
    printf("%s %s ne0=%lld rows %lld..%lld -> %s\n", name, qms_type_name(t->type),
           (long long)t->ne[0], r0, r0 + nr - 1, out);
    qms_gguf_close(&g);
    return 0;
}

static void log_softmax(const float *x, int n, double *out)
{
    double mx = -INFINITY, z = 0.0;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (int i = 0; i < n; i++) z += exp((double)x[i] - mx);
    const double lz = log(z) + mx;
    for (int i = 0; i < n; i++) out[i] = (double)x[i] - lz;
}

static int argmax(const float *x, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

static void print_top(int step, const float *logits, int vocab, int k, double *lp)
{
    log_softmax(logits, vocab, lp);
    printf("step %d top%d:", step, k);
    char *taken = (char *)calloc((size_t)vocab, 1);
    for (int j = 0; j < k; j++) {
        int b = -1;
        for (int i = 0; i < vocab; i++) if (!taken[i] && (b < 0 || lp[i] > lp[b])) b = i;
        taken[b] = 1;
        printf(" %d:%.5f", b, lp[b]);
    }
    printf("\n");
    free(taken);
}

static int cmd_run(int argc, char **argv)
{
    const char *path = argv[2];
    const char *ids_s = NULL, *dump = NULL, *mode = "inc", *trace_path = NULL;
    int gen = 0, top = 10, ctx = 2048, stream = 0, force = 0, direct = 0;
    double cache_gb = 1.0, mem_gb = 0.0, pf_min_p = 0.0;
    int cache_policy = 1, pf_depth = 0, pf_extra = 0, pin_layers = -1, no_overlap = 0;
    {   /* default: one thread per physical core (SMT siblings only contend here) */
        const int procs = omp_get_num_procs();
        omp_set_num_threads(procs > 1 ? procs / 2 : 1);
    }
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top") && i + 1 < argc) top = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--dump-prompt-logits") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--exact")) qms_exact_math = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) omp_set_num_threads(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--stream-experts")) stream = 1;
        else if (!strcmp(argv[i], "--force")) force = 1;
        else if (!strcmp(argv[i], "--direct")) direct = 1;
        else if (!strcmp(argv[i], "--no-overlap")) no_overlap = 1;
        else if (!strcmp(argv[i], "--prefetch-min-p") && i + 1 < argc) pf_min_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--prefetch") && i + 1 < argc) pf_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefetch-extra") && i + 1 < argc) pf_extra = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-policy") && i + 1 < argc) cache_policy = strcmp(argv[++i], "lru") != 0;
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--mem-gb") && i + 1 < argc) mem_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pin-layers") && i + 1 < argc) pin_layers = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!ids_s) { fprintf(stderr, "--ids required\n"); return 2; }
    int *ids = (int *)malloc((size_t)(ctx + 1) * sizeof(int));
    int n = 0;
    for (const char *p = ids_s; *p && n < ctx;) {
        ids[n++] = (int)strtol(p, (char **)&p, 10);
        while (*p == ',' || *p == ' ') p++;
    }
    if (n + gen > ctx) { fprintf(stderr, "prompt + gen exceeds --ctx\n"); return 2; }

    double t0 = now_s();
    QmsModel m;
    QmsLoadOpts lo = { .stream_experts = stream || mem_gb > 0, .ctx = ctx, .force = force,
                       .direct = direct, .cache_gb = cache_gb, .cache_policy = cache_policy,
                       .prefetch_depth = pf_depth, .prefetch_extra = pf_extra, .mem_gb = mem_gb,
                       .pin_layers = pin_layers, .no_overlap = no_overlap, .prefetch_min_p = pf_min_p };
    if (qms_model_load(&m, path, &lo) != 0) return 1;
    if (trace_path && !(m.trace = fopen(trace_path, "wb"))) {
        fprintf(stderr, "cannot write %s\n", trace_path);
        return 1;
    }
    const int V = m.c.vocab;
    fprintf(stderr, "loaded %s: %d layers, %.2f GB resident, %.1f s\n", path, m.c.n_layer,
            m.resident_bytes / 1e9, now_s() - t0);

    float *logits = (float *)malloc((size_t)n * V * sizeof(float));
    double *lp = (double *)malloc((size_t)V * sizeof(double));
    const int recompute = !strcmp(mode, "recompute");

    t0 = now_s();
    if (qms_forward(&m, ids, n, 0, logits, dump != NULL) != 0) return 1;
    fprintf(stderr, "prompt: %d tokens in %.2f s\n", n, now_s() - t0);
    const float *last = dump ? logits + (size_t)(n - 1) * V : logits;
    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f) return 1;
        fwrite(logits, sizeof(float), (size_t)n * V, f);
        fclose(f);
    }
    print_top(0, last, V, top, lp);
    int next = argmax(last, V);

    t0 = now_s();
    qms_prof_on = getenv("QMS_PROF") != NULL;
    const uint64_t dec_bytes0 = m.ex_bytes_read;
    const double dec_s0 = m.ex_read_s;
    memset(qms_prof, 0, sizeof qms_prof);
    for (int s = 1; s < gen; s++) {
        ids[n++] = next;
        if (recompute) {
            if (qms_forward(&m, ids, n, 0, logits, 0) != 0) return 1;   /* rebuild KV from scratch */
        } else {
            if (qms_forward(&m, ids + n - 1, 1, n - 1, logits, 0) != 0) return 1;
        }
        print_top(s, logits, V, top, lp);
        next = argmax(logits, V);
    }
    if (gen > 0) ids[n++] = next;
    if (gen > 1) fprintf(stderr, "decode: %d tokens in %.2f s (%.3f s/token)\n", gen - 1,
                         now_s() - t0, (now_s() - t0) / (gen - 1));
    if (qms_prof_on && gen > 1) {
        static const char *nm[QMS_PROF_N] = { "qkv", "attn", "o_proj", "router", "fetch", "moe", "output" };
        double tot = 0.0;
        for (int k = 0; k < QMS_PROF_N; k++) tot += qms_prof[k];
        fprintf(stderr, "profile per decode token (ms):");
        for (int k = 0; k < QMS_PROF_N; k++) fprintf(stderr, " %s %.2f", nm[k], 1e3 * qms_prof[k] / (gen - 1));
        fprintf(stderr, " | sum %.2f\n", 1e3 * tot / (gen - 1));
    }
    if (m.streamed) {
        const uint64_t db = m.ex_bytes_read - dec_bytes0;
        const double ds = m.ex_read_s - dec_s0;
        const uint64_t req = m.cache.hits + m.cache.misses;
        fprintf(stderr, "expert cache: %llu requests, %llu hits (%.1f%%), %llu misses\n",
                (unsigned long long)req, (unsigned long long)m.cache.hits,
                req ? 100.0 * m.cache.hits / req : 0.0, (unsigned long long)m.cache.misses);
        if (m.pf_depth > 0)
            fprintf(stderr, "prefetch (depth %d, top-%d): %llu experts read ahead, %llu later requested "
                            "(%.1f%% precision), %llu still in flight when needed\n",
                    m.pf_depth, m.c.n_used + m.pf_extra, (unsigned long long)m.cache.pf_issued,
                    (unsigned long long)m.cache.pf_used,
                    m.cache.pf_issued ? 100.0 * m.cache.pf_used / m.cache.pf_issued : 0.0,
                    (unsigned long long)m.cache.pf_waited);
        if (m.tr.n_pin < m.c.n_layer)
            fprintf(stderr, "trunk streamed: %d/%d layers pinned, %.2f GB read in %llu reads, %.1f s waiting\n",
                    m.tr.n_pin, m.c.n_layer, m.tr.bytes_read / 1e9, (unsigned long long)m.tr.reads, m.tr.wait_s);
        fprintf(stderr, "experts streamed (whole run): %.2f GB in %llu reads, %.1f s waiting, %.2f GB/s\n",
                m.ex_bytes_read / 1e9, (unsigned long long)m.ex_reads, m.ex_read_s,
                m.ex_read_s > 0 ? m.ex_bytes_read / 1e9 / m.ex_read_s : 0.0);
        if (gen > 1)
            fprintf(stderr, "experts streamed (decode)   : %.2f GB/token, %.3f s/token waiting, %.2f GB/s\n",
                    db / 1e9 / (gen - 1), ds / (gen - 1), ds > 0 ? db / 1e9 / ds : 0.0);
    }
    fprintf(stderr, "PEAK working set: %.2f GB   <- measured; the plan above is a forecast\n",
            peak_rss_bytes() / 1e9);
    printf("ids:");
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : " ", ids[i]);
    printf("\n");
    free(logits); free(lp); free(ids);
    qms_model_free(&m);
    return 0;
}

/* qms tok MODEL FILE: print the token ids of the file's bytes (special tokens parsed). */
static int cmd_tok(const char *path, const char *file)
{
    QmsGguf g;
    QmsTok t;
    if (qms_gguf_open(&g, path) != 0 || qms_tok_load(&t, &g) != 0) return 1;
    FILE *f = fopen(file, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", file); return 1; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); return 1; }
    fclose(f);
    int *ids = NULL, nid = 0;
    qms_tok_encode(&t, buf, (size_t)n, &ids, &nid);
    for (int i = 0; i < nid; i++) printf("%s%d", i ? "," : "", ids[i]);
    printf("\n");
    free(ids); free(buf);
    qms_tok_free(&t);
    qms_gguf_close(&g);
    return 0;
}

/* qms detok MODEL a,b,c OUT: write the decoded bytes of the ids to OUT. */
static int cmd_detok(const char *path, const char *ids_s, const char *out)
{
    QmsGguf g;
    QmsTok t;
    if (qms_gguf_open(&g, path) != 0 || qms_tok_load(&t, &g) != 0) return 1;
    FILE *f = fopen(out, "wb");
    if (!f) return 1;
    char piece[512];
    for (const char *p = ids_s; *p;) {
        const int id = (int)strtol(p, (char **)&p, 10);
        fwrite(piece, 1, (size_t)qms_tok_piece(&t, id, piece, (int)sizeof piece), f);
        while (*p == ',' || *p == ' ') p++;
    }
    fclose(f);
    qms_tok_free(&t);
    qms_gguf_close(&g);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "info")) return cmd_info(argv[2]);
    if (argc >= 7 && !strcmp(argv[1], "deq"))
        return cmd_deq(argv[2], argv[3], atoll(argv[4]), atoll(argv[5]), argv[6]);
    if (argc >= 3 && !strcmp(argv[1], "run")) return cmd_run(argc, argv);
    if (argc >= 4 && !strcmp(argv[1], "tok")) return cmd_tok(argv[2], argv[3]);
    if (argc >= 3 && !strcmp(argv[1], "chat")) return qms_cmd_chat(argc, argv);
    if (argc >= 5 && !strcmp(argv[1], "detok")) return cmd_detok(argv[2], argv[3], argv[4]);
    fprintf(stderr, "qms %s\nusage:\n  qms info MODEL\n  qms deq MODEL TENSOR ROW0 NROWS OUT.bin\n"
                    "  qms run MODEL --ids a,b,c [--gen N] [--top K] [--ctx N] [--mode inc|recompute]\n"
                    "          [--dump-prompt-logits OUT.bin] [--exact] [--threads N] [--trace F]\n"
                    "          [--stream-experts [--direct] [--cache-gb G] [--cache-policy lfu|lru]\n"
                    "           [--prefetch D] [--prefetch-extra N]] [--force]\n"
                    "  qms chat MODEL [--system TEXT] [--temp T] [--top-k K] [--top-p P] [--seed S]\n"
                    "          [--greedy] [--max-new N] [--dump-ids F] + the engine options above\n"
                    "  qms tok MODEL TEXTFILE          token ids of the file's UTF-8 text\n"
                    "  qms detok MODEL a,b,c OUTFILE   decoded bytes of ids\n", QMS_VERSION);
    return 2;
}
