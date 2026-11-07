/* chat.c - interactive multi-turn chat: ChatML template, sampling, streamed UTF-8 output.
 *
 * The KV cache persists across turns: each turn only runs the new tokens. A reply ends at
 * <|im_end|>; that token is fed to the model at the start of the next turn, so the context is
 * exactly the official template:  <|im_start|>role\ncontent<|im_end|>\n ...                   */
#include "model.h"
#include "tok.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* ------------------------------------------------------------ sampler ---- */

typedef struct {
    float temp, top_p;
    int top_k;
    uint64_t rng;
} Sampler;

static uint32_t pcg32(uint64_t *s)
{
    const uint64_t old = *s;
    *s = old * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t xs = (uint32_t)(((old >> 18) ^ old) >> 27), rot = (uint32_t)(old >> 59);
    return (xs >> rot) | (xs << ((32 - rot) & 31));
}

/* greedy if temp <= 0; else top-k, then nucleus (top-p), then sample. Deterministic per seed. */
static int sample(Sampler *sp, const float *logits, int V)
{
    int best = 0;
    for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
    if (sp->temp <= 0.0f) return best;
    const int K = sp->top_k > 0 && sp->top_k < V ? sp->top_k : (V < 1024 ? V : 1024);
    int *idx = (int *)malloc((size_t)K * sizeof(int));
    double *p = (double *)malloc((size_t)K * sizeof(double));
    int n = 0;
    for (int i = 0; i < V; i++) {                 /* keep the K largest, sorted descending */
        if (n < K || logits[i] > logits[idx[n - 1]]) {
            int j = n < K ? n++ : K - 1;
            while (j > 0 && logits[idx[j - 1]] < logits[i]) { idx[j] = idx[j - 1]; j--; }
            idx[j] = i;
        }
    }
    double z = 0.0;
    for (int j = 0; j < n; j++) { p[j] = exp(((double)logits[idx[j]] - logits[idx[0]]) / sp->temp); z += p[j]; }
    double cum = 0.0;
    int keep = n;
    for (int j = 0; j < n; j++) {
        cum += p[j] / z;
        if (cum >= sp->top_p) { keep = j + 1; break; }
    }
    double zk = 0.0;
    for (int j = 0; j < keep; j++) zk += p[j];
    const double r = (pcg32(&sp->rng) / 4294967296.0) * zk;
    double acc = 0.0;
    int pick = idx[keep - 1];
    for (int j = 0; j < keep; j++) { acc += p[j]; if (r < acc) { pick = idx[j]; break; } }
    free(idx); free(p);
    return pick;
}

/* ------------------------------------------------------------ console ---- */

/* One line of user input as UTF-8 (console: read UTF-16 and convert; pipe/file: bytes). */
static char *read_line(void)
{
#ifdef _WIN32
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(in, &mode)) {
        wchar_t wbuf[8192];
        DWORD got = 0;
        if (!ReadConsoleW(in, wbuf, 8191, &got, NULL) || got == 0) return NULL;
        while (got && (wbuf[got - 1] == L'\n' || wbuf[got - 1] == L'\r')) got--;
        const int nb = WideCharToMultiByte(CP_UTF8, 0, wbuf, (int)got, NULL, 0, NULL, NULL);
        char *s = (char *)malloc((size_t)nb + 1);
        WideCharToMultiByte(CP_UTF8, 0, wbuf, (int)got, s, nb, NULL, NULL);
        s[nb] = 0;
        return s;
    }
#endif
    char *s = (char *)malloc(65536);
    if (!fgets(s, 65536, stdin)) { free(s); return NULL; }
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
    return s;
}

/* Stream bytes to stdout, holding back an incomplete UTF-8 sequence until it completes. */
typedef struct { char pend[8]; int np; } Out8;

static void out_bytes(Out8 *o, const char *b, int n)
{
    char buf[600];
    int m = 0;
    for (int i = 0; i < o->np; i++) buf[m++] = o->pend[i];
    for (int i = 0; i < n && m < (int)sizeof buf; i++) buf[m++] = b[i];
    int cut = m;   /* find a trailing incomplete sequence */
    for (int back = 1; back <= 3 && back <= m; back++) {
        const unsigned char c = (unsigned char)buf[m - back];
        if ((c & 0xC0) == 0x80) continue;
        const int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        if (need > back) cut = m - back;
        break;
    }
    fwrite(buf, 1, (size_t)cut, stdout);
    fflush(stdout);
    o->np = m - cut;
    memcpy(o->pend, buf + cut, (size_t)o->np);
}

/* ------------------------------------------------------------ chat ---- */

static double now_s(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

int qms_cmd_chat(int argc, char **argv)
{
    const char *path = argv[2], *system = NULL, *dump = NULL;
    QmsLoadOpts lo = { .stream_experts = 0, .ctx = 4096, .force = 0, .direct = 0, .cache_gb = 1.0,
                       .cache_policy = 1, .prefetch_depth = 0, .prefetch_extra = 0, .mem_gb = 0.0,
                       .pin_layers = -1, .no_overlap = 0, .prefetch_min_p = 0.0 };
    Sampler sp = { 0.7f, 0.8f, 20, 0 };
    uint64_t seed = (uint64_t)time(NULL);
    int max_new = 1024;
    {
        const int procs = omp_get_num_procs();
        omp_set_num_threads(procs > 1 ? procs / 2 : 1);
    }
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i];
        const int more = i + 1 < argc;
        if (!strcmp(a, "--system") && more) system = argv[++i];
        else if (!strcmp(a, "--temp") && more) sp.temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-p") && more) sp.top_p = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && more) sp.top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && more) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--greedy")) sp.temp = 0.0f;
        else if (!strcmp(a, "--max-new") && more) max_new = atoi(argv[++i]);
        else if (!strcmp(a, "--ctx") && more) lo.ctx = atoi(argv[++i]);
        else if (!strcmp(a, "--stream-experts")) lo.stream_experts = 1;
        else if (!strcmp(a, "--direct")) lo.direct = 1;
        else if (!strcmp(a, "--no-overlap")) lo.no_overlap = 1;
        else if (!strcmp(a, "--prefetch-min-p") && more) lo.prefetch_min_p = atof(argv[++i]);
        else if (!strcmp(a, "--mem-gb") && more) { lo.mem_gb = atof(argv[++i]); lo.stream_experts = 1; }
        else if (!strcmp(a, "--pin-layers") && more) lo.pin_layers = atoi(argv[++i]);
        else if (!strcmp(a, "--cache-gb") && more) lo.cache_gb = atof(argv[++i]);
        else if (!strcmp(a, "--cache-policy") && more) lo.cache_policy = strcmp(argv[++i], "lru") != 0;
        else if (!strcmp(a, "--prefetch") && more) lo.prefetch_depth = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && more) omp_set_num_threads(atoi(argv[++i]));
        else if (!strcmp(a, "--exact")) qms_exact_math = 1;
        else if (!strcmp(a, "--force")) lo.force = 1;
        else if (!strcmp(a, "--dump-ids") && more) dump = argv[++i];
        else { fprintf(stderr, "unknown argument %s\n", a); return 2; }
    }
    sp.rng = seed * 2654435761u + 1;
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    QmsModel m;
    if (qms_model_load(&m, path, &lo) != 0) return 1;
    QmsTok tok;
    if (qms_tok_load(&tok, &m.g) != 0) return 1;
    const int V = m.c.vocab;
    float *logits = (float *)malloc((size_t)V * sizeof(float));
    int *ctx_ids = (int *)malloc((size_t)lo.ctx * sizeof(int));
    int pos = 0, pending = -1, turn = 0;
    fprintf(stderr, "ready: %s, ctx %d, temp %.2f top-k %d top-p %.2f seed %llu. Empty line or /exit quits.\n",
            path, lo.ctx, sp.temp, sp.top_k, sp.top_p, (unsigned long long)seed);

    for (;;) {
        fprintf(stderr, "\n> ");
        char *line = read_line();
        if (!line || !line[0] || !strcmp(line, "/exit")) { free(line); break; }
        /* the template text of this turn */
        const size_t cap = strlen(line) + (system ? strlen(system) : 0) + 256;
        char *text = (char *)malloc(cap);
        int tl = 0;
        if (turn == 0 && system) tl += snprintf(text + tl, cap - (size_t)tl, "<|im_start|>system\n%s<|im_end|>\n", system);
        if (turn > 0) tl += snprintf(text + tl, cap - (size_t)tl, "\n");
        tl += snprintf(text + tl, cap - (size_t)tl, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);
        int *ids = NULL, n = 0;
        qms_tok_encode(&tok, text, (size_t)tl, &ids, &n);
        free(text);
        free(line);
        /* previous reply's <|im_end|> goes first */
        int *all = (int *)malloc(((size_t)n + 1) * sizeof(int));
        int na = 0;
        if (pending >= 0) all[na++] = pending;
        memcpy(all + na, ids, (size_t)n * sizeof(int));
        na += n;
        free(ids);
        if (pos + na + 1 >= lo.ctx) { fprintf(stderr, "[context full: %d of %d]\n", pos, lo.ctx); free(all); break; }

        double t0 = now_s();
        if (qms_forward(&m, all, na, pos, logits, 0) != 0) { free(all); break; }
        memcpy(ctx_ids + pos, all, (size_t)na * sizeof(int));
        pos += na;
        free(all);
        const double t_prompt = now_s() - t0;

        Out8 o = { { 0 }, 0 };
        int gen = 0;
        pending = -1;
        t0 = now_s();
        for (;;) {
            const int id = sample(&sp, logits, V);
            if (id == tok.eos || gen >= max_new || pos + 1 >= lo.ctx) { pending = tok.eos; break; }
            char piece[512];
            out_bytes(&o, piece, qms_tok_piece(&tok, id, piece, (int)sizeof piece));
            ctx_ids[pos] = id;
            if (qms_forward(&m, &id, 1, pos, logits, 0) != 0) { pending = -2; break; }
            pos++;
            gen++;
        }
        if (o.np) out_bytes(&o, "", 0);
        fputc('\n', stdout);   /* one reply per line on stdout, so piped transcripts stay readable */
        fflush(stdout);
        if (pending == -2) break;
        const double t_gen = now_s() - t0;
        fprintf(stderr, "\n[%d prompt tokens in %.1f s | %d tokens in %.1f s, %.2f s/token | context %d/%d]",
                na, t_prompt, gen, t_gen, gen ? t_gen / gen : 0.0, pos, lo.ctx);
        turn++;
    }
    if (dump) {
        FILE *f = fopen(dump, "w");
        if (f) {
            for (int i = 0; i < pos; i++) fprintf(f, "%s%d", i ? "," : "", ctx_ids[i]);
            if (pending >= 0) fprintf(f, "%s%d", pos ? "," : "", pending);
            fprintf(f, "\n");
            fclose(f);
        }
    }
    fprintf(stderr, "\n");
    free(logits); free(ctx_ids);
    qms_tok_free(&tok);
    qms_model_free(&m);
    return 0;
}
