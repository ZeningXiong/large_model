/* model.c - Qwen3-MoE forward pass, everything resident in RAM (correctness baseline).
 *
 * Numerics deliberately mirror tools/ref_qwen3moe.py: RMSNorm and softmax in double,
 * RoPE angles in double, everything else float32. The goal of this file is to agree
 * with the reference, not to be fast; the fast paths are added later and checked
 * against this one. */
#include "model.h"
#include "kernels.h"
#include "aio.h"
#include <omp.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ loading ---- */

int qms_prof_on = 0;
double qms_prof[QMS_PROF_N];
double qms_now(void) { return omp_get_wtime(); }

static void *track(QmsModel *m, void *p)
{
    void **na = (void **)realloc(m->allocs, (size_t)(m->n_allocs + 1) * sizeof *na);
    if (!na) { free(p); return NULL; }
    m->allocs = na;
    m->allocs[m->n_allocs++] = p;
    return p;
}

static const QmsTensor *need(QmsModel *m, const char *name)
{
    const QmsTensor *t = qms_gguf_find(&m->g, name);
    if (!t) fprintf(stderr, "model: missing tensor %s\n", name);
    return t;
}

static void *read_all(QmsModel *m, const QmsTensor *t)
{
    void *buf = malloc((size_t)t->nbytes);
    if (!buf) { fprintf(stderr, "model: out of memory for %s (%llu bytes)\n", t->name,
                        (unsigned long long)t->nbytes); return NULL; }
    if (qms_gguf_read(&m->g, t, 0, t->nbytes, buf) != 0) {
        fprintf(stderr, "model: read failed for %s\n", t->name);
        free(buf);
        return NULL;
    }
    m->resident_bytes += t->nbytes;
    return track(m, buf);
}

static int load_f32_vec(QmsModel *m, const char *name, int64_t n, float **out)
{
    const QmsTensor *t = need(m, name);
    if (!t) return -1;
    if (t->type != QMS_T_F32 || t->ne[0] != n) {
        fprintf(stderr, "model: %s expected F32[%lld], got %s[%lld]\n", name, (long long)n,
                qms_type_name(t->type), (long long)t->ne[0]);
        return -1;
    }
    *out = (float *)read_all(m, t);
    return *out ? 0 : -1;
}

/* 2-D matrix with ne0 = cols (input), ne1 = rows (output) */
static int load_mat(QmsModel *m, const char *name, int64_t rows, int64_t cols, QmsMat *out)
{
    const QmsTensor *t = need(m, name);
    if (!t) return -1;
    if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 || !qms_row_bytes(t->type, cols)) {
        fprintf(stderr, "model: %s expected [%lld x %lld], got ne=[%lld,%lld,%lld] %s\n", name,
                (long long)rows, (long long)cols, (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], qms_type_name(t->type));
        return -1;
    }
    out->type = t->type;
    out->rows = rows;
    out->cols = cols;
    out->row_bytes = qms_row_bytes(t->type, cols);
    out->data = (const unsigned char *)read_all(m, t);
    return out->data ? 0 : -1;
}

/* Like load_mat, but only records shape and type: the data is streamed later. */
static int load_mat_shape(QmsModel *m, const char *name, int64_t rows, int64_t cols, QmsMat *out)
{
    const QmsTensor *t = need(m, name);
    if (!t) return -1;
    if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 || !qms_row_bytes(t->type, cols)) {
        fprintf(stderr, "model: %s has an unexpected shape\n", name);
        return -1;
    }
    out->type = t->type;
    out->rows = rows;
    out->cols = cols;
    out->row_bytes = qms_row_bytes(t->type, cols);
    out->data = NULL;
    return 0;
}

/* Record where layer l's attention tensors sit in the file. Streaming reads the whole block
 * at once, which requires the tensors to be contiguous in one shard (checked here). */
static int att_locate(QmsModel *m, int l, int must_be_contiguous)
{
    static const char *NAMES[7] = { "attn_q", "attn_k", "attn_v", "attn_output",
                                    "attn_norm", "attn_q_norm", "attn_k_norm" };
    QmsLayer *L = &m->L[l];
    const QmsTensor *t[7];
    char n[QMS_MAX_NAME];
    uint64_t lo = UINT64_MAX, hi = 0, sum = 0;
    for (int i = 0; i < 7; i++) {
        snprintf(n, sizeof n, "blk.%d.%s.weight", l, NAMES[i]);
        if (!(t[i] = need(m, n))) return -1;
        if (t[i]->off < lo) lo = t[i]->off;
        if (t[i]->off + t[i]->nbytes > hi) hi = t[i]->off + t[i]->nbytes;
        sum += t[i]->nbytes;
    }
    int same = 1;
    for (int i = 1; i < 7; i++) same &= t[i]->shard == t[0]->shard;
    if (must_be_contiguous && (!same || hi - lo > sum + 7 * 64)) {
        fprintf(stderr, "model: layer %d attention tensors are not one contiguous block; "
                        "cannot stream the trunk for this file\n", l);
        return -1;
    }
    L->att_shard = t[0]->shard;
    L->att_off = lo;
    L->att_len = hi - lo;
    for (int i = 0; i < 4; i++) L->att_rel[i] = t[i]->off - lo;
    return 0;
}

static int load_experts(QmsModel *m, const char *name, int64_t rows, int64_t cols, int n_exp,
                        QmsExperts *out)
{
    const QmsTensor *t = need(m, name);
    if (!t) return -1;
    if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != n_exp || !qms_row_bytes(t->type, cols)) {
        fprintf(stderr, "model: %s expected [%d x %lld x %lld], got ne=[%lld,%lld,%lld]\n", name,
                n_exp, (long long)rows, (long long)cols, (long long)t->ne[0],
                (long long)t->ne[1], (long long)t->ne[2]);
        return -1;
    }
    out->m.type = t->type;
    out->m.rows = rows;
    out->m.cols = cols;
    out->m.row_bytes = qms_row_bytes(t->type, cols);
    out->expert_bytes = out->m.row_bytes * (uint64_t)rows;
    if (m->streamed) {            /* stays on disk; fetched per layer in moe() */
        out->t = t;
        out->m.data = NULL;
        return 0;
    }
    out->t = NULL;
    out->m.data = (const unsigned char *)read_all(m, t);
    return out->m.data ? 0 : -1;
}

static int is_expert_tensor(const char *name) { return strstr(name, "_exps.") != NULL; }

static int ecache_init(QmsModel *m, uint64_t budget, int policy);
static void pf_drain(QmsModel *m);
static void ecache_free(QmsModel *m);
static void trunk_drain(QmsModel *m);

static const char *ATT_NAMES[7] = { "attn_q", "attn_k", "attn_v", "attn_output",
                                     "attn_norm", "attn_q_norm", "attn_k_norm" };

static uint64_t layer_att_bytes(const QmsModel *m, int l)
{
    uint64_t s = 0;
    char n[QMS_MAX_NAME];
    for (int i = 0; i < 7; i++) {
        snprintf(n, sizeof n, "blk.%d.%s.weight", l, ATT_NAMES[i]);
        const QmsTensor *t = qms_gguf_find(&m->g, n);
        if (t) s += t->nbytes;
    }
    return s;
}

static uint64_t round_up_u(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

/* Add up everything this run will hold BEFORE allocating any of it, and compare with what
 * the OS says is available. A plan is a forecast; the measured peak is reported at exit.
 * With a total budget (--mem-gb) the trunk is pinned before anything goes to the expert
 * cache: a pinned byte saves a full byte of reading every token, a cache byte saves less. */
static int memory_plan(QmsModel *m, const QmsLoadOpts *o)
{
    const QmsCfg *c = &m->c;
    uint64_t trunk = 0, experts = 0, one_layer_experts = 0, att_total = 0, att_max = 0, embd = 0;
    for (int i = 0; i < m->g.n_tensors; i++) {
        const QmsTensor *t = &m->g.tensors[i];
        if (is_expert_tensor(t->name)) {
            experts += t->nbytes;
            if (!strncmp(t->name, "blk.0.", 6)) one_layer_experts += t->nbytes;
        } else {
            trunk += t->nbytes;
        }
        if (!strcmp(t->name, "token_embd.weight")) embd = t->nbytes;
    }
    for (int l = 0; l < c->n_layer; l++) {
        const uint64_t a = layer_att_bytes(m, l);
        att_total += a;
        if (a > att_max) att_max = a;
    }
    const uint64_t kv = 2ull * c->n_layer * o->ctx * c->n_head_kv * c->head_dim * sizeof(float);
    const uint64_t per_expert = one_layer_experts / (uint64_t)c->n_expert;
    const uint64_t min_cache = per_expert * (uint64_t)c->n_used + (64ull << 20);
    const uint64_t scratch = 64ull << 20;
    const uint64_t slot = round_up_u(att_max + 2 * QMS_DIRECT_ALIGN, QMS_DIRECT_ALIGN);
    const int tied = qms_gguf_find(&m->g, "output.weight") == NULL;

    QmsTrunk *T = &m->tr;
    T->n_pin = c->n_layer;
    T->stream_embd = 0;
    uint64_t cache = 0;
    if (o->stream_experts) {
        cache = (uint64_t)(o->cache_gb * 1e9);
        if (cache < min_cache) cache = min_cache;
    }
    if (o->mem_gb > 0) {
        const uint64_t budget = (uint64_t)(o->mem_gb * 1e9);
        T->stream_embd = !tied;                       /* 8 KB per token instead of resident */
        const uint64_t other = trunk - att_total - (T->stream_embd ? embd : 0);
        const uint64_t fixed = other + kv + scratch + min_cache;
        if (fixed + att_total <= budget) {
            T->n_pin = c->n_layer;
            cache = budget - fixed - att_total + min_cache;
        } else if (fixed + 2 * slot + att_max <= budget) {
            T->n_pin = (int)((budget - fixed - 2 * slot) / att_max);
            if (T->n_pin > c->n_layer - 1) T->n_pin = c->n_layer - 1;
            cache = budget - fixed - 2 * slot - (uint64_t)T->n_pin * att_max + min_cache;
        } else {
            fprintf(stderr, "REFUSING TO START: --mem-gb %.2f is below the floor of %.2f GB "
                            "(non-attention trunk + KV + one layer's experts + ring).\n",
                    o->mem_gb, (fixed + 2 * slot) / 1e9);
            return -1;
        }
    }
    if (o->pin_layers >= 0) {
        T->n_pin = o->pin_layers < c->n_layer ? o->pin_layers : c->n_layer;
        T->stream_embd = !tied;
    }
    const int nstream = c->n_layer - T->n_pin;
    if (nstream > 0) T->slot_bytes = slot;
    m->plan_cache_bytes = cache;

    uint64_t pinned_att = 0;
    for (int l = 0; l < T->n_pin; l++) pinned_att += layer_att_bytes(m, l);
    const uint64_t resident_trunk = trunk - (att_total - pinned_att) - (T->stream_embd ? embd : 0);
    const uint64_t ring = nstream > 0 ? 2 * slot : 0;
    const uint64_t resident = resident_trunk + (o->stream_experts ? 0 : experts);
    const uint64_t need = resident + ring + kv + cache + scratch;
    const uint64_t avail = qms_mem_available();
    fprintf(stderr, "memory plan%s\n"
                    "  resident weights %8.2f GB  (%s)\n"
                    "    attention pinned %d/%d layers (%.2f GB)%s%s\n"
                    "  trunk ring       %8.2f GB  (2 slots: %d layers streamed per token, %.2f GB/token)\n"
                    "  expert cache     %8.2f GB  (%.1f MB per expert)\n"
                    "  KV cache         %8.2f GB  (ctx %d, fp32)\n"
                    "  scratch          %8.2f GB\n"
                    "  TOTAL            %8.2f GB   available %.2f GB\n",
            o->mem_gb > 0 ? " (budget-driven: trunk first, then expert cache)" : "",
            resident / 1e9, o->stream_experts ? "trunk only; experts streamed from disk" : "everything",
            T->n_pin, c->n_layer, pinned_att / 1e9, T->stream_embd ? ", embedding rows read on demand" : "",
            nstream > 0 ? ", rest streamed" : "",
            ring / 1e9, nstream, (att_total - pinned_att) / 1e9,
            cache / 1e9, per_expert / 1e6, kv / 1e9, o->ctx, scratch / 1e9, need / 1e9, avail / 1e9);
    if (avail && need > avail / 100 * 95) {
        fprintf(stderr, "REFUSING TO START: plan needs %.2f GB, %.2f GB available (95%% rule).\n"
                        "Options: --stream-experts, --mem-gb, a smaller --ctx, or --force.\n",
                need / 1e9, avail / 1e9);
        if (!o->force) return -1;
        fprintf(stderr, "--force given: starting anyway.\n");
    }
    return 0;
}

static int read_cfg(QmsModel *m)
{
    const QmsKV *arch = qms_gguf_kv(&m->g, "general.architecture");
    if (!arch || strcmp(arch->str, "qwen3moe") != 0) {
        fprintf(stderr, "model: architecture %s is not qwen3moe\n", arch ? arch->str : "(missing)");
        return -1;
    }
    QmsCfg *c = &m->c;
    int64_t v;
    double f;
#define GETI(field, key) do { if (qms_gguf_get_int(&m->g, "qwen3moe." key, &v)) return -1; c->field = (int)v; } while (0)
    GETI(n_layer, "block_count");
    GETI(n_embd, "embedding_length");
    GETI(n_head, "attention.head_count");
    GETI(n_head_kv, "attention.head_count_kv");
    GETI(head_dim, "attention.key_length");
    GETI(n_expert, "expert_count");
    GETI(n_used, "expert_used_count");
    GETI(n_ff_exp, "expert_feed_forward_length");
#undef GETI
    if (qms_gguf_get_float(&m->g, "qwen3moe.attention.layer_norm_rms_epsilon", &f)) return -1;
    c->eps = (float)f;
    if (qms_gguf_get_float(&m->g, "qwen3moe.rope.freq_base", &f)) return -1;
    c->rope_base = f;
    const QmsTensor *e = need(m, "token_embd.weight");
    if (!e) return -1;
    c->vocab = (int)e->ne[1];
    if (c->n_head % c->n_head_kv || c->n_used > c->n_expert || c->head_dim % 2 ||
        c->head_dim > QMS_MAX_HEAD_DIM || c->n_used > QMS_MAX_USED) {
        fprintf(stderr, "model: inconsistent config\n");
        return -1;
    }
    return 0;
}

int qms_model_load(QmsModel *m, const char *path, const QmsLoadOpts *o)
{
    memset(m, 0, sizeof *m);
    if (qms_gguf_open(&m->g, path) != 0) return -1;
    if (read_cfg(m) != 0) goto fail;
    m->streamed = o->stream_experts;
    m->pf_depth = o->prefetch_depth;
    m->pf_extra = o->prefetch_extra;
    m->overlap = !o->no_overlap;
    m->pf_min_p = o->prefetch_min_p;
    if (memory_plan(m, o) != 0) goto fail;
    const int ctx = o->ctx;
    const QmsCfg *c = &m->c;
    const int E = c->n_embd, D = c->head_dim, H = c->n_head, Hk = c->n_head_kv;

    m->eptr = (const unsigned char **)track(m, calloc(3 * (size_t)c->n_expert, sizeof(*m->eptr)));
    if (!m->eptr) goto fail;
    m->L = (QmsLayer *)track(m, calloc((size_t)c->n_layer, sizeof(QmsLayer)));
    if (!m->L) goto fail;
    if (m->tr.stream_embd ? load_mat_shape(m, "token_embd.weight", c->vocab, E, &m->embd)
                          : load_mat(m, "token_embd.weight", c->vocab, E, &m->embd)) goto fail;
    if (load_f32_vec(m, "output_norm.weight", E, &m->out_norm)) goto fail;
    if (qms_gguf_find(&m->g, "output.weight")) {
        if (load_mat(m, "output.weight", c->vocab, E, &m->output)) goto fail;
    } else {
        m->output = m->embd;                      /* tied embeddings */
    }

    char n[QMS_MAX_NAME];
    for (int l = 0; l < c->n_layer; l++) {
        QmsLayer *L = &m->L[l];
#define NM(s) (snprintf(n, sizeof n, "blk.%d." s, l), n)
        if (load_f32_vec(m, NM("attn_norm.weight"), E, &L->attn_norm)) goto fail;
        if (load_f32_vec(m, NM("attn_q_norm.weight"), D, &L->q_norm)) goto fail;
        if (load_f32_vec(m, NM("attn_k_norm.weight"), D, &L->k_norm)) goto fail;
        if (load_f32_vec(m, NM("ffn_norm.weight"), E, &L->ffn_norm)) goto fail;
        {
            int (*ld)(QmsModel *, const char *, int64_t, int64_t, QmsMat *) =
                l < m->tr.n_pin ? load_mat : load_mat_shape;   /* streamed layers: shape only */
            if (ld(m, NM("attn_q.weight"), (int64_t)H * D, E, &L->q)) goto fail;
            if (ld(m, NM("attn_k.weight"), (int64_t)Hk * D, E, &L->k)) goto fail;
            if (ld(m, NM("attn_v.weight"), (int64_t)Hk * D, E, &L->v)) goto fail;
            if (ld(m, NM("attn_output.weight"), E, (int64_t)H * D, &L->o)) goto fail;
            if (att_locate(m, l, l >= m->tr.n_pin)) goto fail;
        }
        if (load_mat(m, NM("ffn_gate_inp.weight"), c->n_expert, E, &L->router)) goto fail;
        if (load_experts(m, NM("ffn_gate_exps.weight"), c->n_ff_exp, E, c->n_expert, &L->gate)) goto fail;
        if (load_experts(m, NM("ffn_up_exps.weight"), c->n_ff_exp, E, c->n_expert, &L->up)) goto fail;
        if (load_experts(m, NM("ffn_down_exps.weight"), E, c->n_ff_exp, c->n_expert, &L->down)) goto fail;
#undef NM
    }

    m->ctx = ctx;
    const size_t kv = (size_t)c->n_layer * ctx * Hk * D;
    m->kc = (float *)track(m, calloc(kv, sizeof(float)));
    m->vc = (float *)track(m, calloc(kv, sizeof(float)));
    if (!m->kc || !m->vc) { fprintf(stderr, "model: cannot allocate KV cache for ctx %d\n", ctx); goto fail; }
    m->tr.slot_layer[0] = m->tr.slot_layer[1] = -1;
    if (m->tr.n_pin < c->n_layer) {
        for (int i = 0; i < 2; i++)
            if (!(m->tr.slot[i] = (unsigned char *)qms_alloc_aligned((size_t)m->tr.slot_bytes))) {
                fprintf(stderr, "model: cannot allocate trunk ring\n");
                goto fail;
            }
    }
    if (m->streamed) {
        if (o->direct && qms_gguf_open_direct(&m->g) != 0) goto fail;
        if (ecache_init(m, m->plan_cache_bytes, o->cache_policy) != 0) goto fail;
    }
    return 0;
fail:
    qms_model_free(m);
    return -1;
}

void qms_model_free(QmsModel *m)
{
    if (m->trace) fclose(m->trace);
    trunk_drain(m);   /* never free a slot a read is still writing into */
    for (int i = 0; i < 2; i++) qms_free_aligned(m->tr.slot[i]);
    ecache_free(m);
    for (int i = 0; i < m->n_allocs; i++) free(m->allocs[i]);
    free(m->allocs);
    qms_gguf_close(&m->g);
    memset(m, 0, sizeof *m);
}

/* ------------------------------------------------------------ ops ---- */

static void rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const double inv = 1.0 / sqrt(ss / n + (double)eps);
    for (int i = 0; i < n; i++) y[i] = (float)((double)x[i] * inv) * w[i];
}

/* NEOX rotation: element i pairs with i + D/2. inv_freq[i] = base^(-2i/D), in double. */
static void rope_neox(float *x, int D, int pos, const double *inv_freq)
{
    const int half = D / 2;
    for (int i = 0; i < half; i++) {
        const double ang = (double)pos * inv_freq[i];
        const float c = (float)cos(ang), s = (float)sin(ang);
        const float x1 = x[i], x2 = x[i + half];
        x[i]        = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

static float silu(float a) { return a / (1.0f + expf(-a)); }

/* ------------------------------------------------------------ forward ---- */

static void attention(QmsModel *m, int l, const float *xn, int T, int pos0, float *out)
{
    const QmsCfg *c = &m->c;
    const QmsLayer *L = &m->L[l];
    const int H = c->n_head, Hk = c->n_head_kv, D = c->head_dim, group = H / Hk;
    const int QD = H * D, KD = Hk * D;
    float *q = (float *)malloc((size_t)T * QD * sizeof(float));
    float *k = (float *)malloc((size_t)T * KD * sizeof(float));
    float *v = (float *)malloc((size_t)T * KD * sizeof(float));
    float *att = (float *)malloc((size_t)T * QD * sizeof(float));
    double tp = PROF_NOW();
    {
        const QmsMat *Ws[3] = { &L->q, &L->k, &L->v };
        float *ys[3] = { q, k, v };
        qms_matmul_n(Ws, 3, xn, T, ys);
    }
    PROF_ADD(QMS_PROF_QKV, tp);

    double inv_freq[QMS_MAX_HEAD_DIM / 2];
    for (int i = 0; i < D / 2; i++) inv_freq[i] = pow(c->rope_base, -2.0 * i / D);
    float *kc = m->kc + (size_t)l * m->ctx * KD;
    float *vc = m->vc + (size_t)l * m->ctx * KD;
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            float *qh = q + (size_t)t * QD + (size_t)h * D;
            rmsnorm(qh, qh, L->q_norm, D, c->eps);
            rope_neox(qh, D, pos0 + t, inv_freq);
        }
        for (int h = 0; h < Hk; h++) {
            float *kh = k + (size_t)t * KD + (size_t)h * D;
            rmsnorm(kh, kh, L->k_norm, D, c->eps);
            rope_neox(kh, D, pos0 + t, inv_freq);
        }
        memcpy(kc + (size_t)(pos0 + t) * KD, k + (size_t)t * KD, (size_t)KD * sizeof(float));
        memcpy(vc + (size_t)(pos0 + t) * KD, v + (size_t)t * KD, (size_t)KD * sizeof(float));
    }

    const double scale = 1.0 / sqrt((double)D);
    const int par = (int64_t)T * H * (pos0 + T) * D >= (1 << 18);   /* else a fork costs more */
#pragma omp parallel for collapse(2) schedule(dynamic) if (par)
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            const int npos = pos0 + t + 1;               /* causal: positions 0..pos0+t */
            const float *qh = q + (size_t)t * QD + (size_t)h * D;
            const int kh = h / group;
            double *sc = (double *)malloc((size_t)npos * sizeof(double));
            double mx = -INFINITY;
            for (int p = 0; p < npos; p++) {
                const float *kp = kc + (size_t)p * KD + (size_t)kh * D;
                const float dot = qms_vec_dot_f32(qh, kp, D);
                sc[p] = (double)dot * scale;
                if (sc[p] > mx) mx = sc[p];
            }
            double z = 0.0;
            for (int p = 0; p < npos; p++) { sc[p] = exp(sc[p] - mx); z += sc[p]; }
            float *oh = att + (size_t)t * QD + (size_t)h * D;
            for (int d = 0; d < D; d++) oh[d] = 0.0f;
            for (int p = 0; p < npos; p++) {
                const float w = (float)(sc[p] / z);
                const float *vp = vc + (size_t)p * KD + (size_t)kh * D;
                for (int d = 0; d < D; d++) oh[d] += w * vp[d];
            }
            free(sc);
        }
    }
    PROF_ADD(QMS_PROF_ATTN, tp);
    qms_matmul(&L->o, att, T, out);
    PROF_ADD(QMS_PROF_O, tp);
    free(q); free(k); free(v); free(att);
}

static QmsMat expert_mat(const QmsModel *m, const QmsExperts *x, int which, int e)
{
    QmsMat r = x->m;
    r.data = m->eptr[(size_t)which * m->c.n_expert + e];
    return r;
}

/* ------------------------------------------------------------ expert cache ---- */
/* A fixed arena of slots, each holding one expert (gate, up, down slices). Slot states as in
 * kimi-k3-in-c: EMPTY, INFLIGHT (reserved, being read), or holding key = layer * n_expert + e.
 * A fetch reserves victims serially, submits every read at once (sorted by file offset),
 * waits, then publishes. Experts of the current group carry the current tick and are never
 * chosen as victims. */
#define SLOT_EMPTY (-1)
#define SLOT_INFLIGHT (-2)

static uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static int ecache_init(QmsModel *m, uint64_t budget, int policy)
{
    QmsECache *C = &m->cache;
    const QmsCfg *c = &m->c;
    for (int k = 0; k < 3; k++) C->span[k] = 0;
    for (int l = 0; l < c->n_layer; l++) {
        const QmsExperts *W[3] = { &m->L[l].gate, &m->L[l].up, &m->L[l].down };
        for (int k = 0; k < 3; k++) {
            const uint64_t s = m->g.direct ? round_up(qms_direct_span(W[k]->expert_bytes), QMS_DIRECT_ALIGN)
                                           : round_up(W[k]->expert_bytes, 64);
            if (s > C->span[k]) C->span[k] = s;
        }
    }
    C->slot_bytes = C->span[0] + C->span[1] + C->span[2];
    C->nslot = (int)(budget / C->slot_bytes);
    if (C->nslot < c->n_used) C->nslot = c->n_used;     /* one token's experts must fit */
    if (C->nslot > c->n_layer * c->n_expert) C->nslot = c->n_layer * c->n_expert;
    C->arena = (unsigned char *)qms_alloc_aligned((size_t)C->slot_bytes * C->nslot);
    C->key = (int32_t *)malloc((size_t)C->nslot * sizeof(int32_t));
    C->used = (uint64_t *)calloc((size_t)C->nslot, sizeof(uint64_t));
    C->ptr = (const unsigned char **)calloc((size_t)C->nslot * 3, sizeof(*C->ptr));
    C->where = (int32_t *)malloc((size_t)c->n_layer * c->n_expert * sizeof(int32_t));
    C->score = (double *)calloc((size_t)c->n_layer * c->n_expert, sizeof(double));
    C->score_step = (int *)calloc((size_t)c->n_layer * c->n_expert, sizeof(int));
    C->inflight = (int32_t *)malloc((size_t)c->n_layer * c->n_expert * sizeof(int32_t));
    C->slot_left = (int *)calloc((size_t)C->nslot, sizeof(int));
    C->pf_mark = (char *)calloc((size_t)C->nslot, 1);
    C->pf = (QmsPending *)calloc((size_t)C->nslot * 3, sizeof(QmsPending));
    C->pf_act = (int *)malloc((size_t)C->nslot * 3 * sizeof(int));
    C->pf_free = (int *)malloc((size_t)C->nslot * 3 * sizeof(int));
    C->policy = policy;
    C->half_life = 8.0;   /* tokens; chosen by trace replay (tools/sim_cache.py, bench/traces) */
    if (!C->arena || !C->key || !C->used || !C->ptr || !C->where || !C->score || !C->score_step ||
        !C->inflight || !C->slot_left || !C->pf_mark || !C->pf || !C->pf_act || !C->pf_free) {
        fprintf(stderr, "model: cannot allocate expert cache (%d slots x %.1f MB)\n", C->nslot,
                C->slot_bytes / 1e6);
        return -1;
    }
    for (int i = 0; i < C->nslot; i++) C->key[i] = SLOT_EMPTY;
    for (int i = 0; i < c->n_layer * c->n_expert; i++) C->where[i] = C->inflight[i] = -1;
    for (int i = 0; i < C->nslot * 3; i++) C->pf_free[i] = i;
    C->nfree = C->nslot * 3;
    fprintf(stderr, "expert cache: %d slots x %.2f MB = %.2f GB (%s reads, %s eviction)\n", C->nslot,
            C->slot_bytes / 1e6, (double)C->slot_bytes * C->nslot / 1e9,
            m->g.direct ? "unbuffered" : "buffered", policy ? "decayed-LFU" : "LRU");
    return 0;
}

static void ecache_free(QmsModel *m)
{
    QmsECache *C = &m->cache;
    if (C->npf) pf_drain(m);   /* never free memory a read is still writing into */
    qms_free_aligned(C->arena);
    free(C->key); free(C->used); free(C->ptr); free(C->where);
    free(C->score); free(C->score_step);
    free(C->inflight); free(C->slot_left); free(C->pf_mark); free(C->pf);
    free(C->pf_act); free(C->pf_free);
    memset(C, 0, sizeof *C);
}

/* Decayed request frequency of a key at token `step`: halves every half_life tokens. */
static double key_score(const QmsECache *C, int key, int step)
{
    return C->score[key] * pow(0.5, (double)(step - C->score_step[key]) / C->half_life);
}

static void key_touch(QmsECache *C, int key, int step)
{
    C->score[key] = key_score(C, key, step) + 1.0;
    C->score_step[key] = step;
}

/* LRU: oldest use. Decayed LFU: lowest decayed frequency (oldest use breaks ties).
 * Never a slot being read, never one the current group needs; an empty slot always wins.
 * LRU over a token's 752 experts is a cyclic scan that evicts everything before reuse
 * (0% hits below ~16 GB on 235B traces); decayed LFU keeps the experts this text keeps using. */
static int pick_victim(const QmsECache *C, uint64_t now, int step)
{
    int best = -1;
    uint64_t oldest = UINT64_MAX;
    double low = INFINITY;
    for (int i = 0; i < C->nslot; i++) {
        if (C->key[i] == SLOT_INFLIGHT) continue;          /* being read right now */
        if (C->key[i] == SLOT_EMPTY) return i;             /* free: always the cheapest choice */
        if (C->used[i] >= now) continue;                    /* needed by the current group */
        if (C->policy == 0) {
            if (C->used[i] < oldest) { oldest = C->used[i]; best = i; }
        } else {
            const double s = key_score(C, C->key[i], step);
            if (s < low || (s == low && C->used[i] < oldest)) { low = s; oldest = C->used[i]; best = i; }
        }
    }
    return best;
}

/* ------------------------------------------------------------ reads ---- */

static int cmp_pending(const void *a, const void *b)
{
    const QmsPending *x = (const QmsPending *)a, *y = (const QmsPending *)b;
    if (x->shard != y->shard) return x->shard - y->shard;
    return x->file_off < y->file_off ? -1 : x->file_off > y->file_off;
}

/* Describe the read of slice k (gate/up/down) of expert e into slot s. */
static void make_read(const QmsModel *m, const QmsExperts *W, int k, int e, int s, int key, QmsPending *p)
{
    const QmsECache *C = &m->cache;
    const QmsTensor *t = W->t;
    const uint64_t n = W->expert_bytes;
    const uint64_t off = t->off + (uint64_t)e * n;
    uint64_t sub = 0;
    for (int j = 0; j < k; j++) sub += C->span[j];
    p->shard = t->shard;
    p->file_off = off;
    p->slot = s;
    p->k = k;
    p->key = key;
    p->done = 0;
    memset(&p->io, 0, sizeof p->io);
    p->io.buf = C->arena + (uint64_t)s * C->slot_bytes + sub;
    /* Unbuffered windows are rounded out to 4096; one that would cross end of file (the last
     * tensor of a shard) is read through the buffered handle instead, exact offset and length,
     * because Windows reports such tail reads inconsistently (short count or EOF status). */
    if (m->g.direct && round_up(off + n, QMS_DIRECT_ALIGN) <= m->g.file_size[t->shard]) {
        const uint64_t a0 = off & ~(QMS_DIRECT_ALIGN - 1);
        p->io.fd = m->g.fd_direct[t->shard];
        p->io.off = a0;
        p->io.len = (size_t)(round_up(off + n, QMS_DIRECT_ALIGN) - a0);
        p->io.min_ok = (size_t)(off + n - a0);
        p->want = off - a0;
    } else {
        p->io.fd = m->g.fd[t->shard];
        p->io.off = off;
        p->io.len = (size_t)n;
        p->io.min_ok = (size_t)n;
        p->want = 0;
    }
}

static void reserve_slot(QmsECache *C, int s, uint64_t used)
{
    if (C->key[s] >= 0) C->where[C->key[s]] = -1;
    C->key[s] = SLOT_INFLIGHT;
    C->used[s] = used;
    C->pf_mark[s] = 0;
}

/* Finish one prefetch read (blocking if still in flight); publish its slot when all three
 * slices have arrived. */
static int pf_complete(QmsModel *m, QmsPending *p)
{
    QmsECache *C = &m->cache;
    if (p->done) return 0;
    const double t0 = qms_now();
    const int rc = qms_aio_wait(&p->io);
    m->ex_read_s += qms_now() - t0;
    p->done = 1;
    if (rc) {
        fprintf(stderr, "FATAL: prefetch read failed (key %d)\n", p->key);
        return -1;
    }
    C->ptr[(size_t)p->slot * 3 + p->k] = (const unsigned char *)p->io.buf + p->want;
    if (--C->slot_left[p->slot] == 0) {
        C->key[p->slot] = p->key;
        C->where[p->key] = p->slot;
        C->inflight[p->key] = -1;
        if (C->pf_mark[p->slot] != 2) C->pf_mark[p->slot] = 1;   /* 2: already counted while in flight */
        else C->pf_mark[p->slot] = 0;
    }
    return 0;
}

/* Drop finished requests from the active list; their pool entries become free.
 * Only the int index lists are reordered: pool entries never move. */
static void pf_compact(QmsECache *C)
{
    int j = 0;
    for (int i = 0; i < C->npf; i++) {
        const int r = C->pf_act[i];
        if (C->pf[r].done) C->pf_free[C->nfree++] = r;
        else C->pf_act[j++] = r;
    }
    C->npf = j;
}

/* Publish every prefetch that has already finished, without blocking. */
static int pf_harvest(QmsModel *m)
{
    QmsECache *C = &m->cache;
    for (int i = 0; i < C->npf; i++) {
        QmsPending *p = &C->pf[C->pf_act[i]];
        if (!p->done && qms_aio_poll(&p->io) && pf_complete(m, p) != 0) return -1;
    }
    pf_compact(C);
    return 0;
}

/* Block until the prefetch of `key` has landed. */
static int pf_wait_key(QmsModel *m, int key)
{
    QmsECache *C = &m->cache;
    const int s = C->inflight[key];
    for (int i = 0; i < C->npf && s >= 0; i++) {
        QmsPending *p = &C->pf[C->pf_act[i]];
        if (!p->done && p->slot == s && pf_complete(m, p) != 0) return -1;
    }
    C->pf_waited++;
    return 0;
}

static void pf_drain(QmsModel *m)
{
    QmsECache *C = &m->cache;
    for (int i = 0; i < C->npf; i++) {
        QmsPending *p = &C->pf[C->pf_act[i]];
        if (!p->done) pf_complete(m, p);
    }
    pf_compact(C);
}

static const QmsPending *g_sort_pool;   /* qsort has no context argument */
static int cmp_pool_idx(const void *a, const void *b)
{
    return cmp_pending(&g_sort_pool[*(const int *)a], &g_sort_pool[*(const int *)b]);
}

/* Predict the experts of layers l+1 .. l+depth for the single decode token whose residual
 * (after layer l's attention) is h: apply each upcoming layer's own FFN norm and router to
 * h, take the top (k + extra), and start reading the ones not cached or already in flight.
 * Predicted slots are protected until their layer has run (used = now + d). A wrong guess
 * costs bandwidth, never correctness: experts are only ever computed from what routing chose. */
static int prefetch_issue(QmsModel *m, int l, const float *h)
{
    const QmsCfg *c = &m->c;
    QmsECache *C = &m->cache;
    const int E = c->n_embd, X = c->n_expert;
    if (X <= 0 || E <= 0) return 0;
    const int want = c->n_used + m->pf_extra < X ? c->n_used + m->pf_extra : X;
    const uint64_t now = C->tick;
    float *xn = (float *)malloc((size_t)E * sizeof(float));
    float *lg = (float *)malloc((size_t)X * sizeof(float));
    char *pick = (char *)malloc((size_t)X);
    int *batch = (int *)malloc((size_t)want * 3 * sizeof(int));
    int rc = 0, stop = 0;
    pf_compact(C);   /* every reserved slot below needs 3 free pool entries */
    for (int d = 1; d <= m->pf_depth && !stop && l + d < c->n_layer; d++) {
        const QmsLayer *L2 = &m->L[l + d];
        if (L2->router.type != QMS_T_F32) break;
        rmsnorm(xn, h, L2->ffn_norm, E, c->eps);
        for (int e = 0; e < X; e++)
            lg[e] = qms_vec_dot_f32((const float *)(L2->router.data + (uint64_t)e * L2->router.row_bytes), xn, E);
        memset(pick, 0, (size_t)X);
        /* predicted routing probabilities (the same softmax real routing uses) */
        double lmax = -INFINITY, z = 0.0;
        for (int e = 0; e < X; e++) if (lg[e] > lmax) lmax = lg[e];
        for (int e = 0; e < X; e++) z += exp((double)lg[e] - lmax);
        int nb = 0;
        for (int j = 0; j < want && !stop; j++) {
            int best = -1;
            for (int e = 0; e < X; e++) if (!pick[e] && (best < 0 || lg[e] > lg[best])) best = e;
            pick[best] = 1;
            /* method 3: skip low-confidence guesses; candidates come in descending order */
            if (m->pf_min_p > 0.0 && exp((double)lg[best] - lmax) / z < m->pf_min_p) break;
            const int key = (l + d) * X + best;
            int s = C->where[key];
            if (s >= 0) { if (C->used[s] < now + (uint64_t)d) C->used[s] = now + (uint64_t)d; continue; }
            if (C->inflight[key] >= 0) continue;
            if (C->nfree < 3) { stop = 1; break; }
            s = pick_victim(C, now, m->trace_step);
            if (s < 0) { stop = 1; break; }          /* cache full of protected slots: stop guessing */
            reserve_slot(C, s, now + (uint64_t)d);
            C->inflight[key] = s;
            C->slot_left[s] = 3;
            C->pf_issued++;
            const QmsExperts *W[3] = { &L2->gate, &L2->up, &L2->down };
            for (int k = 0; k < 3; k++) {
                const int r = C->pf_free[--C->nfree];
                make_read(m, W[k], k, best, s, key, &C->pf[r]);
                batch[nb++] = r;
            }
        }
        if (nb) {   /* submit in disk order; pool entries stay where they are */
            g_sort_pool = C->pf;
            qsort(batch, (size_t)nb, sizeof *batch, cmp_pool_idx);
            for (int i = 0; i < nb; i++) {
                QmsPending *p = &C->pf[batch[i]];
                if (qms_aio_submit(&p->io) != 0) rc = -1;
                m->ex_bytes_read += p->io.len;
                C->pf_act[C->npf++] = batch[i];
            }
            m->ex_reads += (uint64_t)nb;
        }
    }
    free(xn); free(lg); free(pick); free(batch);
    return rc;
}

/* A layer group being fetched. Demand reads are kept per expert (3 consecutive entries), so
 * each expert can be used as soon as its own three slices have landed. */
typedef struct {
    int ng;
    const int *grp;
    int *slot_of_grp;
    char *state;          /* per expert: 0 ready (hit), 1 demand miss in flight, 2 prefetch in flight */
    int *first;           /* demand miss: index of its first read in pend */
    QmsPending *pend;
    int np;
} FetchJob;

enum { FJ_READY = 0, FJ_MISS = 1, FJ_PF = 2 };

static void fetch_publish(QmsModel *m, int l, const FetchJob *J, int i)
{
    const int X = m->c.n_expert, s = J->slot_of_grp[i];
    (void)l;
    for (int k = 0; k < 3; k++) m->eptr[(size_t)k * X + J->grp[i]] = m->cache.ptr[(size_t)s * 3 + k];
}

/* Classify, reserve slots, submit this group's misses expert by expert, then issue any
 * prefetch for later layers. Nothing is waited for here. */
static int fetch_begin(QmsModel *m, int l, const int *grp, int ng, const float *h_pf, FetchJob *J)
{
    const QmsCfg *c = &m->c;
    const QmsLayer *L = &m->L[l];
    const int X = c->n_expert;
    const QmsExperts *W[3] = { &L->gate, &L->up, &L->down };
    memset(J, 0, sizeof *J);
    J->ng = ng;
    J->grp = grp;
    J->slot_of_grp = (int *)malloc((size_t)ng * sizeof(int));
    J->state = (char *)calloc((size_t)ng, 1);
    J->first = (int *)malloc((size_t)ng * sizeof(int));
    J->pend = (QmsPending *)malloc((size_t)ng * 3 * sizeof(QmsPending));
    if (!m->streamed) {
        for (int i = 0; i < ng; i++)
            for (int k = 0; k < 3; k++)
                m->eptr[(size_t)k * X + grp[i]] = W[k]->m.data + (uint64_t)grp[i] * W[k]->expert_bytes;
        return 0;
    }
    QmsECache *C = &m->cache;
    const uint64_t now = ++C->tick;
    int rc = 0;
    if (C->npf && pf_harvest(m) != 0) rc = -1;        /* finished prefetches become plain hits */
    for (int i = 0; i < ng; i++) key_touch(C, l * X + grp[i], m->trace_step);
    /* stamp hits and in-flight prefetches first, so no miss can evict what this group needs */
    for (int i = 0; i < ng; i++) {
        const int key = l * X + grp[i];
        int s = C->where[key];
        J->state[i] = FJ_READY;
        if (s < 0 && C->inflight[key] >= 0) { s = C->inflight[key]; J->state[i] = FJ_PF; }
        J->slot_of_grp[i] = s;
        J->first[i] = -1;
        if (s >= 0) {
            C->used[s] = now;
            C->hits++;
            if (J->state[i] == FJ_PF) { C->pf_used++; C->pf_mark[s] = 2; }      /* count once; publish clears */
            else if (C->pf_mark[s] == 1) { C->pf_used++; C->pf_mark[s] = 0; }
        }
    }
    for (int i = 0; i < ng && rc == 0; i++) {
        if (J->slot_of_grp[i] >= 0) continue;
        const int s = pick_victim(C, now, m->trace_step);
        if (s < 0) { fprintf(stderr, "FATAL: expert cache has no free slot (%d slots)\n", C->nslot); rc = -1; break; }
        reserve_slot(C, s, now);
        C->misses++;
        J->slot_of_grp[i] = s;
        J->state[i] = FJ_MISS;
        J->first[i] = J->np;
        for (int k = 0; k < 3; k++) make_read(m, W[k], k, grp[i], s, l * X + grp[i], &J->pend[J->np++]);
    }
    /* submit expert by expert (grp is in ascending expert order), so experts complete one
     * after another instead of all at the end; the device still sees every request at once */
    int fail = 0;
    for (int i = 0; i < J->np && rc == 0; i++) {
        fail |= qms_aio_submit(&J->pend[i].io) != 0;
        m->ex_bytes_read += J->pend[i].io.len;
    }
    m->ex_reads += (uint64_t)J->np;
    if (fail) rc = -1;
    if (rc == 0 && h_pf && m->pf_depth > 0 && prefetch_issue(m, l, h_pf) != 0) rc = -1;
    for (int i = 0; i < ng; i++) if (J->state[i] == FJ_READY) fetch_publish(m, l, J, i);
    return rc;
}

/* Is expert i of the group usable? block = 0: only check; block = 1: wait for it.
 * Returns 1 when ready (and published to m->eptr), 0 when still in flight, -1 on error. */
static int fetch_ready(QmsModel *m, int l, FetchJob *J, int i, int block)
{
    QmsECache *C = &m->cache;
    if (J->state[i] == FJ_READY) return 1;
    if (J->state[i] == FJ_MISS) {
        QmsPending *p = &J->pend[J->first[i]];
        if (!block) {
            for (int k = 0; k < 3; k++) if (!qms_aio_poll(&p[k].io)) return 0;
        }
        const double t0 = qms_now();
        int fail = 0;
        for (int k = 0; k < 3; k++) {
            fail |= qms_aio_wait(&p[k].io) != 0;
            C->ptr[(size_t)p[k].slot * 3 + p[k].k] = (const unsigned char *)p[k].io.buf + p[k].want;
        }
        m->ex_read_s += qms_now() - t0;
        if (fail) {
            fprintf(stderr, "FATAL: routed expert read failed in layer %d; output would be corrupt.\n", l);
            return -1;
        }
    } else {   /* FJ_PF: a prefetch started earlier is still landing in this slot */
        const int s = J->slot_of_grp[i];
        if (!block) {
            for (int a = 0; a < C->npf; a++) {
                QmsPending *p = &C->pf[C->pf_act[a]];
                if (!p->done && p->slot == s && !qms_aio_poll(&p->io)) return 0;
            }
        }
        if (pf_wait_key(m, l * m->c.n_expert + J->grp[i]) != 0) return -1;
    }
    J->state[i] = FJ_READY;
    fetch_publish(m, l, J, i);
    return 1;
}

/* Every expert of the group is ready: record the cache keys, release the job. */
static void fetch_end(QmsModel *m, int l, FetchJob *J, int ok)
{
    if (m->streamed && ok) {
        QmsECache *C = &m->cache;
        const int X = m->c.n_expert;
        for (int i = 0; i < J->ng; i++) {
            const int s = J->slot_of_grp[i], key = l * X + J->grp[i];
            C->key[s] = key;
            C->where[key] = s;
            C->inflight[key] = -1;
        }
    }
    if (!ok) for (int i = 0; i < J->np; i++) qms_aio_wait(&J->pend[i].io);   /* never leave reads in flight */
    free(J->slot_of_grp); free(J->state); free(J->first); free(J->pend);
}

/* ------------------------------------------------------------ MoE ---- */
/* Every (token, chosen expert) pair is a "slot" s; slots are grouped by expert so each expert
 * row is touched once for all tokens that chose it. Experts are processed in groups that fit
 * the cache; each group runs gate/up then down as single parallel loops. dn[s] holds the
 * expert output for slot s, and the weighted sum per token runs once at the end in selection
 * order, so grouping, threads and streaming cannot change the result. */
typedef struct {
    int T, NS, na;
    int *start, *slot_t, *slot_of, *active;   /* start[e]..start[e+1]: slots of expert e */
    QmsQ8 *xq, *aq;
    float *gu, *dn;
} MoeWork;

static void moe_group_fast(const QmsModel *m, const QmsLayer *L, const MoeWork *w, const int *grp, int ng)
{
    const QmsCfg *c = &m->c;
    const int E = c->n_embd, X = c->n_expert, F = c->n_ff_exp, nbE = E / 32, nbF = F / 32;
    const unsigned char *const *ep = m->eptr;
    /* one parallel region, three phases separated by the implicit barriers of the loops:
     * computing experts in small batches as they land (overlap) made three fork/joins per
     * batch measurable (+150 ms/token on 235B); one region per batch cuts that by 3x */
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int64_t it = 0; it < (int64_t)ng * 2 * F; it++) {
            const int e = grp[it / (2 * F)];
            const int r = (int)(it % (2 * F));
            const QmsExperts *W = r < F ? &L->gate : &L->up;
            const int rr = r < F ? r : r - F;
            const unsigned char *row = ep[(size_t)(r < F ? 0 : 1) * X + e] + (uint64_t)rr * W->m.row_bytes;
            for (int s = w->start[e]; s < w->start[e + 1]; s++)
                w->gu[(size_t)s * 2 * F + r] = qms_vec_dot_q8(W->m.type, row, w->xq + (size_t)w->slot_t[s] * nbE, E);
        }
#pragma omp for schedule(static)
        for (int i = 0; i < ng; i++) {
            const int e = grp[i];
            for (int s = w->start[e]; s < w->start[e + 1]; s++) {
                float *a = w->gu + (size_t)s * 2 * F;
                for (int j = 0; j < F; j++) a[j] = silu(a[j]) * a[F + j];
                qms_quantize_q8(a, F, w->aq + (size_t)s * nbF);
            }
        }
#pragma omp for schedule(static)
        for (int64_t it = 0; it < (int64_t)ng * E; it++) {
            const int e = grp[it / E];
            const int r = (int)(it % E);
            const unsigned char *row = ep[(size_t)2 * X + e] + (uint64_t)r * L->down.m.row_bytes;
            for (int s = w->start[e]; s < w->start[e + 1]; s++)
                w->dn[(size_t)s * E + r] = qms_vec_dot_q8(L->down.m.type, row, w->aq + (size_t)s * nbF, F);
        }
    }
}

/* reference path: one token-expert pair at a time, fp32 via qms_matmul */
static void moe_group_exact(const QmsModel *m, const QmsLayer *L, const MoeWork *w, const float *xn,
                            const int *grp, int ng)
{
    const QmsCfg *c = &m->c;
    const int E = c->n_embd, F = c->n_ff_exp;
    float *g = (float *)malloc((size_t)F * sizeof(float));
    float *u = (float *)malloc((size_t)F * sizeof(float));
    for (int i = 0; i < ng; i++) {
        const int e = grp[i];
        QmsMat Wg = expert_mat(m, &L->gate, 0, e), Wu = expert_mat(m, &L->up, 1, e),
               Wd = expert_mat(m, &L->down, 2, e);
        for (int s = w->start[e]; s < w->start[e + 1]; s++) {
            const float *xt = xn + (size_t)w->slot_t[s] * E;
            qms_matmul(&Wg, xt, 1, g);
            qms_matmul(&Wu, xt, 1, u);
            for (int j = 0; j < F; j++) g[j] = silu(g[j]) * u[j];
            qms_matmul(&Wd, g, 1, w->dn + (size_t)s * E);
        }
    }
    free(g); free(u);
}

static int moe(QmsModel *m, int l, const float *xn, const float *h, int T, float *out)
{
    const QmsCfg *c = &m->c;
    const QmsLayer *L = &m->L[l];
    const int E = c->n_embd, X = c->n_expert, F = c->n_ff_exp, K = c->n_used;
    const int NS = T * K;
    float *logits = (float *)malloc((size_t)T * X * sizeof(float));
    double *prob = (double *)malloc((size_t)X * sizeof(double));
    int *selv = (int *)malloc((size_t)NS * sizeof(int));
    float *wtv = (float *)malloc((size_t)NS * sizeof(float));
    double tp = PROF_NOW();
    qms_matmul(&L->router, xn, T, logits);

    for (int t = 0; t < T; t++) {
        const float *lt = logits + (size_t)t * X;
        double mx = -INFINITY, z = 0.0;
        for (int e = 0; e < X; e++) if (lt[e] > mx) mx = lt[e];
        for (int e = 0; e < X; e++) { prob[e] = exp((double)lt[e] - mx); z += prob[e]; }
        for (int e = 0; e < X; e++) prob[e] /= z;
        int *sel = selv + (size_t)t * K;
        double wsum = 0.0;
        for (int j = 0; j < K; j++) {                 /* top-k; ties go to the lower index */
            int best = -1;
            for (int e = 0; e < X; e++) {
                int taken = 0;
                for (int i = 0; i < j; i++) if (sel[i] == e) { taken = 1; break; }
                if (!taken && (best < 0 || prob[e] > prob[best])) best = e;
            }
            sel[j] = best;
            wsum += prob[best];
        }
        for (int j = 0; j < K; j++) wtv[(size_t)t * K + j] = (float)(prob[sel[j]] / wsum);
    }

    /* group slots by expert */
    MoeWork w;
    memset(&w, 0, sizeof w);
    w.T = T;
    w.NS = NS;
    w.start = (int *)calloc((size_t)X + 1, sizeof(int));
    w.slot_t = (int *)malloc((size_t)NS * sizeof(int));
    w.slot_of = (int *)malloc((size_t)NS * sizeof(int));
    w.active = (int *)malloc((size_t)X * sizeof(int));
    int *cnt = (int *)calloc((size_t)X, sizeof(int));
    for (int i = 0; i < NS; i++) cnt[selv[i]]++;
    for (int e = 0; e < X; e++) {
        w.start[e + 1] = w.start[e] + cnt[e];
        if (cnt[e]) w.active[w.na++] = e;
    }
    memset(cnt, 0, (size_t)X * sizeof(int));
    for (int t = 0; t < T; t++)
        for (int j = 0; j < K; j++) {
            const int e = selv[t * K + j], s = w.start[e] + cnt[e]++;
            w.slot_t[s] = t;
            w.slot_of[t * K + j] = s;
        }
    free(cnt);

    const int fast = !qms_exact_math && qms_fast_supported(L->gate.m.type) &&
                     qms_fast_supported(L->up.m.type) && qms_fast_supported(L->down.m.type) &&
                     L->gate.m.type != QMS_T_F32 && L->down.m.type != QMS_T_F32 &&
                     E % 256 == 0 && F % 32 == 0;
    w.dn = (float *)malloc((size_t)NS * E * sizeof(float));
    if (fast) {
        w.xq = (QmsQ8 *)malloc((size_t)T * (E / 32) * sizeof(QmsQ8));
        for (int t = 0; t < T; t++) qms_quantize_q8(xn + (size_t)t * E, E, w.xq + (size_t)t * (E / 32));
        w.gu = (float *)malloc((size_t)NS * 2 * F * sizeof(float));
        w.aq = (QmsQ8 *)malloc((size_t)NS * (F / 32) * sizeof(QmsQ8));
    }
    PROF_ADD(QMS_PROF_ROUTER, tp);
    if (m->trace) {   /* one record per distinct expert this layer needs, in expert order */
        for (int i = 0; i < w.na; i++) {
            const int32_t st = m->trace_step;
            const int16_t ly = (int16_t)l, ex = (int16_t)w.active[i];
            fwrite(&st, 4, 1, m->trace);
            fwrite(&ly, 2, 1, m->trace);
            fwrite(&ex, 2, 1, m->trace);
        }
    }

    const int gmax = m->streamed ? m->cache.nslot : w.na;
    int rc = 0;
    for (int g0 = 0; g0 < w.na && rc == 0; g0 += gmax) {
        const int ng = w.na - g0 < gmax ? w.na - g0 : gmax;
        const int *grp = w.active + g0;
        FetchJob J;
        rc = fetch_begin(m, l, grp, ng, T == 1 && ng == w.na ? h : NULL, &J);
        PROF_ADD(QMS_PROF_FETCH, tp);
        /* Overlap: compute every expert that has landed (cache hits first) while the rest
         * are still being read; block only when nothing is ready. Without overlap: wait for
         * the whole group, then compute it at once. Each expert writes its own dn rows and
         * the weighted sum below runs in selection order, so batching cannot change results. */
        char *done = (char *)calloc((size_t)ng, 1);
        int *bat = (int *)malloc((size_t)ng * sizeof(int));
        int left = rc == 0 ? ng : 0;
        while (left > 0 && rc == 0) {
            int nb = 0;
            for (int i = 0; i < ng && rc == 0; i++) {
                if (done[i]) continue;
                const int r = fetch_ready(m, l, &J, i, !m->overlap);
                if (r < 0) rc = -1;
                else if (r) { done[i] = 1; bat[nb++] = grp[i]; }
            }
            if (rc == 0 && nb == 0) {   /* nothing landed yet: wait for the earliest submitted */
                int i = 0;
                while (done[i]) i++;
                if (fetch_ready(m, l, &J, i, 1) < 0) rc = -1;
                else { done[i] = 1; bat[nb++] = grp[i]; }
            }
            PROF_ADD(QMS_PROF_FETCH, tp);
            if (rc) break;
            if (fast) moe_group_fast(m, L, &w, bat, nb);
            else      moe_group_exact(m, L, &w, xn, bat, nb);
            PROF_ADD(QMS_PROF_MOE, tp);
            left -= nb;
        }
        fetch_end(m, l, &J, rc == 0);
        free(done);
        free(bat);
    }
    if (rc == 0) {   /* weighted sum per token, in selection order */
        memset(out, 0, (size_t)T * E * sizeof(float));
        for (int t = 0; t < T; t++) {
            float *ot = out + (size_t)t * E;
            for (int j = 0; j < K; j++) {
                const float wj = wtv[t * K + j];
                const float *d = w.dn + (size_t)w.slot_of[t * K + j] * E;
                for (int i = 0; i < E; i++) ot[i] += wj * d[i];
            }
        }
    }
    free(w.start); free(w.slot_t); free(w.slot_of); free(w.active);
    free(w.xq); free(w.aq); free(w.gu); free(w.dn);
    free(logits); free(prob); free(selv); free(wtv);
    return rc;
}

/* ------------------------------------------------------------ trunk ring ---- */

static int trunk_submit(QmsModel *m, int s, int l)
{
    QmsTrunk *T = &m->tr;
    const QmsLayer *L = &m->L[l];
    const uint64_t off = L->att_off, n = L->att_len;
    QmsIoReq *r = &T->io[s];
    memset(r, 0, sizeof *r);
    r->buf = T->slot[s];
    if (m->g.direct && round_up_u(off + n, QMS_DIRECT_ALIGN) <= m->g.file_size[L->att_shard]) {
        const uint64_t a0 = off & ~(QMS_DIRECT_ALIGN - 1);
        r->fd = m->g.fd_direct[L->att_shard];
        r->off = a0;
        r->len = (size_t)(round_up_u(off + n, QMS_DIRECT_ALIGN) - a0);
        r->min_ok = (size_t)(off + n - a0);
        T->want[s] = off - a0;
    } else {
        r->fd = m->g.fd[L->att_shard];
        r->off = off;
        r->len = (size_t)n;
        r->min_ok = (size_t)n;
        T->want[s] = 0;
    }
    T->slot_layer[s] = l;
    T->pending[s] = 1;
    T->bytes_read += r->len;
    T->reads++;
    if (qms_aio_submit(r) != 0) {
        fprintf(stderr, "FATAL: trunk read for layer %d could not be submitted\n", l);
        return -1;
    }
    return 0;
}

static int trunk_wait(QmsModel *m, int s)
{
    QmsTrunk *T = &m->tr;
    if (!T->pending[s]) return 0;
    const double t0 = qms_now();
    const int rc = qms_aio_wait(&T->io[s]);
    T->wait_s += qms_now() - t0;
    T->pending[s] = 0;
    if (rc) {
        fprintf(stderr, "FATAL: trunk read for layer %d failed\n", T->slot_layer[s]);
        T->slot_layer[s] = -1;
    }
    return rc;
}

static void trunk_drain(QmsModel *m)
{
    for (int s = 0; s < 2; s++) trunk_wait(m, s);
}

/* Make layer l's attention weights addressable. Pinned layers already are. A streamed layer
 * is taken from the slot it was prefetched into (or read now), then the next streamed layer
 * is started in the other slot, so the slot being computed on is never overwritten. */
static int trunk_acquire(QmsModel *m, int l)
{
    QmsTrunk *T = &m->tr;
    const int nl_total = m->c.n_layer;
    if (l < T->n_pin) return 0;
    int s = T->slot_layer[0] == l ? 0 : T->slot_layer[1] == l ? 1 : -1;
    if (s < 0) {   /* not prefetched (first use): take the slot that is not the newest */
        s = T->slot_layer[0] < 0 ? 0 : T->slot_layer[1] < 0 ? 1 : 0;
        if (trunk_wait(m, s) || trunk_submit(m, s, l)) return -1;
    }
    if (trunk_wait(m, s)) return -1;
    QmsLayer *L = &m->L[l];
    const unsigned char *b = T->slot[s] + T->want[s];
    L->q.data = b + L->att_rel[0];
    L->k.data = b + L->att_rel[1];
    L->v.data = b + L->att_rel[2];
    L->o.data = b + L->att_rel[3];
    /* next streamed layer: l + 1, or the next token's first streamed layer */
    const int nl = l + 1 < nl_total ? l + 1 : T->n_pin;
    if (nl != l && T->slot_layer[s ^ 1] != nl) {
        if (trunk_wait(m, s ^ 1) || trunk_submit(m, s ^ 1, nl)) return -1;
    }
    return 0;
}

int qms_forward(QmsModel *m, const int *ids, int T, int pos0, float *logits, int all_logits)
{
    const QmsCfg *c = &m->c;
    const int E = c->n_embd;
    m->trace_step++;
    if (pos0 + T > m->ctx) { fprintf(stderr, "forward: context %d exceeded\n", m->ctx); return -1; }
    float *h = (float *)malloc((size_t)T * E * sizeof(float));
    float *xn = (float *)malloc((size_t)T * E * sizeof(float));
    float *tmp = (float *)malloc((size_t)T * E * sizeof(float));
    if (!h || !xn || !tmp) { free(h); free(xn); free(tmp); return -1; }

    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= c->vocab) { fprintf(stderr, "forward: bad id %d\n", ids[t]); return -1; }
        if (m->tr.stream_embd) {   /* one row from disk: 2.3 KB for 235B */
            const QmsTensor *et = qms_gguf_find(&m->g, "token_embd.weight");
            unsigned char *row = (unsigned char *)malloc((size_t)m->embd.row_bytes);
            const int rc = qms_gguf_read(&m->g, et, (uint64_t)ids[t] * m->embd.row_bytes, m->embd.row_bytes, row);
            if (rc == 0) qms_dequant_row(m->embd.type, row, h + (size_t)t * E, E);
            free(row);
            if (rc) { fprintf(stderr, "FATAL: embedding row read failed\n"); return -1; }
        } else {
            qms_dequant_row(m->embd.type, m->embd.data + (uint64_t)ids[t] * m->embd.row_bytes,
                            h + (size_t)t * E, E);
        }
    }
    for (int l = 0; l < c->n_layer; l++) {
        const QmsLayer *L = &m->L[l];
        for (int t = 0; t < T; t++) rmsnorm(xn + (size_t)t * E, h + (size_t)t * E, L->attn_norm, E, c->eps);
        if (trunk_acquire(m, l) != 0) { free(h); free(xn); free(tmp); return -1; }
        attention(m, l, xn, T, pos0, tmp);
        for (size_t i = 0; i < (size_t)T * E; i++) h[i] += tmp[i];
        for (int t = 0; t < T; t++) rmsnorm(xn + (size_t)t * E, h + (size_t)t * E, L->ffn_norm, E, c->eps);
        if (moe(m, l, xn, h, T, tmp) != 0) { free(h); free(xn); free(tmp); return -1; }
        for (size_t i = 0; i < (size_t)T * E; i++) h[i] += tmp[i];
    }
    const int t0 = all_logits ? 0 : T - 1;
    const int nT = T - t0;
    for (int t = t0; t < T; t++)
        rmsnorm(xn + (size_t)(t - t0) * E, h + (size_t)t * E, m->out_norm, E, c->eps);
    double tp = PROF_NOW();
    qms_matmul(&m->output, xn, nT, logits);
    PROF_ADD(QMS_PROF_OUT, tp);
    free(h); free(xn); free(tmp);
    return 0;
}
