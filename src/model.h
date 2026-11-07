/* model.h - Qwen3-MoE model: config, resident weights, KV cache, forward pass. */
#ifndef QMS_MODEL_H
#define QMS_MODEL_H

#include <stdio.h>

#include "aio.h"
#include "gguf.h"
#include "quant.h"

/* One read of one slice (gate/up/down) of one expert; demand reads and prefetches alike. */
typedef struct {
    int shard;
    uint64_t file_off;
    int slot, k, key;
    uint64_t want;        /* offset of the first wanted byte inside the read window */
    int done;
    QmsIoReq io;
} QmsPending;

#define QMS_MAX_HEAD_DIM 256
#define QMS_MAX_USED 64

typedef struct {
    int n_layer, n_embd, n_head, n_head_kv, head_dim;
    int n_expert, n_used, n_ff_exp, vocab;
    float eps;
    double rope_base;
} QmsCfg;

typedef struct {          /* n_expert stacked matrices of rows x cols */
    QmsMat m;             /* resident: m.data points at expert 0; streamed: m.data == NULL */
    uint64_t expert_bytes;
    const QmsTensor *t;   /* streamed: where the experts live on disk */
} QmsExperts;

typedef struct {
    float *attn_norm, *q_norm, *k_norm, *ffn_norm;
    QmsMat q, k, v, o, router;
    QmsExperts gate, up, down;
    /* the layer's attention tensors form one contiguous block in the file */
    int att_shard;
    uint64_t att_off, att_len;         /* absolute file offset and length of the block */
    uint64_t att_rel[4];               /* q, k, v, o: offset inside the block */
} QmsLayer;

/* Streaming the dense trunk (kimi-k3-in-c's "pinned prefix + ring"): attention weights of
 * layers [0, n_pin) stay resident; later layers are read each token into one of two ring
 * slots, the next one always in flight while the current one computes (the order is fixed,
 * so this prefetch is never wrong). */
typedef struct {
    int n_pin;                 /* == n_layer: trunk fully resident, ring unused */
    int stream_embd;           /* token embedding rows read on demand (8 KB per token) */
    uint64_t slot_bytes;
    unsigned char *slot[2];
    int slot_layer[2];         /* layer held or being read by each slot, -1 if none */
    int pending[2];
    uint64_t want[2];          /* offset of the block's first byte inside the slot */
    QmsIoReq io[2];
    uint64_t bytes_read, reads;
    double wait_s;
} QmsTrunk;

typedef struct {
    int stream_experts;   /* read routed experts from disk on demand instead of loading them */
    int ctx;              /* KV cache capacity in positions */
    int force;            /* start even if the memory plan exceeds 95% of available RAM */
    int direct;           /* streamed reads bypass the OS file cache (unbuffered) */
    double cache_gb;      /* expert cache budget when streaming */
    int cache_policy;     /* 0 = LRU, 1 = decayed LFU (default) */
    int prefetch_depth;   /* layers ahead to predict and prefetch (decode only); 0 = off */
    int prefetch_extra;   /* predict top-(k + extra) experts per layer to raise recall */
    double prefetch_min_p;  /* prefetch only predictions with routing probability >= this */
    double mem_gb;        /* > 0: total memory budget; trunk pinned first, rest to expert cache */
    int pin_layers;       /* >= 0: pin exactly this many attention layers (testing); -1 = auto */
    int no_overlap;       /* 1: wait for a whole layer's experts before computing (A/B baseline) */
} QmsLoadOpts;

typedef struct {          /* routed-expert cache (streaming mode) */
    int nslot;
    uint64_t span[3];     /* bytes reserved per slot for gate / up / down */
    uint64_t slot_bytes;
    unsigned char *arena;
    int32_t *key;         /* per slot: SLOT_EMPTY, SLOT_INFLIGHT, or layer * n_expert + e */
    uint64_t *used;       /* per slot: tick of last use */
    const unsigned char **ptr;  /* [slot][3]: first byte of each slice */
    int32_t *where;       /* key -> slot, or -1 */
    uint64_t tick, hits, misses;
    /* eviction policy: 0 = LRU, 1 = decayed LFU (score halves every `half_life` tokens) */
    int policy;
    double half_life;
    double *score;        /* per key (layer * n_expert + e), survives eviction */
    int *score_step;      /* token step at which score was last updated */
    /* prefetch: reads issued for predicted experts of upcoming layers. Requests live in a
     * fixed pool that is never moved or reallocated: Windows writes completion status into
     * each request's OVERLAPPED at the address it was submitted with. */
    int32_t *inflight;    /* key -> slot reserved by a prefetch still being read, or -1 */
    int *slot_left;       /* per slot: prefetch slice reads not yet completed */
    char *pf_mark;        /* per slot: filled by a prefetch and not requested yet */
    QmsPending *pf;       /* pool of 3 * nslot requests */
    int *pf_act, npf;     /* indices of outstanding requests */
    int *pf_free, nfree;  /* indices of unused pool entries */
    uint64_t pf_issued, pf_used, pf_waited;
} QmsECache;

typedef struct {
    QmsGguf g;
    QmsCfg c;
    QmsLayer *L;
    QmsMat embd, output;
    float *out_norm;
    /* every buffer read from the file, freed together */
    void **allocs;
    int n_allocs;
    uint64_t resident_bytes;
    /* KV cache: [layer][pos][n_head_kv * head_dim] */
    int ctx;
    float *kc, *vc;
    /* expert streaming */
    int streamed;
    int pf_depth, pf_extra;
    double pf_min_p;
    int overlap;                      /* compute experts as their reads land (default on) */
    QmsTrunk tr;
    uint64_t plan_cache_bytes;        /* expert cache size decided by the memory plan */
    QmsECache cache;
    const unsigned char **eptr;       /* [3][n_expert]: gate/up/down base of each expert */
    uint64_t ex_bytes_read, ex_reads;
    double ex_read_s;
    /* optional expert-request trace: records {int32 step, int16 layer, int16 expert} */
    FILE *trace;
    int trace_step;
} QmsModel;

int  qms_model_load(QmsModel *m, const char *path, const QmsLoadOpts *o);   /* 0 on success */
void qms_model_free(QmsModel *m);

/* Run T tokens at positions pos0..pos0+T-1, appending to the KV cache.
 * Writes logits for the last position (all_logits == 0) or for every position
 * (logits sized [T][vocab]). Returns 0 on success. */
int qms_forward(QmsModel *m, const int *ids, int T, int pos0, float *logits, int all_logits);

/* Optional section timers (enabled when qms_prof_on is set, e.g. from QMS_PROF=1). */
enum { QMS_PROF_QKV, QMS_PROF_ATTN, QMS_PROF_O, QMS_PROF_ROUTER, QMS_PROF_FETCH, QMS_PROF_MOE, QMS_PROF_OUT,
       QMS_PROF_N };
extern int qms_prof_on;
extern double qms_prof[QMS_PROF_N];
double qms_now(void);
#define PROF_NOW() (qms_prof_on ? qms_now() : 0.0)
#define PROF_ADD(k, t) do { if (qms_prof_on) { double n_ = qms_now(); qms_prof[k] += n_ - (t); (t) = n_; } } while (0)

#endif /* QMS_MODEL_H */
