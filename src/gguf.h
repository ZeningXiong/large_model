/* gguf.h - minimal GGUF v3 reader: metadata scalars we need + tensor table, multi-shard. */
#ifndef QMS_GGUF_H
#define QMS_GGUF_H

#include <stdint.h>
#include "qms_io.h"

enum {  /* ggml tensor types used by Qwen3 GGUFs */
    QMS_T_F32 = 0, QMS_T_F16 = 1, QMS_T_Q4_0 = 2, QMS_T_Q4_1 = 3, QMS_T_Q8_0 = 8, QMS_T_Q6_K = 14
};

#define QMS_MAX_SHARDS 16
#define QMS_MAX_NAME 96

typedef struct {
    char name[QMS_MAX_NAME];
    int type;
    int n_dims;
    int64_t ne[4];        /* ne[0] is the innermost (contiguous) dimension */
    int shard;
    uint64_t off;         /* absolute byte offset in its shard file */
    uint64_t nbytes;
} QmsTensor;

typedef struct {
    char key[96];
    int vtype;            /* GGUF value type of a scalar; arrays are not stored */
    struct { uint64_t u; int64_t i; double f; } v;   /* all three views filled on read */
    char str[256];        /* string values (truncated) */
} QmsKV;

typedef struct {
    int n_shards;
    char path[QMS_MAX_SHARDS][512];
    qms_fd fd[QMS_MAX_SHARDS];
    qms_fd fd_direct[QMS_MAX_SHARDS];   /* unbuffered handles, opened by qms_gguf_open_direct */
    uint64_t file_size[QMS_MAX_SHARDS];
    int direct;
    int n_tensors;
    QmsTensor *tensors;
    int n_kv;
    QmsKV *kv;
} QmsGguf;

int  qms_gguf_open(QmsGguf *g, const char *path);   /* 0 on success; prints the reason on failure */
void qms_gguf_close(QmsGguf *g);

const QmsTensor *qms_gguf_find(const QmsGguf *g, const char *name);
const QmsKV *qms_gguf_kv(const QmsGguf *g, const char *key);
/* Required numeric value; returns -1 and prints when missing. Never invents a default. */
int qms_gguf_get_int(const QmsGguf *g, const char *key, int64_t *out);
int qms_gguf_get_float(const QmsGguf *g, const char *key, double *out);

/* Bytes per row of ne0 elements for a tensor type; 0 if unsupported or not divisible. */
uint64_t qms_row_bytes(int type, int64_t ne0);
const char *qms_type_name(int type);

/* Read a tensor's bytes (or a byte range of it) into buf. */
int qms_gguf_read(const QmsGguf *g, const QmsTensor *t, uint64_t rel_off, uint64_t n, void *buf);

/* Metadata arrays (tokenizer tables), read from the first shard's header on demand.
 * Strings: *out is one malloc'd block of NUL-terminated strings plus an index; free with
 * qms_gguf_free_strs. Integers: any integer element type, widened to int32. */
typedef struct { char *blob; char **s; uint32_t *len; int n; } QmsStrArr;
int  qms_gguf_get_strs(const QmsGguf *g, const char *key, QmsStrArr *out);
void qms_gguf_free_strs(QmsStrArr *a);
int  qms_gguf_get_i32s(const QmsGguf *g, const char *key, int32_t **out, int *n);

/* Unbuffered reads (bypass the OS file cache). */
#define QMS_DIRECT_ALIGN 4096ull
int qms_gguf_open_direct(QmsGguf *g);
/* Bytes of aligned buffer needed to read n bytes at any offset unbuffered. */
static inline uint64_t qms_direct_span(uint64_t n) { return n + 2 * QMS_DIRECT_ALIGN; }

#endif /* QMS_GGUF_H */
