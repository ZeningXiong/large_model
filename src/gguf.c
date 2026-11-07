/* gguf.c - GGUF v3 header/tensor-table parser. Only reads headers; tensor data stays on disk. */
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GGUF value types */
enum { GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64,
       GV_I64, GV_F64 };

static const int gv_size[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };

/* ---- buffered sequential reader over a positional fd ---- */
typedef struct {
    qms_fd fd;
    uint64_t file_size;
    uint64_t pos;          /* file position of buf[0] */
    size_t len, cur;
    unsigned char *buf;
    int err;
} Rd;

#define RD_BUF (1u << 20)

static int rd_fill(Rd *r)
{
    r->pos += r->cur;
    r->cur = 0;
    uint64_t left = r->file_size > r->pos ? r->file_size - r->pos : 0;
    r->len = left < RD_BUF ? (size_t)left : RD_BUF;
    if (r->len == 0 || qms_pread_full(r->fd, r->buf, r->len, r->pos) != 0) {
        r->err = 1;
        return -1;
    }
    return 0;
}

static int rd_bytes(Rd *r, void *dst, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    while (n > 0) {
        if (r->cur == r->len && rd_fill(r) != 0) return -1;
        size_t k = r->len - r->cur < n ? r->len - r->cur : n;
        if (d) { memcpy(d, r->buf + r->cur, k); d += k; }
        r->cur += k;
        n -= k;
    }
    return 0;
}

static uint64_t rd_u64(Rd *r) { uint64_t v = 0; rd_bytes(r, &v, 8); return v; }
static uint32_t rd_u32(Rd *r) { uint32_t v = 0; rd_bytes(r, &v, 4); return v; }
static uint64_t rd_tell(const Rd *r) { return r->pos + r->cur; }

/* Read a GGUF string into out (truncated to cap-1), always consuming the full string. */
static int rd_str(Rd *r, char *out, size_t cap)
{
    uint64_t n = rd_u64(r);
    if (r->err || n > (1ull << 32)) { r->err = 1; return -1; }
    size_t keep = out ? (n < cap - 1 ? (size_t)n : cap - 1) : 0;
    if (keep && rd_bytes(r, out, keep) != 0) return -1;
    if (out) out[keep] = 0;
    return rd_bytes(r, NULL, (size_t)(n - keep));
}

static int rd_skip_value(Rd *r, uint32_t t)
{
    if (t == GV_STR) return rd_str(r, NULL, 0);
    if (t == GV_ARR) {
        uint32_t et = rd_u32(r);
        uint64_t n = rd_u64(r);
        if (et > GV_F64) { r->err = 1; return -1; }
        if (et == GV_STR || et == GV_ARR) {
            for (uint64_t i = 0; i < n && !r->err; i++) rd_skip_value(r, et);
            return r->err ? -1 : 0;
        }
        return rd_bytes(r, NULL, (size_t)(n * (uint64_t)gv_size[et]));
    }
    if (t > GV_F64) { r->err = 1; return -1; }
    return rd_bytes(r, NULL, (size_t)gv_size[t]);
}

/* ---- types ---- */
uint64_t qms_row_bytes(int type, int64_t ne0)
{
    switch (type) {
    case QMS_T_F32:  return (uint64_t)ne0 * 4;
    case QMS_T_F16:  return (uint64_t)ne0 * 2;
    case QMS_T_Q4_0: return ne0 % 32 ? 0 : (uint64_t)ne0 / 32 * 18;
    case QMS_T_Q4_1: return ne0 % 32 ? 0 : (uint64_t)ne0 / 32 * 20;
    case QMS_T_Q8_0: return ne0 % 32 ? 0 : (uint64_t)ne0 / 32 * 34;
    case QMS_T_Q6_K: return ne0 % 256 ? 0 : (uint64_t)ne0 / 256 * 210;
    default:         return 0;
    }
}

const char *qms_type_name(int type)
{
    switch (type) {
    case QMS_T_F32: return "F32";
    case QMS_T_F16: return "F16";
    case QMS_T_Q4_0: return "Q4_0";
    case QMS_T_Q4_1: return "Q4_1";
    case QMS_T_Q8_0: return "Q8_0";
    case QMS_T_Q6_K: return "Q6_K";
    default: return "?";
    }
}

/* ---- one shard ---- */
static int parse_shard(QmsGguf *g, int si, int keep_kv)
{
    Rd r;
    memset(&r, 0, sizeof r);
    r.fd = g->fd[si];
    int64_t fsz = qms_file_size(r.fd);
    if (fsz <= 0) { fprintf(stderr, "gguf: cannot stat %s\n", g->path[si]); return -1; }
    r.file_size = (uint64_t)fsz;
    g->file_size[si] = (uint64_t)fsz;
    r.buf = (unsigned char *)malloc(RD_BUF);
    if (!r.buf) return -1;

    int rc = -1;
    char magic[4];
    if (rd_bytes(&r, magic, 4) != 0 || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "gguf: %s is not a GGUF file\n", g->path[si]);
        goto done;
    }
    uint32_t ver = rd_u32(&r);
    if (ver != 3) { fprintf(stderr, "gguf: %s version %u unsupported\n", g->path[si], ver); goto done; }
    uint64_t nt = rd_u64(&r), nkv = rd_u64(&r);
    uint64_t alignment = 32;

    for (uint64_t i = 0; i < nkv && !r.err; i++) {
        QmsKV kv;
        memset(&kv, 0, sizeof kv);
        rd_str(&r, kv.key, sizeof kv.key);
        uint32_t t = rd_u32(&r);
        kv.vtype = (int)t;
        int store = keep_kv;
        switch (t) {
        case GV_U8:  { uint8_t v = 0;  rd_bytes(&r, &v, 1); kv.v.u = v; kv.v.i = v; kv.v.f = v; break; }
        case GV_I8:  { int8_t v = 0;   rd_bytes(&r, &v, 1); kv.v.i = v; kv.v.u = (uint64_t)v; kv.v.f = v; break; }
        case GV_U16: { uint16_t v = 0; rd_bytes(&r, &v, 2); kv.v.u = v; kv.v.i = v; kv.v.f = v; break; }
        case GV_I16: { int16_t v = 0;  rd_bytes(&r, &v, 2); kv.v.i = v; kv.v.u = (uint64_t)v; kv.v.f = v; break; }
        case GV_U32: { uint32_t v = 0; rd_bytes(&r, &v, 4); kv.v.u = v; kv.v.i = v; kv.v.f = v; break; }
        case GV_I32: { int32_t v = 0;  rd_bytes(&r, &v, 4); kv.v.i = v; kv.v.u = (uint64_t)v; kv.v.f = v; break; }
        case GV_F32: { float v = 0;    rd_bytes(&r, &v, 4); kv.v.f = v; kv.v.i = (int64_t)v; break; }
        case GV_BOOL:{ uint8_t v = 0;  rd_bytes(&r, &v, 1); kv.v.u = v; kv.v.i = v; kv.v.f = v; break; }
        case GV_U64: { uint64_t v = 0; rd_bytes(&r, &v, 8); kv.v.u = v; kv.v.i = (int64_t)v; kv.v.f = (double)v; break; }
        case GV_I64: { int64_t v = 0;  rd_bytes(&r, &v, 8); kv.v.i = v; kv.v.u = (uint64_t)v; kv.v.f = (double)v; break; }
        case GV_F64: { double v = 0;   rd_bytes(&r, &v, 8); kv.v.f = v; kv.v.i = (int64_t)v; break; }
        case GV_STR: rd_str(&r, kv.str, sizeof kv.str); break;
        default:     rd_skip_value(&r, t); store = 0; break;   /* arrays: tokenizer, not needed here */
        }
        if (!strcmp(kv.key, "general.alignment")) alignment = kv.v.u;
        if (store) {
            QmsKV *nk = (QmsKV *)realloc(g->kv, (size_t)(g->n_kv + 1) * sizeof *nk);
            if (!nk) goto done;
            g->kv = nk;
            g->kv[g->n_kv++] = kv;
        }
    }
    if (r.err) { fprintf(stderr, "gguf: truncated metadata in %s\n", g->path[si]); goto done; }

    QmsTensor *nt_arr = (QmsTensor *)realloc(g->tensors, (size_t)(g->n_tensors + nt) * sizeof *nt_arr);
    if (!nt_arr) goto done;
    g->tensors = nt_arr;
    int first = g->n_tensors;
    for (uint64_t i = 0; i < nt && !r.err; i++) {
        QmsTensor *t = &g->tensors[g->n_tensors++];
        memset(t, 0, sizeof *t);
        rd_str(&r, t->name, sizeof t->name);
        t->n_dims = (int)rd_u32(&r);
        if (t->n_dims < 1 || t->n_dims > 4) { r.err = 1; break; }
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (int d = 0; d < t->n_dims; d++) t->ne[d] = (int64_t)rd_u64(&r);
        t->type = (int)rd_u32(&r);
        t->off = rd_u64(&r);        /* relative for now */
        t->shard = si;
    }
    if (r.err) { fprintf(stderr, "gguf: truncated tensor table in %s\n", g->path[si]); goto done; }

    uint64_t data_start = (rd_tell(&r) + alignment - 1) / alignment * alignment;
    for (int i = first; i < g->n_tensors; i++) {
        QmsTensor *t = &g->tensors[i];
        uint64_t rb = qms_row_bytes(t->type, t->ne[0]);
        t->nbytes = rb * (uint64_t)(t->ne[1] * t->ne[2] * t->ne[3]);
        t->off += data_start;
        if (rb && t->off + t->nbytes > r.file_size) {
            fprintf(stderr, "gguf: tensor %s runs past end of %s\n", t->name, g->path[si]);
            goto done;
        }
    }
    rc = 0;
done:
    free(r.buf);
    return rc;
}

/* "...-00001-of-00003.gguf" -> list of all shard paths */
static int expand_shards(QmsGguf *g, const char *path)
{
    size_t n = strlen(path);
    const char *tail = "-00001-of-00000.gguf";
    size_t tl = strlen(tail);
    if (n > tl && !strncmp(path + n - tl, "-00001-of-", 10)) {
        int count = atoi(path + n - tl + 10);
        if (count < 1 || count > QMS_MAX_SHARDS) return -1;
        for (int i = 0; i < count; i++) {
            snprintf(g->path[i], sizeof g->path[i], "%.*s-%05d-of-%05d.gguf",
                     (int)(n - tl), path, i + 1, count);
        }
        g->n_shards = count;
    } else {
        snprintf(g->path[0], sizeof g->path[0], "%s", path);
        g->n_shards = 1;
    }
    return 0;
}

int qms_gguf_open(QmsGguf *g, const char *path)
{
    memset(g, 0, sizeof *g);
    if (expand_shards(g, path) != 0) { fprintf(stderr, "gguf: bad shard name %s\n", path); return -1; }
    for (int i = 0; i < g->n_shards; i++) {
        g->fd[i] = qms_open_read(g->path[i]);
        if (g->fd[i] == QMS_BAD_FD) {
            fprintf(stderr, "gguf: cannot open %s\n", g->path[i]);
            g->n_shards = i;
            qms_gguf_close(g);
            return -1;
        }
        if (parse_shard(g, i, i == 0) != 0) {
            g->n_shards = i + 1;
            qms_gguf_close(g);
            return -1;
        }
    }
    return 0;
}

void qms_gguf_close(QmsGguf *g)
{
    for (int i = 0; i < g->n_shards; i++) {
        if (g->fd[i] != QMS_BAD_FD && g->fd[i] != 0) qms_close(g->fd[i]);
        if (g->direct && g->fd_direct[i] != QMS_BAD_FD) qms_close(g->fd_direct[i]);
    }
    free(g->tensors);
    free(g->kv);
    memset(g, 0, sizeof *g);
}

const QmsTensor *qms_gguf_find(const QmsGguf *g, const char *name)
{
    for (int i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

const QmsKV *qms_gguf_kv(const QmsGguf *g, const char *key)
{
    for (int i = 0; i < g->n_kv; i++)
        if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

int qms_gguf_get_int(const QmsGguf *g, const char *key, int64_t *out)
{
    const QmsKV *kv = qms_gguf_kv(g, key);
    if (!kv || kv->vtype == GV_STR) { fprintf(stderr, "gguf: missing required key %s\n", key); return -1; }
    *out = kv->v.i;
    return 0;
}

int qms_gguf_get_float(const QmsGguf *g, const char *key, double *out)
{
    const QmsKV *kv = qms_gguf_kv(g, key);
    if (!kv || kv->vtype == GV_STR) { fprintf(stderr, "gguf: missing required key %s\n", key); return -1; }
    *out = kv->v.f;
    return 0;
}

int qms_gguf_read(const QmsGguf *g, const QmsTensor *t, uint64_t rel_off, uint64_t n, void *buf)
{
    if (rel_off + n > t->nbytes) return -1;
    return qms_pread_full(g->fd[t->shard], buf, (size_t)n, t->off + rel_off);
}

int qms_gguf_open_direct(QmsGguf *g)
{
    for (int i = 0; i < g->n_shards; i++) {
        g->fd_direct[i] = qms_open_read_direct(g->path[i]);
        if (g->fd_direct[i] == QMS_BAD_FD) {
            fprintf(stderr, "gguf: cannot open %s unbuffered\n", g->path[i]);
            for (int j = 0; j < i; j++) qms_close(g->fd_direct[j]);
            return -1;
        }
    }
    g->direct = 1;
    return 0;
}

/* ---- metadata arrays ---- */

/* Position a reader at the value of `key` in shard 0's metadata. Returns the value type, or -1. */
static int seek_key(const QmsGguf *g, Rd *r, const char *key)
{
    memset(r, 0, sizeof *r);
    r->fd = g->fd[0];
    r->file_size = g->file_size[0];
    r->buf = (unsigned char *)malloc(RD_BUF);
    if (!r->buf) return -1;
    char magic[4];
    rd_bytes(r, magic, 4);
    rd_u32(r);
    rd_u64(r);
    const uint64_t nkv = rd_u64(r);
    char name[256];
    for (uint64_t i = 0; i < nkv && !r->err; i++) {
        rd_str(r, name, sizeof name);
        const uint32_t t = rd_u32(r);
        if (!strcmp(name, key)) return (int)t;
        rd_skip_value(r, t);
    }
    return -1;
}

int qms_gguf_get_strs(const QmsGguf *g, const char *key, QmsStrArr *out)
{
    memset(out, 0, sizeof *out);
    Rd r;
    const int t = seek_key(g, &r, key);
    int rc = -1;
    if (t != GV_ARR) goto done;
    if (rd_u32(&r) != GV_STR) goto done;
    const uint64_t n = rd_u64(&r);
    /* pass 1 would need a second seek; instead grow the blob as strings arrive */
    size_t cap = (size_t)n * 8 + 1024, used = 0;
    out->blob = (char *)malloc(cap);
    uint64_t *offs = (uint64_t *)malloc((size_t)n * sizeof *offs);
    out->len = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!out->blob || !offs || !out->len) { free(offs); goto done; }
    for (uint64_t i = 0; i < n && !r.err; i++) {
        const uint64_t len = rd_u64(&r);
        if (len > (1u << 20)) { r.err = 1; break; }
        if (used + len + 1 > cap) {
            while (used + len + 1 > cap) cap *= 2;
            char *nb = (char *)realloc(out->blob, cap);
            if (!nb) { r.err = 1; break; }
            out->blob = nb;
        }
        rd_bytes(&r, out->blob + used, (size_t)len);
        out->blob[used + len] = 0;
        offs[i] = used;
        out->len[i] = (uint32_t)len;
        used += len + 1;
    }
    if (!r.err) {
        out->s = (char **)malloc((size_t)n * sizeof(char *));
        if (out->s) {
            for (uint64_t i = 0; i < n; i++) out->s[i] = out->blob + offs[i];
            out->n = (int)n;
            rc = 0;
        }
    }
    free(offs);
done:
    free(r.buf);
    if (rc) { qms_gguf_free_strs(out); fprintf(stderr, "gguf: cannot read string array %s\n", key); }
    return rc;
}

void qms_gguf_free_strs(QmsStrArr *a)
{
    free(a->blob); free(a->s); free(a->len);
    memset(a, 0, sizeof *a);
}

int qms_gguf_get_i32s(const QmsGguf *g, const char *key, int32_t **out, int *n)
{
    *out = NULL;
    *n = 0;
    Rd r;
    const int t = seek_key(g, &r, key);
    int rc = -1;
    if (t == GV_ARR) {
        const uint32_t et = rd_u32(&r);
        const uint64_t cnt = rd_u64(&r);
        const int sz = et <= GV_F64 ? gv_size[et] : 0;
        if (sz && et != GV_F32 && et != GV_F64 && et != GV_STR && et != GV_ARR) {
            int32_t *v = (int32_t *)malloc((size_t)cnt * sizeof(int32_t));
            for (uint64_t i = 0; v && i < cnt && !r.err; i++) {
                int64_t x = 0;
                rd_bytes(&r, &x, (size_t)sz);
                if ((et == GV_I8 && sz == 1) || et == GV_I16 || et == GV_I32) {   /* sign-extend */
                    const int sh = 64 - 8 * sz;
                    x = (int64_t)((uint64_t)x << sh) >> sh;
                }
                v[i] = (int32_t)x;
            }
            if (v && !r.err) { *out = v; *n = (int)cnt; rc = 0; } else free(v);
        }
    }
    free(r.buf);
    if (rc) fprintf(stderr, "gguf: cannot read integer array %s\n", key);
    return rc;
}
