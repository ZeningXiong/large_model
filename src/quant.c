/* quant.c - block dequantization (bit-compatible with ggml's reference dequantizers). */
#include "quant.h"
#include "kernels.h"
#include "gguf.h"
#include "qms.h"

#include <stdlib.h>
#include <string.h>

static inline float f16(const unsigned char *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
    return qms_f16_to_f32(h);
}

/* Q4_0: { f16 d; u8 qs[16] }  x[j] = (lo(qs[j]) - 8) * d,  x[j+16] = (hi(qs[j]) - 8) * d */
static void deq_q4_0(const unsigned char *b, float *y, int64_t n)
{
    for (int64_t i = 0; i < n; i += 32, b += 18, y += 32) {
        const float d = f16(b);
        const unsigned char *qs = b + 2;
        for (int j = 0; j < 16; j++) {
            y[j]      = (float)((qs[j] & 0x0F) - 8) * d;
            y[j + 16] = (float)((qs[j] >> 4) - 8) * d;
        }
    }
}

/* Q4_1: { f16 d; f16 m; u8 qs[16] }  x = q * d + m */
static void deq_q4_1(const unsigned char *b, float *y, int64_t n)
{
    for (int64_t i = 0; i < n; i += 32, b += 20, y += 32) {
        const float d = f16(b), m = f16(b + 2);
        const unsigned char *qs = b + 4;
        for (int j = 0; j < 16; j++) {
            y[j]      = (float)(qs[j] & 0x0F) * d + m;
            y[j + 16] = (float)(qs[j] >> 4) * d + m;
        }
    }
}

/* Q8_0: { f16 d; i8 qs[32] } */
static void deq_q8_0(const unsigned char *b, float *y, int64_t n)
{
    for (int64_t i = 0; i < n; i += 32, b += 34, y += 32) {
        const float d = f16(b);
        const int8_t *qs = (const int8_t *)(b + 2);
        for (int j = 0; j < 32; j++) y[j] = (float)qs[j] * d;
    }
}

/* Q6_K super-block of 256: { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d } = 210 bytes */
static void deq_q6_k(const unsigned char *b, float *y, int64_t n)
{
    for (int64_t i = 0; i < n; i += 256, b += 210) {
        const unsigned char *ql = b;
        const unsigned char *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        const float d = f16(b + 208);
        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

int qms_dequant_row(int type, const void *src, float *dst, int64_t n)
{
    const unsigned char *b = (const unsigned char *)src;
    switch (type) {
    case QMS_T_F32:  memcpy(dst, b, (size_t)n * 4); return 0;
    case QMS_T_F16:  for (int64_t i = 0; i < n; i++) dst[i] = f16(b + 2 * i); return 0;
    case QMS_T_Q4_0: if (n % 32) return -1; deq_q4_0(b, dst, n); return 0;
    case QMS_T_Q4_1: if (n % 32) return -1; deq_q4_1(b, dst, n); return 0;
    case QMS_T_Q8_0: if (n % 32) return -1; deq_q8_0(b, dst, n); return 0;
    case QMS_T_Q6_K: if (n % 256) return -1; deq_q6_k(b, dst, n); return 0;
    default: return -1;
    }
}

int qms_exact_math = 0;

static void matmul_exact(const QmsMat *W, const float *x, int T, float *y)
{
    const int64_t R = W->rows, C = W->cols;
#pragma omp parallel
    {
        float *row = (float *)malloc((size_t)C * sizeof(float));
#pragma omp for schedule(static)
        for (int64_t r = 0; r < R; r++) {
            qms_dequant_row(W->type, W->data + (uint64_t)r * W->row_bytes, row, C);
            for (int t = 0; t < T; t++) {
                const float *xt = x + (size_t)t * C;
                float acc = 0.0f;
                for (int64_t c = 0; c < C; c++) acc += row[c] * xt[c];
                y[(size_t)t * R + r] = acc;
            }
        }
        free(row);
    }
}

void qms_matmul(const QmsMat *W, const float *x, int T, float *y)
{
    const QmsMat *Ws[1] = { W };
    float *ys[1] = { y };
    qms_matmul_n(Ws, 1, x, T, ys);
}

/* Below this many multiply-adds a parallel region costs more than it saves (measured: a
 * fork/join is tens of microseconds here). */
#define QMS_PAR_MIN_WORK (1 << 21)

void qms_matmul_n(const QmsMat *const *Ws, int nW, const float *x, int T, float *const *ys)
{
    const int64_t C = Ws[0]->cols;
    int fast = !qms_exact_math;
    int64_t R_total = 0;
    for (int i = 0; i < nW; i++) {
        const QmsMat *W = Ws[i];
        if (W->cols != C || !qms_fast_supported(W->type) ||
            W->cols % (W->type == QMS_T_Q6_K ? 256 : 32) != 0) fast = 0;
        R_total += W->rows;
    }
    if (!fast) {
        for (int i = 0; i < nW; i++) matmul_exact(Ws[i], x, T, ys[i]);
        return;
    }
    const int par = R_total * C * (int64_t)T >= QMS_PAR_MIN_WORK;
    const int64_t nb = C / 32;
    QmsQ8 *xq = NULL;
    int any_q = 0;
    for (int i = 0; i < nW; i++) any_q |= Ws[i]->type != QMS_T_F32;
    if (any_q) {
        xq = (QmsQ8 *)malloc((size_t)T * nb * sizeof(QmsQ8));
        for (int t = 0; t < T; t++) qms_quantize_q8(x + (size_t)t * C, C, xq + (size_t)t * nb);
    }
    /* one parallel loop over the rows of all nW matrices */
#pragma omp parallel for schedule(static) if (par)
    for (int64_t g = 0; g < R_total; g++) {
        int i = 0;
        int64_t r = g;
        while (r >= Ws[i]->rows) { r -= Ws[i]->rows; i++; }
        const QmsMat *W = Ws[i];
        const int64_t R = W->rows;
        const unsigned char *wr = W->data + (uint64_t)r * W->row_bytes;
        if (W->type == QMS_T_F32) {
            for (int t = 0; t < T; t++)
                ys[i][(size_t)t * R + r] = qms_vec_dot_f32((const float *)wr, x + (size_t)t * C, C);
        } else {
            for (int t = 0; t < T; t++)
                ys[i][(size_t)t * R + r] = qms_vec_dot_q8(W->type, wr, xq + (size_t)t * nb, C);
        }
    }
    free(xq);
}
