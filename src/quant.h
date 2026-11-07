/* quant.h - ggml block formats: dequantize rows and a reference matmul (fp32 activations). */
#ifndef QMS_QUANT_H
#define QMS_QUANT_H

#include <stdint.h>

/* Dequantize n elements (one row, n a multiple of the block size) into dst. 0 on success. */
int qms_dequant_row(int type, const void *src, float *dst, int64_t n);

/* A 2-D weight view: rows x cols, row r starts at data + r * row_bytes. */
typedef struct {
    int type;
    int64_t rows, cols;
    uint64_t row_bytes;
    const unsigned char *data;
} QmsMat;

/* y[t][r] = sum_c W[r][c] * x[t][c] for t < T.
 * Fast path (default): activations quantized to int8 per 32, integer dot products.
 * Exact path (qms_exact_math = 1): each row dequantized to fp32, fp32 dot product in a fixed
 * order. Either way results do not depend on the thread count. */
extern int qms_exact_math;
void qms_matmul(const QmsMat *W, const float *x, int T, float *y);
/* Several matrices sharing the same input (e.g. q, k, v): one quantization of x and one
 * parallel loop over all their rows. ys[i] is [T][Ws[i]->rows]. */
void qms_matmul_n(const QmsMat *const *Ws, int nW, const float *x, int T, float *const *ys);

#endif /* QMS_QUANT_H */
