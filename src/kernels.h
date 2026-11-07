/* kernels.h - int8-activation dot products for quantized weights. */
#ifndef QMS_KERNELS_H
#define QMS_KERNELS_H

#include <stdint.h>

typedef struct {
    float d;           /* scale: x ~= q * d */
    float ds;          /* d * sum(q[0..31]) */
    int16_t s16[2];    /* sum(q[0..15]), sum(q[16..31]) - Q6_K needs 16-element sums */
    int8_t q[32];
} QmsQ8;

/* Quantize n floats (n % 32 == 0) into n/32 blocks. */
void qms_quantize_q8(const float *x, int64_t n, QmsQ8 *out);

int qms_fast_supported(int type);
/* Dot of one weight row (n elements, given type) with a quantized activation vector. */
float qms_vec_dot_q8(int type, const unsigned char *w, const QmsQ8 *x, int64_t n);
/* fp32 dot (AVX2 when available) for F32 weights such as the router. */
float qms_vec_dot_f32(const float *w, const float *x, int64_t n);

#endif /* QMS_KERNELS_H */
