/* test_kernels.c - integer dot kernels vs dequantize-then-dot, on random blocks.
 * The reference uses the SAME int8 activations (dequantized), so the only allowed difference
 * is float summation order: relative error must be tiny. Built both with SIMD and scalar. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "kernels.h"
#include "quant.h"

static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static void put_f16(unsigned char *p, int sign, int exp, int mant)
{
    uint16_t h = (uint16_t)((sign << 15) | (exp << 10) | mant);
    memcpy(p, &h, 2);
}

static void fill_block(int type, unsigned char *w, int64_t n)
{
    const uint64_t rb = qms_row_bytes(type, n);
    for (uint64_t i = 0; i < rb; i++) w[i] = (unsigned char)rnd();
    if (type == QMS_T_Q4_0 || type == QMS_T_Q4_1) {
        const int bs = type == QMS_T_Q4_0 ? 18 : 20;
        for (int64_t b = 0; b < n / 32; b++) {
            put_f16(w + b * bs, 0, 9 + rnd() % 4, rnd() % 1024);                   /* d > 0 */
            if (type == QMS_T_Q4_1) put_f16(w + b * bs + 2, 1, 9 + rnd() % 4, rnd() % 1024);
        }
    } else if (type == QMS_T_Q6_K) {
        for (int64_t b = 0; b < n / 256; b++) {
            unsigned char *sb = w + b * 210;
            for (int j = 0; j < 16; j++) sb[192 + j] = (unsigned char)(int8_t)((int)(rnd() % 128) - 64);
            put_f16(sb + 208, 0, 5 + rnd() % 3, rnd() % 1024);
        }
    }
}

int main(void)
{
    const int types[] = { QMS_T_Q4_0, QMS_T_Q4_1, QMS_T_Q6_K };
    const int64_t sizes[] = { 256, 768, 1536, 2048, 4096, 8192 };
    int fails = 0;
#if defined(__AVX2__)
    printf("kernel test (AVX2%s)\n",
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
           " + VNNI"
#else
           ""
#endif
    );
#else
    printf("kernel test (scalar)\n");
#endif
    for (size_t ti = 0; ti < sizeof types / sizeof *types; ti++) {
        double worst = 0.0;
        for (size_t si = 0; si < sizeof sizes / sizeof *sizes; si++) {
            const int64_t n = sizes[si];
            unsigned char *w = (unsigned char *)malloc(qms_row_bytes(types[ti], n));
            float *x = (float *)malloc((size_t)n * sizeof(float));
            float *wd = (float *)malloc((size_t)n * sizeof(float));
            QmsQ8 *xq = (QmsQ8 *)malloc((size_t)(n / 32) * sizeof(QmsQ8));
            for (int trial = 0; trial < 50; trial++) {
                fill_block(types[ti], w, n);
                for (int64_t i = 0; i < n; i++) x[i] = ((float)(rnd() % 20001) - 10000.0f) / 3000.0f;
                if (trial % 5 == 0) for (int i = 0; i < 32; i++) x[i] = 0.0f;   /* an all-zero block */
                qms_quantize_q8(x, n, xq);
                qms_dequant_row(types[ti], w, wd, n);
                double ref = 0.0, mag = 0.0;
                for (int64_t i = 0; i < n; i++) {
                    const double xi = (double)xq[i / 32].q[i % 32] * xq[i / 32].d;
                    ref += (double)wd[i] * xi;
                    mag += fabs((double)wd[i] * xi);
                }
                const double got = qms_vec_dot_q8(types[ti], w, xq, n);
                const double rel = fabs(got - ref) / (mag + 1e-30);
                if (rel > worst) worst = rel;
            }
            free(w); free(x); free(wd); free(xq);
        }
        const int ok = worst < 1e-5;
        fails += !ok;
        printf("  %s  %-5s worst |got-ref| / sum|terms| = %.2e\n", ok ? "PASS" : "FAIL",
               qms_type_name(types[ti]), worst);
    }
    printf(fails ? "KERNELS FAILED\n" : "KERNELS PASSED\n");
    return fails ? 1 : 0;
}
