/* kernels.c - fast dot products: 4/6-bit weights x int8 activations (AVX2, VNNI if available).
 *
 * Activations are quantized per 32 elements to int8 with one float scale (like ggml's Q8_0),
 * then each weight block is multiplied with integer instructions. Results differ from the
 * fp32 reference path by activation rounding only; the reference path stays available
 * (qms_exact_math = 1) and is what these kernels are tested against.
 *
 * Block layouts (ggml):
 *   Q4_0  { f16 d; u8 qs[16] }            w = (q - 8) * d,   q in 0..15
 *   Q4_1  { f16 d; f16 m; u8 qs[16] }     w = q * d + m
 *   Q6_K  256 per super-block, 16-element sub-scales, see deq_q6_k in quant.c
 * For element j of a 32-block: j < 16 is the low nibble of qs[j], j >= 16 the high nibble
 * of qs[j - 16].
 */
#include "kernels.h"
#include "gguf.h"
#include "qms.h"

#include <math.h>
#include <string.h>
#include <immintrin.h>

static inline float h2f(const unsigned char *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return qms_f16_to_f32(h);
#endif
}

void qms_quantize_q8(const float *x, int64_t n, QmsQ8 *out)
{
    for (int64_t b = 0; b < n / 32; b++) {
        const float *xb = x + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(xb[j]));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int s0 = 0, s1 = 0;
        for (int j = 0; j < 32; j++) {
            const int q = (int)nearbyintf(xb[j] * id);
            out[b].q[j] = (int8_t)q;
            if (j < 16) s0 += q; else s1 += q;
        }
        out[b].d = d;
        out[b].s16[0] = (int16_t)s0;
        out[b].s16[1] = (int16_t)s1;
        out[b].ds = d * (float)(s0 + s1);
    }
}

#if defined(__AVX2__)
static inline __m256i dot_u8s8(__m256i acc, __m256i u, __m256i s)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, u, s);
#else
    const __m256i p = _mm256_maddubs_epi16(u, s);          /* |u*s| <= 63*127*2: no saturation */
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#endif
}

static inline float hsum256(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline __m256i unpack_q4(const unsigned char *qs)
{
    const __m128i raw = _mm_loadu_si128((const __m128i *)qs);
    const __m256i both = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);
    return _mm256_and_si256(both, _mm256_set1_epi8(0x0F));
}
#endif

static float dot_q4_0(const unsigned char *w, const QmsQ8 *x, int64_t nb)
{
    float corr = 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++, w += 18) {
        const float d = h2f(w);
        const __m256i q8 = _mm256_loadu_si256((const __m256i *)x[b].q);
        const __m256i s = dot_u8s8(_mm256_setzero_si256(), unpack_q4(w + 2), q8);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(d * x[b].d), acc);
        corr += d * x[b].ds;
    }
    return hsum256(acc) - 8.0f * corr;
#else
    float sum = 0.0f;
    for (int64_t b = 0; b < nb; b++, w += 18) {
        const float d = h2f(w);
        int s = 0;
        for (int j = 0; j < 16; j++)
            s += (w[2 + j] & 0x0F) * x[b].q[j] + (w[2 + j] >> 4) * x[b].q[j + 16];
        sum += (float)s * d * x[b].d;
        corr += d * x[b].ds;
    }
    return sum - 8.0f * corr;
#endif
}

static float dot_q4_1(const unsigned char *w, const QmsQ8 *x, int64_t nb)
{
    float off = 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++, w += 20) {
        const float d = h2f(w), m = h2f(w + 2);
        const __m256i q8 = _mm256_loadu_si256((const __m256i *)x[b].q);
        const __m256i s = dot_u8s8(_mm256_setzero_si256(), unpack_q4(w + 4), q8);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(d * x[b].d), acc);
        off += m * x[b].ds;
    }
    return hsum256(acc) + off;
#else
    float sum = 0.0f;
    for (int64_t b = 0; b < nb; b++, w += 20) {
        const float d = h2f(w), m = h2f(w + 2);
        int s = 0;
        for (int j = 0; j < 16; j++)
            s += (w[4 + j] & 0x0F) * x[b].q[j] + (w[4 + j] >> 4) * x[b].q[j + 16];
        sum += (float)s * d * x[b].d;
        off += m * x[b].ds;
    }
    return sum + off;
#endif
}

/* Q6_K: each 256-super-block covers 8 consecutive Q8 blocks. Inside a 128-half, the 32-block
 * g (0..3) takes nibble (g < 2 ? low : high) of ql[(g & 1) * 32 + l] and bits 2g..2g+1 of
 * qh[l], for l = 0..31, with sub-scale sc[2g] for l < 16 and sc[2g + 1] for l >= 16. */
static float dot_q6_k(const unsigned char *w, const QmsQ8 *x, int64_t nsb)
{
    float corr = 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(0x03);
    for (int64_t sb = 0; sb < nsb; sb++, w += 210) {
        const float d = h2f(w + 208);
        const int8_t *scs = (const int8_t *)(w + 192);
        for (int half = 0; half < 2; half++) {
            const unsigned char *ql = w + half * 64;
            const unsigned char *qh = w + 128 + half * 32;
            const int8_t *sc = scs + half * 8;
            const __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);
            for (int g = 0; g < 4; g++) {
                const QmsQ8 *xb = &x[sb * 8 + half * 4 + g];
                const __m256i qlv = _mm256_loadu_si256((const __m256i *)(ql + (g & 1) * 32));
                const __m256i nib = g < 2 ? _mm256_and_si256(qlv, m4)
                                          : _mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4);
                const __m256i hb = _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qhv, 2 * g), m2), 4);
                const __m256i q6 = _mm256_or_si256(nib, hb);                /* 0..63 */
                const __m256i q8 = _mm256_loadu_si256((const __m256i *)xb->q);
                __m256i s = dot_u8s8(_mm256_setzero_si256(), q6, q8);     /* lanes 0-3: l<16 */
                const int sa = sc[2 * g], sbb = sc[2 * g + 1];
                s = _mm256_mullo_epi32(s, _mm256_setr_epi32(sa, sa, sa, sa, sbb, sbb, sbb, sbb));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(d * xb->d), acc);
                corr += d * xb->d * (float)(sa * xb->s16[0] + sbb * xb->s16[1]);
            }
        }
    }
    return hsum256(acc) - 32.0f * corr;
#else
    float sum = 0.0f;
    for (int64_t sb = 0; sb < nsb; sb++, w += 210) {
        const float d = h2f(w + 208);
        const int8_t *scs = (const int8_t *)(w + 192);
        for (int half = 0; half < 2; half++) {
            const unsigned char *ql = w + half * 64;
            const unsigned char *qh = w + 128 + half * 32;
            const int8_t *sc = scs + half * 8;
            for (int g = 0; g < 4; g++) {
                const QmsQ8 *xb = &x[sb * 8 + half * 4 + g];
                int s0 = 0, s1 = 0;
                for (int l = 0; l < 32; l++) {
                    const int lo = ql[(g & 1) * 32 + l];
                    const int q = ((g < 2 ? lo & 0xF : lo >> 4) | (((qh[l] >> (2 * g)) & 3) << 4)) - 32;
                    if (l < 16) s0 += q * xb->q[l]; else s1 += q * xb->q[l];
                }
                sum += d * xb->d * (float)(sc[2 * g] * s0 + sc[2 * g + 1] * s1);
            }
        }
    }
    (void)corr;
    return sum;
#endif
}

int qms_fast_supported(int type)
{
    return type == QMS_T_Q4_0 || type == QMS_T_Q4_1 || type == QMS_T_Q6_K || type == QMS_T_F32;
}

static float dot_f32(const float *w, const float *x, int64_t n)
{
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);
    }
    float s = hsum256(_mm256_add_ps(a0, a1));
    for (; i < n; i++) s += w[i] * x[i];
    return s;
#else
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += w[i] * x[i];
    return s;
#endif
}

float qms_vec_dot_f32(const float *w, const float *x, int64_t n) { return dot_f32(w, x, n); }

float qms_vec_dot_q8(int type, const unsigned char *w, const QmsQ8 *x, int64_t n)
{
    switch (type) {
    case QMS_T_Q4_0: return dot_q4_0(w, x, n / 32);
    case QMS_T_Q4_1: return dot_q4_1(w, x, n / 32);
    case QMS_T_Q6_K: return dot_q6_k(w, x, n / 256);
    default: return NAN;
    }
}
