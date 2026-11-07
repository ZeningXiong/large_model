/* qms.h - shared definitions for the Qwen3-MoE streaming engine. */
#ifndef QMS_H
#define QMS_H

#include <stdint.h>
#include <stddef.h>

#define QMS_VERSION "0.0.1"

/* fp16 <-> fp32, exact for every finite half value (no hardware F16C needed). */
static inline float qms_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                /* +-0 */
        } else {                                        /* subnormal: renormalise */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);       /* inf / nan */
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = { bits };
    return v.f;
}

#endif /* QMS_H */
