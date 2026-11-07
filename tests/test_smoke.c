/* Step-0 smoke test: toolchain, ISA, OpenMP, fp16 decode. */
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "qms.h"

static int fails = 0;
#define CHECK(c, msg) do { if (c) printf("  PASS  %s\n", msg); \
                           else { printf("  FAIL  %s\n", msg); fails++; } } while (0)

int main(void)
{
    printf("qms %s smoke test\n", QMS_VERSION);
#if defined(__GNUC__)
    printf("  compiler : gcc %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif
    printf("  ISA      :");
#ifdef __AVX2__
    printf(" AVX2");
#endif
#ifdef __FMA__
    printf(" FMA");
#endif
#ifdef __AVX512F__
    printf(" AVX512F");
#endif
#ifdef __AVX512BW__
    printf(" AVX512BW");
#endif
#ifdef __AVX512VNNI__
    printf(" AVX512VNNI");
#endif
    printf("\n  threads  : %d (OpenMP max)\n", omp_get_max_threads());

    CHECK(sizeof(void *) == 8, "64-bit build");
    CHECK(qms_f16_to_f32(0x3C00) == 1.0f, "f16 1.0");
    CHECK(qms_f16_to_f32(0xC000) == -2.0f, "f16 -2.0");
    CHECK(qms_f16_to_f32(0x7BFF) == 65504.0f, "f16 max");
    CHECK(qms_f16_to_f32(0x0001) == 5.9604644775390625e-08f, "f16 min subnormal");

    long sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < 1000; i++) sum += i;
    CHECK(sum == 499500, "OpenMP reduction");

    printf(fails ? "SMOKE FAILED (%d)\n" : "SMOKE PASSED\n", fails);
    return fails ? 1 : 0;
}
