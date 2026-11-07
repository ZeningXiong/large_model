/* devbw.c - what this disk delivers for the engine's access pattern: random, unbuffered,
 * aligned reads of a given size at a given queue depth (requests in flight at once).
 *   devbw FILE [MB_per_request ...]      prints GB/s for queue depth 1, 4, 8, 16, 32, 64 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aio.h"
#include "qms_io.h"

#ifdef _WIN32
static double now(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#endif

static uint64_t rng = 88172645463325252ull;
static uint64_t xs(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: devbw FILE [MB_per_request ...]\n"); return 2; }
    qms_fd fd = qms_open_read_direct(argv[1]);
    if (fd == QMS_BAD_FD) { fprintf(stderr, "cannot open %s unbuffered\n", argv[1]); return 1; }
    const int64_t fsz = qms_file_size(fd);
    double sizes_mb[8] = { 1.0, 3.5 };
    int nsz = 2;
    if (argc > 2) { nsz = 0; for (int i = 2; i < argc && nsz < 8; i++) sizes_mb[nsz++] = atof(argv[i]); }
    const int qds[] = { 1, 4, 8, 16, 32, 64 };
    printf("file %s (%.1f GB), random unbuffered reads, 4096-aligned\n", argv[1], fsz / 1e9);
    printf("%10s", "req MB");
    for (size_t q = 0; q < sizeof qds / sizeof *qds; q++) printf("   QD%-3d", qds[q]);
    printf("   (GB/s)\n");
    for (int si = 0; si < nsz; si++) {
        const size_t len = ((size_t)(sizes_mb[si] * 1e6) + 4095) & ~(size_t)4095;
        printf("%10.2f", len / 1e6);
        for (size_t q = 0; q < sizeof qds / sizeof *qds; q++) {
            const int qd = qds[q];
            QmsIoReq *r = (QmsIoReq *)calloc((size_t)qd, sizeof *r);
            unsigned char *buf = (unsigned char *)qms_alloc_aligned(len * (size_t)qd);
            const uint64_t target = 2ull << 30;          /* ~2 GiB per cell */
            uint64_t done = 0;
            const double t0 = now();
            while (done < target) {
                for (int i = 0; i < qd; i++) {
                    r[i].fd = fd;
                    r[i].off = (xs() % (uint64_t)((fsz - (int64_t)len) / 4096)) * 4096;
                    r[i].len = len;
                    r[i].min_ok = len;
                    r[i].buf = buf + (size_t)i * len;
                    qms_aio_submit(&r[i]);
                }
                for (int i = 0; i < qd; i++) {
                    if (qms_aio_wait(&r[i]) != 0) { fprintf(stderr, "read failed\n"); return 1; }
                    done += len;
                }
            }
            printf("  %6.2f", done / 1e9 / (now() - t0));
            fflush(stdout);
            qms_free_aligned(buf);
            free(r);
        }
        printf("\n");
    }
    qms_close(fd);
    return 0;
}
