/* memhog.c - make this machine behave like one with only TARGET_GB of memory available.
 * Allocates (available - TARGET_GB), asks Windows to keep it resident (working-set minimum,
 * a right ordinary users have), and touches one byte per page every few seconds so it stays
 * in use. Runs until the file STOPFILE exists. Prints "READY <avail GB>" once in place.
 *   memhog TARGET_GB STOPFILE */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

static double avail_gb(void)
{
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    return ms.ullAvailPhys / 1e9;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: memhog TARGET_GB STOPFILE\n"); return 2; }
    const double target = atof(argv[1]);
    const double hog_gb = avail_gb() - target;
    if (hog_gb <= 0.5) { printf("READY %.2f (nothing to hold)\n", avail_gb()); fflush(stdout); }
    const size_t page = 4096, chunk = (size_t)1 << 30;
    const size_t total = hog_gb > 0.5 ? (size_t)(hog_gb * 1e9) / page * page : 0;
    if (total) {
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), total + (64u << 20), total + (256u << 20),
                                   QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    }
    size_t nchunk = (total + chunk - 1) / chunk, have = 0;
    unsigned char **c = (unsigned char **)calloc(nchunk ? nchunk : 1, sizeof *c);
    for (size_t i = 0; i < nchunk; i++) {
        const size_t n = i + 1 < nchunk ? chunk : total - i * chunk;
        c[i] = (unsigned char *)VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!c[i]) { fprintf(stderr, "allocation stopped at %.1f GB\n", have / 1e9); nchunk = i; break; }
        for (size_t o = 0; o < n; o += page) c[i][o] = 1;
        have += n;
    }
    /* Pin it: a working-set minimum alone is a soft request that Windows trims under pressure
     * (measured: the engine under test then grew to 14 GB). VirtualLock pages cannot be trimmed
     * or paged out while locked. */
    size_t locked = 0;
    for (size_t i = 0; i < nchunk; i++) {
        const size_t n = i + 1 < nchunk ? chunk : total - i * chunk;
        if (VirtualLock(c[i], n)) locked += n;
        else { fprintf(stderr, "VirtualLock failed after %.1f GB (error %lu)\n", locked / 1e9, GetLastError()); break; }
    }
    if (total) { printf("READY %.2f (holding %.1f GB, locked %.1f GB)\n", avail_gb(), have / 1e9, locked / 1e9); fflush(stdout); }
    for (unsigned tick = 0;; tick++) {
        if (GetFileAttributesA(argv[2]) != INVALID_FILE_ATTRIBUTES) break;
        for (size_t i = 0; i < nchunk; i++) {
            const size_t n = i + 1 < nchunk ? chunk : total - i * chunk;
            for (size_t o = 0; o < n; o += page) c[i][o]++;
        }
        if (tick % 12 == 0) { fprintf(stderr, "[memhog] holding %.1f GB, available %.2f GB\n", have / 1e9, avail_gb()); }
        Sleep(5000);
    }
    for (size_t i = 0; i < nchunk; i++) VirtualFree(c[i], 0, MEM_RELEASE);
    return 0;
}
