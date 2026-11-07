/* qms_io.h - positional file reads that work on Windows and POSIX. */
#ifndef QMS_IO_H
#define QMS_IO_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HANDLE qms_fd;
#define QMS_BAD_FD INVALID_HANDLE_VALUE

static inline qms_fd qms_open_read(const char *path)
{
    /* OVERLAPPED so that concurrent positional reads from several threads really run in
     * parallel: Windows serializes I/O on handles opened for synchronous access. */
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);
}

static inline void qms_close(qms_fd fd) { CloseHandle(fd); }

/* Unbuffered handle: reads bypass the OS file cache entirely, so memory use is exactly what
 * the process allocates. Offsets, lengths and buffers must be 4096-aligned. */
static inline qms_fd qms_open_read_direct(const char *path)
{
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, NULL);
}

static inline void *qms_alloc_aligned(size_t n)
{
    return VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);   /* page-aligned */
}
static inline void qms_free_aligned(void *p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }

/* Read exactly n bytes at offset; returns 0 on success, -1 on any short read.
 * Safe to call from many threads at once on the same fd.
 * min_ok: accept a short read at end of file as long as at least min_ok bytes arrived
 * (unbuffered reads round the length up past EOF); pass n for "exactly n". */
static inline int qms_pread_min(qms_fd fd, void *buf, size_t n, uint64_t off, size_t min_ok)
{
    unsigned char *p = (unsigned char *)buf;
    size_t done = 0;
    HANDLE ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!ev) return -1;
    int rc = 0;
    while (done < n) {
        size_t left = n - done;
        DWORD chunk = left > (1u << 30) ? (1u << 30) : (DWORD)left;
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)((off + done) & 0xFFFFFFFFu);
        ov.OffsetHigh = (DWORD)((off + done) >> 32);
        ov.hEvent = ev;
        DWORD got = 0;
        if (!ReadFile(fd, p + done, chunk, NULL, &ov)) {
            DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            if (e != ERROR_IO_PENDING) { rc = -1; break; }
        }
        if (!GetOverlappedResult(fd, &ov, &got, TRUE)) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            rc = -1;
            break;
        }
        if (got == 0) break;
        done += got;
    }
    CloseHandle(ev);
    return rc == 0 && done >= min_ok ? 0 : -1;
}

static inline int qms_pread_full(qms_fd fd, void *buf, size_t n, uint64_t off)
{
    return qms_pread_min(fd, buf, n, off, n);
}

static inline int64_t qms_file_size(qms_fd fd)
{
    LARGE_INTEGER sz;
    return GetFileSizeEx(fd, &sz) ? (int64_t)sz.QuadPart : -1;
}

/* Physical memory the OS reports as available right now (bytes); 0 if unknown. */
static inline uint64_t qms_mem_available(void)
{
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullAvailPhys : 0;
}
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdlib.h>
typedef int qms_fd;
#define QMS_BAD_FD (-1)

static inline qms_fd qms_open_read(const char *path) { return open(path, O_RDONLY); }
static inline void qms_close(qms_fd fd) { close(fd); }

static inline int qms_pread_full(qms_fd fd, void *buf, size_t n, uint64_t off)
{
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        ssize_t got = pread(fd, p, n, (off_t)off);
        if (got <= 0) return -1;
        p += got;
        n -= (size_t)got;
        off += (uint64_t)got;
    }
    return 0;
}

static inline int qms_pread_min(qms_fd fd, void *buf, size_t n, uint64_t off, size_t min_ok)
{
    unsigned char *p = (unsigned char *)buf;
    size_t done = 0;
    while (done < n) {
        ssize_t got = pread(fd, p + done, n - done, (off_t)(off + done));
        if (got < 0) return -1;
        if (got == 0) break;
        done += (size_t)got;
    }
    return done >= min_ok ? 0 : -1;
}

static inline qms_fd qms_open_read_direct(const char *path) { return open(path, O_RDONLY | O_DIRECT); }

static inline void *qms_alloc_aligned(size_t n)
{
    void *p = NULL;
    return posix_memalign(&p, 4096, n) == 0 ? p : NULL;
}
static inline void qms_free_aligned(void *p) { free(p); }

static inline int64_t qms_file_size(qms_fd fd)
{
    struct stat st;
    return fstat(fd, &st) == 0 ? (int64_t)st.st_size : -1;
}

static inline uint64_t qms_mem_available(void)
{
    long pages = sysconf(_SC_AVPHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
    return pages > 0 && psz > 0 ? (uint64_t)pages * (uint64_t)psz : 0;
}
#endif

#endif /* QMS_IO_H */
