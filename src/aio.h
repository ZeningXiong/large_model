/* aio.h - submit many positional reads at once, wait for them later.
 *
 * Windows: overlapped ReadFile, one event per request, so the device sees every request of a
 * batch at the same time (queue depth = batch size) and the calling thread is free between
 * submit and wait (used for prefetching the next layer while computing this one).
 * POSIX: synchronous pread at wait time (correct, not overlapped). */
#ifndef QMS_AIO_H
#define QMS_AIO_H

#include "qms_io.h"

typedef struct {
    qms_fd fd;
    uint64_t off;
    size_t len;
    size_t min_ok;      /* bytes that must arrive (unbuffered windows may run past EOF) */
    void *buf;
    /* private */
#ifdef _WIN32
    OVERLAPPED ov;
#endif
    int state;          /* 0 idle, 1 submitted, 2 done */
    int err;
} QmsIoReq;

#ifdef _WIN32
static inline int qms_aio_submit(QmsIoReq *r)
{
    memset(&r->ov, 0, sizeof r->ov);
    r->ov.Offset = (DWORD)(r->off & 0xFFFFFFFFu);
    r->ov.OffsetHigh = (DWORD)(r->off >> 32);
    r->ov.hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
    r->err = 0;
    r->state = 1;
    if (!r->ov.hEvent || r->len > (1u << 30)) { r->err = 1; r->state = 2; return -1; }
    if (!ReadFile(r->fd, r->buf, (DWORD)r->len, NULL, &r->ov)) {
        const DWORD e = GetLastError();
        if (e != ERROR_IO_PENDING && e != ERROR_HANDLE_EOF) { r->err = 1; return -1; }
    }
    return 0;
}

static inline int qms_aio_wait(QmsIoReq *r)
{
    if (r->state != 1) return r->err ? -1 : 0;
    DWORD got = 0;
    if (!GetOverlappedResult(r->fd, &r->ov, &got, TRUE) && GetLastError() != ERROR_HANDLE_EOF) r->err = 1;
    if (!r->err && got < r->min_ok) {
        /* a short read before EOF: finish it synchronously */
        if (qms_pread_min(r->fd, (unsigned char *)r->buf + got, r->len - got, r->off + got,
                          r->min_ok - got) != 0) r->err = 1;
    }
    CloseHandle(r->ov.hEvent);
    r->state = 2;
    return r->err ? -1 : 0;
}

/* Non-blocking: 1 if the request has completed (it is then finished like qms_aio_wait). */
static inline int qms_aio_poll(QmsIoReq *r)
{
    if (r->state != 1) return 1;
    if (!HasOverlappedIoCompleted(&r->ov)) return 0;
    qms_aio_wait(r);
    return 1;
}
#else
static inline int qms_aio_submit(QmsIoReq *r) { r->state = 1; r->err = 0; return 0; }
static inline int qms_aio_wait(QmsIoReq *r)
{
    if (r->state != 1) return r->err ? -1 : 0;
    r->err = qms_pread_min(r->fd, r->buf, r->len, r->off, r->min_ok) != 0;
    r->state = 2;
    return r->err ? -1 : 0;
}
static inline int qms_aio_poll(QmsIoReq *r) { qms_aio_wait(r); return 1; }
#endif

#endif /* QMS_AIO_H */
