/* tok.h - Qwen2 byte-level BPE tokenizer read from GGUF metadata (no external dependency). */
#ifndef QMS_TOK_H
#define QMS_TOK_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

enum { QMS_TOKT_NORMAL = 1, QMS_TOKT_UNKNOWN = 2, QMS_TOKT_CONTROL = 3, QMS_TOKT_USER = 4,
       QMS_TOKT_UNUSED = 5, QMS_TOKT_BYTE = 6 };

typedef struct {
    int n_vocab;
    QmsStrArr tok;          /* token strings, in GPT-2 byte-to-unicode form for normal tokens */
    int32_t *type;
    QmsStrArr merges;       /* "left right", rank = index */
    int *vh, vh_mask;       /* hash: token string -> id */
    int *mh, mh_mask;       /* hash: merge string -> rank */
    int *special, n_special;   /* control / user-defined ids, longest text first */
    unsigned char special_first[256];   /* bytes some special token starts with */
    uint32_t byte2cp[256];
    int16_t cp2byte[512];   /* inverse map; byte-level code points are all < 512 */
    int eos;                /* end of generation (<|im_end|> for chat models) */
} QmsTok;

int  qms_tok_load(QmsTok *t, const QmsGguf *g);
void qms_tok_free(QmsTok *t);
/* Encode UTF-8 text; special tokens written literally in the text become their ids.
 * *ids is malloc'd. Returns 0 on success. */
int  qms_tok_encode(const QmsTok *t, const char *text, size_t len, int **ids, int *n);
/* Bytes of token `id` as text (control tokens: their literal text). Returns length written. */
int  qms_tok_piece(const QmsTok *t, int id, char *buf, int cap);
int  qms_tok_find(const QmsTok *t, const char *s);   /* exact token string -> id, or -1 */

#endif /* QMS_TOK_H */
