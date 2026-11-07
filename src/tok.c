/* tok.c - Qwen2 tokenizer: special-token split, Qwen2 pre-tokenizer, byte-level BPE.
 *
 * Pre-tokenizer = the Qwen2 regex, applied leftmost-first:
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *   |[^\r\n\p{L}\p{N}]?\p{L}+ |\p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* |\s*[\r\n]+ |\s+(?!\S) |\s+
 * Each alternative is a function below, tried in that order, with the regex's greedy and
 * backtracking behaviour written out explicitly. */
#include "tok.h"
#include "unicode_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ unicode ---- */

static int in_ranges(const uint32_t (*r)[2], int n, uint32_t cp)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static int is_L(uint32_t cp) { return in_ranges(qms_uni_letter, QMS_UNI_LETTER_N, cp); }
static int is_N(uint32_t cp) { return in_ranges(qms_uni_number, QMS_UNI_NUMBER_N, cp); }
static int is_S(uint32_t cp)   /* \s: the Unicode White_Space property */
{
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}
static int is_nl(uint32_t cp) { return cp == '\r' || cp == '\n'; }
static int is_other(uint32_t cp) { return !is_S(cp) && !is_L(cp) && !is_N(cp); }

/* Decode UTF-8; invalid bytes become U+FFFD of length 1 (they still map to their own byte). */
static int utf8_decode(const unsigned char *s, size_t n, size_t i, uint32_t *cp)
{
    const unsigned char c = s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (len == 0 || i + (size_t)len > n) { *cp = 0xFFFD; return 1; }
    uint32_t v = len == 1 ? c : len == 2 ? c & 0x1F : len == 3 ? c & 0x0F : c & 0x07;
    for (int k = 1; k < len; k++) {
        if ((s[i + k] & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        v = (v << 6) | (s[i + k] & 0x3F);
    }
    *cp = v;
    return len;
}

static int utf8_encode(uint32_t cp, char *out)
{
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* ------------------------------------------------------------ hashing ---- */

static uint64_t fnv(const char *s, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h;
}

static int build_hash(const QmsStrArr *a, int **tab, int *mask)
{
    int cap = 1;
    while (cap < a->n * 2) cap <<= 1;
    *tab = (int *)malloc((size_t)cap * sizeof(int));
    if (!*tab) return -1;
    for (int i = 0; i < cap; i++) (*tab)[i] = -1;
    *mask = cap - 1;
    for (int i = 0; i < a->n; i++) {
        size_t h = (size_t)fnv(a->s[i], a->len[i]) & (size_t)*mask;
        while ((*tab)[h] >= 0) h = (h + 1) & (size_t)*mask;
        (*tab)[h] = i;
    }
    return 0;
}

static int lookup(const QmsStrArr *a, const int *tab, int mask, const char *s, size_t n)
{
    size_t h = (size_t)fnv(s, n) & (size_t)mask;
    while (tab[h] >= 0) {
        const int i = tab[h];
        if (a->len[i] == n && !memcmp(a->s[i], s, n)) return i;
        h = (h + 1) & (size_t)mask;
    }
    return -1;
}

int qms_tok_find(const QmsTok *t, const char *s) { return lookup(&t->tok, t->vh, t->vh_mask, s, strlen(s)); }

/* ------------------------------------------------------------ load ---- */

static const QmsTok *g_sort_tok;
static int cmp_special(const void *a, const void *b)
{
    const int x = *(const int *)a, y = *(const int *)b;
    const int lx = (int)g_sort_tok->tok.len[x], ly = (int)g_sort_tok->tok.len[y];
    return lx != ly ? ly - lx : x - y;
}

int qms_tok_load(QmsTok *t, const QmsGguf *g)
{
    memset(t, 0, sizeof *t);
    const QmsKV *model = qms_gguf_kv(g, "tokenizer.ggml.model");
    const QmsKV *pre = qms_gguf_kv(g, "tokenizer.ggml.pre");
    if (!model || strcmp(model->str, "gpt2") || !pre || strcmp(pre->str, "qwen2")) {
        fprintf(stderr, "tok: need a gpt2/qwen2 tokenizer, found %s/%s\n",
                model ? model->str : "?", pre ? pre->str : "?");
        return -1;
    }
    int nt = 0;
    if (qms_gguf_get_strs(g, "tokenizer.ggml.tokens", &t->tok) ||
        qms_gguf_get_strs(g, "tokenizer.ggml.merges", &t->merges) ||
        qms_gguf_get_i32s(g, "tokenizer.ggml.token_type", &t->type, &nt) || nt != t->tok.n) {
        qms_tok_free(t);
        return -1;
    }
    t->n_vocab = t->tok.n;
    if (build_hash(&t->tok, &t->vh, &t->vh_mask) || build_hash(&t->merges, &t->mh, &t->mh_mask)) {
        qms_tok_free(t);
        return -1;
    }
    /* GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to 256 + k */
    int k = 0;
    for (int b = 0; b < 256; b++) {
        const int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        t->byte2cp[b] = printable ? (uint32_t)b : (uint32_t)(256 + k++);
    }
    for (int i = 0; i < 512; i++) t->cp2byte[i] = -1;
    for (int b = 0; b < 256; b++) t->cp2byte[t->byte2cp[b]] = (int16_t)b;

    t->special = (int *)malloc((size_t)t->n_vocab * sizeof(int));
    for (int i = 0; i < t->n_vocab; i++)
        if ((t->type[i] == QMS_TOKT_CONTROL || t->type[i] == QMS_TOKT_USER) && t->tok.len[i]) {
            t->special[t->n_special++] = i;
            t->special_first[(unsigned char)t->tok.s[i][0]] = 1;
        }
    g_sort_tok = t;
    qsort(t->special, (size_t)t->n_special, sizeof(int), cmp_special);

    t->eos = qms_tok_find(t, "<|im_end|>");
    int64_t v;
    if (t->eos < 0 && qms_gguf_get_int(g, "tokenizer.ggml.eos_token_id", &v) == 0) t->eos = (int)v;
    return 0;
}

void qms_tok_free(QmsTok *t)
{
    qms_gguf_free_strs(&t->tok);
    qms_gguf_free_strs(&t->merges);
    free(t->type); free(t->vh); free(t->mh); free(t->special);
    memset(t, 0, sizeof *t);
}

/* ------------------------------------------------------------ encode ---- */

typedef struct { int *v; int n, cap; } IdVec;

static void push(IdVec *o, int id)
{
    if (o->n == o->cap) {
        o->cap = o->cap ? o->cap * 2 : 64;
        o->v = (int *)realloc(o->v, (size_t)o->cap * sizeof(int));
    }
    o->v[o->n++] = id;
}

/* BPE on one pre-token. Symbols are byte ranges of the mapped word, so merging two adjacent
 * symbols is just extending a range. */
static void bpe_word(const QmsTok *t, const unsigned char *bytes, size_t nb, IdVec *out)
{
    char *w = (char *)malloc(nb * 2 + 1);        /* mapped code points are < 0x200: <= 2 bytes */
    int *beg = (int *)malloc((nb + 1) * sizeof(int));
    int *end = (int *)malloc((nb + 1) * sizeof(int));
    char *key = (char *)malloc(nb * 2 + 2);
    int len = 0, ns = 0;
    for (size_t i = 0; i < nb; i++) {
        beg[ns] = len;
        len += utf8_encode(t->byte2cp[bytes[i]], w + len);
        end[ns++] = len;
    }
    const int whole = lookup(&t->tok, t->vh, t->vh_mask, w, (size_t)len);
    if (whole >= 0) { push(out, whole); goto done; }   /* common fast path */
    for (;;) {
        int best = -1, best_rank = 0x7FFFFFFF;
        for (int i = 0; i + 1 < ns; i++) {
            const int la = end[i] - beg[i], lb = end[i + 1] - beg[i + 1];
            memcpy(key, w + beg[i], (size_t)la);
            key[la] = ' ';
            memcpy(key + la + 1, w + beg[i + 1], (size_t)lb);
            const int r = lookup(&t->merges, t->mh, t->mh_mask, key, (size_t)(la + 1 + lb));
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
        }
        if (best < 0) break;
        end[best] = end[best + 1];
        for (int i = best + 1; i + 1 < ns; i++) { beg[i] = beg[i + 1]; end[i] = end[i + 1]; }
        ns--;
    }
    for (int i = 0; i < ns; i++) {
        const int id = lookup(&t->tok, t->vh, t->vh_mask, w + beg[i], (size_t)(end[i] - beg[i]));
        if (id >= 0) { push(out, id); continue; }
        /* not in vocab (should not happen for byte-level BPE): fall back to single bytes */
        for (int p = beg[i]; p < end[i];) {
            uint32_t cp;
            const int l = utf8_decode((const unsigned char *)w, (size_t)end[i], (size_t)p, &cp);
            const int sid = lookup(&t->tok, t->vh, t->vh_mask, w + p, (size_t)l);
            if (sid >= 0) push(out, sid);
            p += l;
        }
    }
done:
    free(w); free(beg); free(end); free(key);
}

/* Pre-tokenizer: returns the length (in code points) of the match at i, never 0. */
static int match_at(const uint32_t *c, int n, int i)
{
    /* 1. contractions (case-insensitive letters) */
    if (c[i] == '\'' && i + 1 < n) {
        const uint32_t a = c[i + 1] | 0x20, b = i + 2 < n ? (c[i + 2] | 0x20) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }
    /* 2. [^\r\n\p{L}\p{N}]?\p{L}+ */
    {
        int j = i;
        if (!is_L(c[j]) && !is_nl(c[j]) && !is_N(c[j]) && j + 1 < n && is_L(c[j + 1])) j++;
        if (is_L(c[j])) {
            while (j < n && is_L(c[j])) j++;
            return j - i;
        }
    }
    /* 3. \p{N}: exactly one */
    if (is_N(c[i])) return 1;
    /* 4. " ?[^\s\p{L}\p{N}]+[\r\n]*" */
    {
        int j = i;
        if (c[j] == ' ' && j + 1 < n && is_other(c[j + 1])) j++;
        if (is_other(c[j])) {
            while (j < n && is_other(c[j])) j++;
            while (j < n && is_nl(c[j])) j++;
            return j - i;
        }
    }
    /* whitespace alternatives operate on the maximal whitespace run [i, e) */
    if (is_S(c[i])) {
        int e = i;
        while (e < n && is_S(c[e])) e++;
        /* 5. \s*[\r\n]+ : ends right after the last newline in the run */
        int last_nl = -1;
        for (int j = i; j < e; j++) if (is_nl(c[j])) last_nl = j;
        if (last_nl >= 0) return last_nl + 1 - i;
        /* 6. \s+(?!\S) : the whole run at end of text, else the run minus its last char */
        if (e == n) return e - i;
        if (e - i >= 2) return e - i - 1;
        /* 7. \s+ */
        return e - i;
    }
    return 1;   /* unreachable for valid input: every code point falls in some class */
}

static void encode_plain(const QmsTok *t, const unsigned char *s, size_t n, IdVec *out)
{
    if (n == 0) return;
    uint32_t *cp = (uint32_t *)malloc(n * sizeof(uint32_t));
    size_t *off = (size_t *)malloc((n + 1) * sizeof(size_t));
    int nc = 0;
    for (size_t i = 0; i < n;) {
        off[nc] = i;
        i += (size_t)utf8_decode(s, n, i, &cp[nc]);
        nc++;
    }
    off[nc] = n;
    for (int i = 0; i < nc;) {
        const int m = match_at(cp, nc, i);
        bpe_word(t, s + off[i], off[i + m] - off[i], out);
        i += m;
    }
    free(cp); free(off);
}

int qms_tok_encode(const QmsTok *t, const char *text, size_t len, int **ids, int *n)
{
    IdVec out = { 0 };
    const unsigned char *s = (const unsigned char *)text;
    size_t start = 0;
    for (size_t i = 0; i < len;) {
        int hit = -1;
        if (t->special_first[s[i]]) {
            for (int k = 0; k < t->n_special; k++) {
                const int id = t->special[k];
                const size_t l = t->tok.len[id];
                if (l && i + l <= len && !memcmp(s + i, t->tok.s[id], l)) { hit = id; break; }
            }
        }
        if (hit < 0) { i++; continue; }
        encode_plain(t, s + start, i - start, &out);
        push(&out, hit);
        i += t->tok.len[hit];
        start = i;
    }
    encode_plain(t, s + start, len - start, &out);
    *ids = out.v;
    *n = out.n;
    return 0;
}

/* ------------------------------------------------------------ decode ---- */

int qms_tok_piece(const QmsTok *t, int id, char *buf, int cap)
{
    if (id < 0 || id >= t->n_vocab) return 0;
    const char *s = t->tok.s[id];
    const int n = (int)t->tok.len[id];
    if (t->type[id] == QMS_TOKT_CONTROL || t->type[id] == QMS_TOKT_USER) {
        const int k = n < cap ? n : cap;
        memcpy(buf, s, (size_t)k);
        return k;
    }
    int o = 0;
    for (int i = 0; i < n && o < cap;) {
        uint32_t cp;
        i += utf8_decode((const unsigned char *)s, (size_t)n, (size_t)i, &cp);
        if (cp < 512 && t->cp2byte[cp] >= 0) buf[o++] = (char)t->cp2byte[cp];
    }
    return o;
}
