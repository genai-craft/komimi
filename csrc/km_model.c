/* .kmm の読み込み (komimi/kmm.py の書式)。 */
#include "km.h"
#include <string.h>
#include <stdio.h>

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

typedef struct { char name[49]; int dtype, ndim, shape[4]; uint32_t doff, soff; } km_entry;

static const km_entry *find(const km_entry *tab, int n, const char *name) {
    for (int i = 0; i < n; i++) if (strcmp(tab[i].name, name) == 0) return &tab[i];
    return NULL;
}

static int set_mat(const km_model *m, const km_entry *tab, int n, const char *name, km_mat *out, const uint8_t *data) {
    const km_entry *e = find(tab, n, name);
    if (!e) { fprintf(stderr, "komimi: missing %s\n", name); return -1; }
    int rows = e->shape[0], cols = 1;
    for (int i = 1; i < e->ndim; i++) cols *= e->shape[i];
    out->rows = rows; out->cols = cols;
    out->bits = e->dtype == 2 ? 4 : 8;
    if (e->dtype == 1 || e->dtype == 2) { out->q = (const int8_t *)(data + e->doff); out->s = (const float *)(data + e->soff); out->f = NULL; }
    else { out->f = (const float *)(data + e->doff); out->q = NULL; out->s = NULL; }
    (void)m; return 0;
}

static const float *vec(const km_entry *tab, int n, const char *name, const uint8_t *data) {
    const km_entry *e = find(tab, n, name);
    if (!e) { fprintf(stderr, "komimi: missing %s\n", name); return NULL; }
    return (const float *)(data + e->doff);
}

int km_load(km_model *m, const uint8_t *buf, size_t size) {
    static km_entry tab[1024];
    if (size < 72 || memcmp(buf, "KMM1", 4) != 0) return -1;
    memset(m, 0, sizeof *m);
    const uint8_t *p = buf + 4;
    int *hdr[16] = { &m->d, &m->heads, &m->ff, &m->kernel, &m->n_layers, &m->n_mel, &m->vocab, &m->sub_ch, &m->flags, &m->chunk, &m->left, &m->sr, &m->hop, &m->win, &m->nfft, &m->bits };
    for (int i = 0; i < 16; i++) *hdr[i] = (int)rd32(p + 4 * i);
    p += 64;
    int n = (int)rd32(p); p += 4;
    if (n > 1024 || m->n_layers > KM_MAX_LAYERS) return -2;
    for (int i = 0; i < n; i++) {
        memcpy(tab[i].name, p, 48); tab[i].name[48] = 0;
        tab[i].dtype = (int)rd32(p + 48); tab[i].ndim = (int)rd32(p + 52);
        for (int k = 0; k < 4; k++) tab[i].shape[k] = (int)rd32(p + 56 + 4 * k);
        tab[i].doff = rd32(p + 72); tab[i].soff = rd32(p + 76);
        p += 80;
    }
    m->n_tokens = (int)rd32(p); p += 4;
    if (m->n_tokens > KM_MAX_TOKENS) return -3;
    for (int i = 0; i < m->n_tokens; i++) { m->tok_len[i] = rd16(p); p += 2; m->tok[i] = (const char *)p; p += m->tok_len[i]; }
    size_t pre = (size_t)(p - buf); pre = (pre + 63) / 64 * 64;
    const uint8_t *data = buf + pre;
    m->base = buf; m->size = size;
    int err = 0;
    err |= set_mat(m, tab, n, "sub.conv1.weight", &m->conv1, data); m->conv1b = vec(tab, n, "sub.conv1.bias", data);
    err |= set_mat(m, tab, n, "sub.conv2.weight", &m->conv2, data); m->conv2b = vec(tab, n, "sub.conv2.bias", data);
    err |= set_mat(m, tab, n, "sub.out.weight", &m->sub_out, data); m->sub_outb = vec(tab, n, "sub.out.bias", data);
    err |= set_mat(m, tab, n, "head.weight", &m->head, data); m->headb = vec(tab, n, "head.bias", data);
    m->window = vec(tab, n, "mel.window", data); m->fb = vec(tab, n, "mel.fb", data);
    if (m->flags & 1) { m->norm_mean = vec(tab, n, "mel.norm_mean", data); m->norm_std = vec(tab, n, "mel.norm_std", data); }
    char nm[96];
    for (int l = 0; l < m->n_layers; l++) {
        km_layer *L = &m->layer[l];
#define MAT(field, suffix) snprintf(nm, sizeof nm, "layers.%d.%s", l, suffix); err |= set_mat(m, tab, n, nm, &L->field, data)
#define VEC(field, suffix) snprintf(nm, sizeof nm, "layers.%d.%s", l, suffix); L->field = vec(tab, n, nm, data); if (!L->field) err = -1
        MAT(ff1_1, "ff1.1.weight"); VEC(ff1_1b, "ff1.1.bias"); MAT(ff1_2, "ff1.2.weight"); VEC(ff1_2b, "ff1.2.bias");
        MAT(ff2_1, "ff2.1.weight"); VEC(ff2_1b, "ff2.1.bias"); MAT(ff2_2, "ff2.2.weight"); VEC(ff2_2b, "ff2.2.bias");
        MAT(q, "att.q.weight"); VEC(qb, "att.q.bias"); MAT(k, "att.k.weight"); VEC(kb, "att.k.bias");
        MAT(v, "att.v.weight"); VEC(vb, "att.v.bias"); MAT(o, "att.out.weight"); VEC(ob, "att.out.bias");
        MAT(pos, "att.pos.weight"); VEC(bias_u, "att.bias_u"); VEC(bias_v, "att.bias_v");
        MAT(pw1, "conv.pw1.weight"); VEC(pw1b, "conv.pw1.bias"); MAT(pw2, "conv.pw2.weight"); VEC(pw2b, "conv.pw2.bias");
        VEC(dw_w, "conv.dw.weight"); VEC(dw_b, "conv.dw.bias");
        VEC(ln_ff1[0], "norm_ff1.weight"); VEC(ln_ff1[1], "norm_ff1.bias"); VEC(ln_att[0], "norm_att.weight"); VEC(ln_att[1], "norm_att.bias");
        VEC(ln_conv[0], "norm_conv.weight"); VEC(ln_conv[1], "norm_conv.bias"); VEC(ln_ff2[0], "norm_ff2.weight"); VEC(ln_ff2[1], "norm_ff2.bias");
        VEC(ln_out[0], "norm_out.weight"); VEC(ln_out[1], "norm_out.bias");
#undef MAT
#undef VEC
    }
    return err ? -4 : 0;
}

int km_detok(const km_model *m, const int *ids, int n, char *out, int cap) {
    int w = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= m->n_tokens) continue;
        const unsigned char *s = (const unsigned char *)m->tok[ids[i]]; int len = m->tok_len[ids[i]];
        for (int k = 0; k < len; k++) {
            if (k + 2 < len && s[k] == 0xE2 && s[k+1] == 0x96 && s[k+2] == 0x81) { if (w > 0 && w + 1 < cap) out[w++] = ' '; k += 2; continue; }
            if (w + 1 < cap) out[w++] = (char)s[k];
        }
    }
    if (cap > 0) out[w < cap ? w : cap - 1] = 0;
    return w;
}

static size_t mat_bytes(const km_mat *w) { return w->q ? (size_t)w->rows * w->cols * (w->bits == 4 ? 1 : 2) / 2 : 0; }

static size_t cache_one(km_mat *w, void *(*alloc)(size_t), size_t *budget) {
    size_t nq = mat_bytes(w), ns = (size_t)w->rows * sizeof(float);
    if (!w->q || nq + ns > *budget) return 0;
    uint8_t *q = (uint8_t *)alloc(nq + 64); float *sc = (float *)alloc(ns);
    if (!q || !sc) return 0;
    /* 16 byte 境界 (PIE のベクタロード) */
    uint8_t *qa = (uint8_t *)(((uintptr_t)q + 15) & ~(uintptr_t)15);
    memcpy(qa, w->q, nq); memcpy(sc, w->s, ns);
    w->q = (const int8_t *)qa; w->s = sc; *budget -= nq + ns;
    return nq + ns;
}

size_t km_model_cache_weights(km_model *m, void *(*alloc)(size_t), size_t budget) {
    size_t total = 0;
    /* 1 チャンクあたりの読み回数が多い順: conv2 (フレームごと) → 層の行列 → sub_out → head */
    total += cache_one(&m->conv2, alloc, &budget);
    total += cache_one(&m->sub_out, alloc, &budget);                /* チャンクあたり数回読む */
    for (int l = 0; l < m->n_layers; l++) {
        km_layer *L = &m->layer[l];
        km_mat *ms[] = { &L->ff1_1, &L->ff1_2, &L->ff2_1, &L->ff2_2, &L->pw1, &L->pw2, &L->q, &L->k, &L->v, &L->o };
        for (size_t i = 0; i < sizeof ms / sizeof *ms; i++) total += cache_one(ms[i], alloc, &budget);
    }
    total += cache_one(&m->head, alloc, &budget);
    for (int l = 0; l < m->n_layers; l++) total += cache_one(&m->layer[l].pos, alloc, &budget);
    return total;
}
