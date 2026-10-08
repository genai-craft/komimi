/* エンコーダ本体 (全文脈版)。層ごとの手順は km.h 冒頭のとおり。 */
#include "km.h"
#include "km_feat.h"
#include "km_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "km_internal.h"

void (*km_par)(km_range_fn fn, void *ctx, int n) = NULL;
static void matvec_multi_(const km_mat *w, const float *X, int n, const float *bias, float *Out, int8_t *xq, float *sx);
int km_dot_variant = 0;
long long km_gemv_us = 0, km_gemv_calls = 0;
long long (*km_clock_us_ref)(void) = NULL;

/* 行列×ベクトル: int8 ならベクトルを量子化して km_gemv_s8、float 重みなら km_gemv_f32。 */
void matvec(const km_mat *w, const float *x, const float *bias, float *out, int8_t *qbuf) {
    /* qbuf は 2 × cols 以上: 前半が量子化した入力、後半が int4 の行展開 */
    if (w->q) {
        float sx = km_quant_vec(x, w->cols, qbuf);
        if (w->bits == 4) km_gemv_s4(w->q, w->s, w->rows, w->cols, qbuf, sx, bias, out, qbuf + w->cols);
        else km_gemv_s8(w->q, w->s, w->rows, w->cols, qbuf, sx, bias, out);
    } else km_gemv_f32(w->f, w->rows, w->cols, x, bias, out);
}

void matvec_multi(const km_mat *w, const float *X, int n, const float *bias, float *Out, int8_t *xq, float *sx) {
    long long t0 = km_clock_us_ref ? km_clock_us_ref() : 0;
    matvec_multi_(w, X, n, bias, Out, xq, sx);
    if (km_clock_us_ref) { km_gemv_us += km_clock_us_ref() - t0; km_gemv_calls++; }
}

typedef struct { const float *X; int cols; int8_t *xq; float *sx; } quant_job;
static void quant_rows(void *ctx, int lo, int hi) {
    const quant_job *j = (const quant_job *)ctx;
    for (int i = lo; i < hi; i++) j->sx[i] = km_quant_vec(j->X + (size_t)i * j->cols, j->cols, j->xq + (size_t)i * j->cols);
}

static void matvec_multi_(const km_mat *w, const float *X, int n, const float *bias, float *Out, int8_t *xq, float *sx) {
    if (!w->q) { for (int i = 0; i < n; i++) km_gemv_f32(w->f, w->rows, w->cols, X + (size_t)i * w->cols, bias, Out + (size_t)i * w->rows); return; }
    quant_job qj = { X, w->cols, xq, sx };
    km_run(quant_rows, &qj, n, 8);                                   /* 入力の量子化も 2 コアで */
    km_gemv_s8_multi(w->q, w->s, w->bits, w->rows, w->cols, xq, sx, n, bias, Out, xq + (size_t)n * w->cols);
}


void layernorm(const float *x, const float *g, const float *b, float *out, int d) {
    float s = 0.f; for (int i = 0; i < d; i++) s += x[i];
    float mean = s / d, v = 0.f;
    for (int i = 0; i < d; i++) { float t = x[i] - mean; v += t * t; }
    float inv = 1.f / sqrtf(v / d + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
}

/* 相対位置 (T-1-r) の正弦埋め込みを pe[r] (r < 2T-1) に。 */
void rel_pos(float *pe, int T, int d) {
    for (int r = 0; r < 2 * T - 1; r++) {
        float p = (float)(T - 1 - r); float *row = pe + (size_t)r * d;
        for (int i = 0; i < d; i += 2) {
            float div = expf((float)i * (-logf(10000.f) / (float)d));
            row[i] = sinf(p * div); row[i + 1] = cosf(p * div);
        }
    }
}

typedef struct { float *x, *tmp, *big, *q, *k, *v, *pe, *P, *sc; int8_t *qb; } ws_t;

/* FF (半分の重みで残差に足す) */
static void feed_forward(const km_mat *w1, const float *b1, const km_mat *w2, const float *b2, const float *const ln[2],
                         float *x, int T, int d, int ff, ws_t *W) {
    for (int t = 0; t < T; t++) {
        float *xt = x + (size_t)t * d;
        layernorm(xt, ln[0], ln[1], W->tmp, d);
        matvec(w1, W->tmp, b1, W->big, W->qb);
        for (int i = 0; i < ff; i++) W->big[i] = silu(W->big[i]);
        matvec(w2, W->big, b2, W->tmp, W->qb);
        for (int i = 0; i < d; i++) xt[i] += 0.5f * W->tmp[i];
    }
}

static void self_attention(const km_layer *L, float *x, int T, int d, int heads, ws_t *W) {
    int dk = d / heads; float scale = 1.f / sqrtf((float)dk);
    for (int t = 0; t < T; t++) {
        layernorm(x + (size_t)t * d, L->ln_att[0], L->ln_att[1], W->tmp, d);
        matvec(&L->q, W->tmp, L->qb, W->q + (size_t)t * d, W->qb);
        matvec(&L->k, W->tmp, L->kb, W->k + (size_t)t * d, W->qb);
        matvec(&L->v, W->tmp, L->vb, W->v + (size_t)t * d, W->qb);
    }
    for (int r = 0; r < 2 * T - 1; r++) matvec(&L->pos, W->pe + (size_t)r * d, NULL, W->P + (size_t)r * d, W->qb);
    for (int t = 0; t < T; t++) {
        float *out = W->big;                               /* d 個: 全ヘッドの出力を並べる */
        for (int h = 0; h < heads; h++) {
            const float *q = W->q + (size_t)t * d + h * dk, *bu = L->bias_u + h * dk, *bv = L->bias_v + h * dk;
            float mx = -1e30f;
            for (int j = 0; j < T; j++) {
                const float *k = W->k + (size_t)j * d + h * dk, *p = W->P + (size_t)(T - 1 - t + j) * d + h * dk;
                float ac = 0.f, bd = 0.f;
                for (int i = 0; i < dk; i++) { ac += (q[i] + bu[i]) * k[i]; bd += (q[i] + bv[i]) * p[i]; }
                float s = (ac + bd) * scale; W->sc[j] = s; if (s > mx) mx = s;
            }
            float sum = 0.f;
            for (int j = 0; j < T; j++) { W->sc[j] = km_expf(W->sc[j] - mx); sum += W->sc[j]; }
            float inv = 1.f / sum; float *o = out + h * dk;
            for (int i = 0; i < dk; i++) o[i] = 0.f;
            for (int j = 0; j < T; j++) { float a = W->sc[j] * inv; const float *v = W->v + (size_t)j * d + h * dk; for (int i = 0; i < dk; i++) o[i] += a * v[i]; }
        }
        matvec(&L->o, out, L->ob, W->tmp, W->qb);
        float *xt = x + (size_t)t * d; for (int i = 0; i < d; i++) xt[i] += W->tmp[i];
    }
}

static void conv_module(const km_layer *L, float *x, int T, int d, int K, ws_t *W) {
    /* GLU 後の列 (T, d) を W->q に溜める (この時点で q/k/v は不要) */
    float *g = W->q;
    for (int t = 0; t < T; t++) {
        layernorm(x + (size_t)t * d, L->ln_conv[0], L->ln_conv[1], W->tmp, d);
        matvec(&L->pw1, W->tmp, L->pw1b, W->big, W->qb);
        float *gt = g + (size_t)t * d;
        for (int i = 0; i < d; i++) gt[i] = W->big[i] * sigmoid(W->big[d + i]);
    }
    int half = K / 2;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < d; c++) {
            const float *w = L->dw_w + (size_t)c * K; float acc = L->dw_b[c];
            int j0 = t - half < 0 ? half - t : 0, j1 = t - half + K > T ? T - (t - half) : K;
            const float *col = g + (size_t)(t - half) * d + c;
            for (int j = j0; j < j1; j++) acc += w[j] * col[(size_t)j * d];
            W->tmp[c] = silu(acc);
        }
        matvec(&L->pw2, W->tmp, L->pw2b, W->big, W->qb);
        float *xt = x + (size_t)t * d; for (int i = 0; i < d; i++) xt[i] += W->big[i];
    }
}

size_t km_workspace_bytes(const km_model *m, int T) {
    size_t d = m->d, fl = sizeof(float);
    return fl * (T * d * 4 + (2 * T - 1) * d * 2 + T + 2 * m->ff + d + 64) + 4096;
}

/* サブサンプリング: feat (Tin, n_mel) → x (Tout, d)、Tout を返す。 */
static int subsample(const km_model *m, const float *feat, int Tin, float *x, int8_t *qb) {
    int F0 = m->n_mel, C = m->sub_ch;
    int T1 = (Tin - 1) / 2 + 1, F1 = (F0 - 1) / 2 + 1;
    int T2 = (T1 - 1) / 2 + 1, F2 = (F1 - 1) / 2 + 1;
    float *h1 = (float *)malloc(sizeof(float) * (size_t)C * T1 * F1);      /* (C, T1, F1) */
    float *col = (float *)malloc(sizeof(float) * (size_t)C * 9 + 64);
    float *o = (float *)malloc(sizeof(float) * (size_t)C);
    float *flat = (float *)malloc(sizeof(float) * (size_t)C * F2);
    if (!h1 || !col || !o || !flat) { free(h1); free(col); free(o); free(flat); return -1; }
    for (int t = 0; t < T1; t++) for (int f = 0; f < F1; f++) {
        for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
            int ti = 2 * t - 1 + kh, fi = 2 * f - 1 + kw;
            col[kh * 3 + kw] = (ti >= 0 && ti < Tin && fi >= 0 && fi < F0) ? feat[(size_t)ti * F0 + fi] : 0.f;
        }
        matvec(&m->conv1, col, m->conv1b, o, qb);
        for (int c = 0; c < C; c++) h1[((size_t)c * T1 + t) * F1 + f] = o[c] > 0.f ? o[c] : 0.f;
    }
    float xs = sqrtf((float)m->d);
    for (int t = 0; t < T2; t++) {
        for (int f = 0; f < F2; f++) {
            for (int c = 0; c < C; c++) for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                int ti = 2 * t - 1 + kh, fi = 2 * f - 1 + kw;
                col[c * 9 + kh * 3 + kw] = (ti >= 0 && ti < T1 && fi >= 0 && fi < F1) ? h1[((size_t)c * T1 + ti) * F1 + fi] : 0.f;
            }
            matvec(&m->conv2, col, m->conv2b, o, qb);
            for (int c = 0; c < C; c++) flat[(size_t)c * F2 + f] = o[c] > 0.f ? o[c] : 0.f;   /* C 優先で平らに */
        }
        matvec(&m->sub_out, flat, m->sub_outb, x + (size_t)t * m->d, qb);
        for (int i = 0; i < m->d; i++) x[(size_t)t * m->d + i] *= xs;
    }
    free(h1); free(col); free(o); free(flat);
    return T2;
}

int km_recognize(const km_model *m, const float *pcm, int n, int *ids, int max_out, int *n_frames_out) {
    int Tin = km_num_frames(m, n), d = m->d;
    float *feat = (float *)malloc(sizeof(float) * (size_t)Tin * m->n_mel);
    float *fw = (float *)malloc(sizeof(float) * (size_t)m->nfft * 3);
    if (!feat || !fw) { free(feat); free(fw); return -1; }
    km_logmel(m, pcm, n, feat, fw); free(fw);
    int Tmax = ((Tin - 1) / 2) / 2 + 1;
    ws_t W; memset(&W, 0, sizeof W);
    size_t fl = sizeof(float);
    W.x = (float *)malloc(fl * Tmax * d); W.q = (float *)malloc(fl * Tmax * d); W.k = (float *)malloc(fl * Tmax * d); W.v = (float *)malloc(fl * Tmax * d);
    W.pe = (float *)malloc(fl * (2 * Tmax) * d); W.P = (float *)malloc(fl * (2 * Tmax) * d); W.sc = (float *)malloc(fl * Tmax);
    W.tmp = (float *)malloc(fl * (d + 64)); W.big = (float *)malloc(fl * (2 * m->ff + 64));
    W.qb = (int8_t *)km_aligned_alloc((size_t)m->sub_out.cols * 3 + 64);
    int T = subsample(m, feat, Tin, W.x, W.qb); free(feat);
    if (T < 0) return -1;
    rel_pos(W.pe, T, d);
    for (int l = 0; l < m->n_layers; l++) {
        const km_layer *L = &m->layer[l];
        feed_forward(&L->ff1_1, L->ff1_1b, &L->ff1_2, L->ff1_2b, L->ln_ff1, W.x, T, d, m->ff, &W);
        self_attention(L, W.x, T, d, m->heads, &W);
        conv_module(L, W.x, T, d, m->kernel, &W);
        feed_forward(&L->ff2_1, L->ff2_1b, &L->ff2_2, L->ff2_2b, L->ln_ff2, W.x, T, d, m->ff, &W);
        for (int t = 0; t < T; t++) { layernorm(W.x + (size_t)t * d, L->ln_out[0], L->ln_out[1], W.tmp, d); memcpy(W.x + (size_t)t * d, W.tmp, fl * d); }
    }
    /* CTC greedy: blank = vocab-1 */
    float *lg = (float *)malloc(fl * m->vocab);
    int blank = m->vocab - 1, prev = blank, n_out = 0;
    for (int t = 0; t < T; t++) {
        matvec(&m->head, W.x + (size_t)t * d, m->headb, lg, W.qb);
        int best = 0; for (int i = 1; i < m->vocab; i++) if (lg[i] > lg[best]) best = i;
        if (best != blank && best != prev && n_out < max_out) ids[n_out++] = best;
        prev = best;
    }
    free(lg);
    free(W.x); free(W.q); free(W.k); free(W.v); free(W.pe); free(W.P); free(W.sc); free(W.tmp); free(W.big); km_aligned_free(W.qb);
    if (n_frames_out) *n_frames_out = T;
    return n_out;
}
