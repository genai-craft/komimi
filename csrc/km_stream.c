/* ストリーミング推論。学習側 (komimi_train/model.py) の chunk/left マスクと同じ値を、チャンク単位の逐次計算で出す。
 *
 * 出力フレーム t2 (40 ms) はメル行 [4·t2-3, 4·t2+4) を見るので、チャンク c (t2 ∈ [cC, cC+C)) は
 * メル行 [4cC-3, 4(c+1)C) だけで計算できる = 先読みはチャンクの外に出ない (遅延 = チャンク長 + 窓半分 12.5 ms)。
 * 各層は K/V (left+C 行) と GLU 出力 (half+C 行) のリングを持ち、相対位置の射影 P は起動時に 1 度だけ作る。
 * -DKM_KV_INT8 で K/V と P を行ごと int8 で持つ (端末向け。作業メモリ 6.5 MB → 1.5 MB、精度は dev で確認)。 */
#include "km_stream.h"
#include "km_internal.h"
#include "km_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#endif

#ifdef KM_ATT_INT8
#ifndef KM_KV_INT8
#define KM_KV_INT8
#endif
#endif

int64_t (*km_clock_us)(void) = NULL;
#define TICK(idx) do { if (km_clock_us) { int64_t _n = km_clock_us(); s->prof[idx] += _n - s->t_last; s->t_last = _n; } } while (0)

struct km_stream {
    const km_model *m;
    int C, left, d, heads, dk, K, half, nl, F0, F1, F2, Cch;
    int conv2_batch;                          /* conv2 を何フレームまとめて回すか (xq が内部 RAM に入る範囲で 2、入らなければ 1) */
    int dkp, ks, LCp;                         /* KM_ATT_INT8: ヘッドの幅を 16 の倍数に詰めた dk、K/P 行の長さ (heads×dkp)、V 転置リングの行長 (left+C を 16 の倍数に) */
    int8_t *vt; float *vts; int8_t *aq; void *vt_raw;   /* KM_ATT_INT8: V 転置 nl×d×LCp、位置ごとの scale nl×LCp、量子化した注意重み (2 コア分) */
    /* 音 */
    float *pcm; long n_total; long pcm_cap;   /* リング (絶対サンプル番号 % pcm_cap)、受け取った総サンプル数。cap はチャンク 1 つ分 + 余裕 */
    long T_total, T1_total;                   /* finish 時に確定する総フレーム数 (それまで -1) */
    /* 作業領域 */
    float *mel;                               /* (4C+3) × n_mel */
    float *h1; long h1_next;                  /* conv1 行のリング (3 行 × Cch × F1)。h1_next = 次に計算する行番号 */
    float *x;                                 /* C × d */
#ifdef KM_KV_INT8
    int8_t *kc, *vc; float *ks_, *vs;         /* nl × (left+C) × d の int8 と行ごとの scale (ks_ は K の scale) */
    int8_t *P; float *Ps;                     /* nl × (left+2C-1) × d */
#else
    float *kc, *vc;                           /* nl × (left+C) × d */
    float *P;                                 /* nl × (left+2C-1) × d */
#endif
    float *gc;                                /* nl × (half+C) × d */
    float *kf, *vf;                           /* 1 行ぶんの float 作業 (int8 を戻す) */
    float *win;                               /* 窓 (RAM に写し)、疎メル: fb_w (重みを詰めたもの)、fb_off/fb_len (フィルタごとの bin 範囲) */
    float *fb_w; int *fb_off, *fb_len, *fb_pos;
    float *conv1w;                            /* conv1 (Cch×9) の float 重み (RAM)。9 タップは float で直接回す方が速い */
    float *dwT;                               /* 現在の層の dw 重みを (K, d) 配置で (内部 RAM、層の頭でコピー) */
    km_mat conv2p; void *conv2p_raw; int c2p;  /* conv2 (int8) の列を (タップ, チャネル) 順に並べ替え、列数を 16 の倍数に零詰めした写し (c2p = 詰めた列数) */
    /* チャンクまとめ処理用: Xn (n×d) / Q (n×d) / B (n×d) / O (n×d) / A (n×max(ff,2d)) / S (heads×n×(left+C)) / Hd (n×max(vocab, sub_cols)) */
    float *Xn, *Q, *B, *O, *A, *S, *Hd, *sx;
    int8_t *qu8, *qv8; float *squ, *sqv;
    void *kc_raw, *P_raw;      /* KM_ATT_INT8: (q+bias_u), (q+bias_v) を int8 にしたもの (n×ks) と scale */
    int8_t *xq;                               /* 量子化した入力 (max(n,F2) 本 × 最大 cols) + 展開行 */
    float *tmp, *big, *sc, *col, *o, *flat, *lg, *fw;
    int8_t *qb;
    /* 状態 */
    long frames_done; int prev, finished;
    float silence_db; long ctx_start; int skipped;      /* 無音ゲート: しきい値 (dBFS、0 = 無効)、文脈の開始フレーム (飛ばした直後はリセット)、飛ばしたチャンク数 */
    int *ids; int *id_frame; int n_ids, cap_ids;
    int dedup_prefix;                         /* 1: 直前の piece が隣接フレームで出た次の piece の接頭辞なら捨てる (CTC+サブワードの「タ」「タイムズ」対策) */
    void (*tap)(void *, int, const float *, int); void *tap_ctx;
    size_t bytes, hot_psram;                  /* hot_psram: 内部 RAM に入らず PSRAM に落ちた作業バッファの bytes */
    int64_t prof[32], t_last;
};

const int64_t *km_stream_profile(const km_stream *s) { return s->prof; }

static void *zalloc(km_stream *s, size_t n) { s->bytes += n; void *p = calloc(1, n); return p; }
/* 毎フレーム読み書きする小さなバッファ: ESP では内部 SRAM に置く (PSRAM はキャッシュ越しで遅い)。MALLOC_CAP_DMA で RTC (LP) RAM を除く: そこへの SIMD ベクタロードは Load access fault になる */
static void *hot_alloc(km_stream *s, size_t n) {
    s->bytes += n;
#ifdef ESP_PLATFORM
    void *p = heap_caps_calloc(1, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA); if (p) return p;
    s->hot_psram += n;
#endif
    return calloc(1, n);
}

/* (d,) をヘッドごとに dkp 幅へ零詰めして (ks,) に */
static inline void pad_heads(const km_stream *s, const float *src, float *dst) {
    for (int h = 0; h < s->heads; h++) {
        for (int e = 0; e < s->dk; e++) dst[h * s->dkp + e] = src[h * s->dk + e];
        for (int e = s->dk; e < s->dkp; e++) dst[h * s->dkp + e] = 0.f;
    }
}

km_stream *km_stream_new(const km_model *m, int chunk, int left) {
    if (!(m->flags & 1)) return NULL;                       /* 固定標準化が要る */
    km_stream *s = (km_stream *)calloc(1, sizeof *s);
    s->m = m; s->C = chunk; s->left = left; s->d = m->d; s->heads = m->heads; s->dk = m->d / m->heads;
    s->K = m->kernel; s->half = m->kernel / 2; s->nl = m->n_layers; s->Cch = m->sub_ch;
    s->F0 = m->n_mel; s->F1 = (s->F0 - 1) / 2 + 1; s->F2 = (s->F1 - 1) / 2 + 1;
    s->dkp = (s->dk + 15) / 16 * 16; s->ks = s->heads * s->dkp; s->LCp = (left + chunk + 15) / 16 * 16;   /* 下の確保で使うので先に */
    int C = chunk, d = m->d;
    s->pcm_cap = (long)(4 * C + 8) * m->hop + 2 * m->win;        /* chunk 32 なら約 21k サンプル (以前は固定 16384 で溢れていた) */
    s->pcm = (float *)zalloc(s, sizeof(float) * s->pcm_cap);
    s->mel = (float *)zalloc(s, sizeof(float) * (4 * C + 3) * s->F0);
    /* 確保順 = 内部 RAM に残したい順 (足りないと後ろから PSRAM に落ちる): fw/xq/A/Xn.. → 小物 → h1 */
    s->fw = (float *)hot_alloc(s, sizeof(float) * m->nfft * 3);        /* FFT 作業 */
    {
        int abig = m->ff > 2 * d ? m->ff : 2 * d;
        int grp = C < 8 ? C : 8;                                   /* sub_out は grp 本ずつ (xq を内部 RAM に収めるため) */
        /* xq (PIE の入力。最優先で内部 RAM に): conv2 は 2 フレーム同時が入るならそれ、入らなければ 1 フレーム */
        size_t xq2 = (size_t)(grp + 1) * m->sub_out.cols, xq3 = (size_t)(C + 1) * abig;
        size_t xq_base = xq2 > xq3 ? xq2 : xq3;
        int c2pad = (m->conv2.cols + 15) / 16 * 16;
        size_t xq_c2 = (size_t)(2 * s->F2 + 1) * c2pad, xq_c1 = (size_t)(s->F2 + 1) * c2pad;
        size_t xqn = (xq_c2 > xq_base ? xq_c2 : xq_base) + (size_t)m->sub_out.cols + 64;
        s->conv2_batch = 2;
#ifdef ESP_PLATFORM
        s->xq = (int8_t *)heap_caps_aligned_alloc(16, xqn, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA);
        if (!s->xq) {                                                  /* 2 フレーム分が入らない (S3): 1 フレームで */
            s->conv2_batch = 1; xqn = (xq_c1 > xq_base ? xq_c1 : xq_base) + (size_t)m->sub_out.cols + 64;
            s->xq = (int8_t *)heap_caps_aligned_alloc(16, xqn, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA);
        }
        if (!s->xq) { s->xq = (int8_t *)heap_caps_aligned_alloc(16, xqn, MALLOC_CAP_8BIT); s->hot_psram += xqn; }
#else
        s->xq = (int8_t *)km_aligned_alloc(xqn);
#endif
        s->bytes += xqn;
        int nb = C > s->conv2_batch * s->F2 ? C : s->conv2_batch * s->F2;
        s->Xn = (float *)hot_alloc(s, sizeof(float) * C * d); s->Q = (float *)hot_alloc(s, sizeof(float) * C * d);
        s->B = (float *)hot_alloc(s, sizeof(float) * C * d); s->O = (float *)hot_alloc(s, sizeof(float) * C * d);
        size_t an = (size_t)C * abig; if (an < (size_t)s->conv2_batch * s->F2 * s->Cch) an = (size_t)s->conv2_batch * s->F2 * s->Cch;
        if (an < (size_t)C * 2 * d + (size_t)(2 * s->half + C) * d) an = (size_t)C * 2 * d + (size_t)(2 * s->half + C) * d;   /* pw1 出力 + GLU の窓 */
        s->A = (float *)hot_alloc(s, sizeof(float) * an); s->S = (float *)zalloc(s, sizeof(float) * m->heads * C * (left + C));
        s->Hd = (float *)zalloc(s, sizeof(float) * C * (m->vocab > m->sub_out.cols ? m->vocab : m->sub_out.cols)); s->sx = (float *)hot_alloc(s, sizeof(float) * nb);
#ifdef KM_ATT_INT8
        s->qu8 = (int8_t *)hot_alloc(s, (size_t)C * s->ks + 16); s->qv8 = (int8_t *)hot_alloc(s, (size_t)C * s->ks + 16);
        s->qu8 = (int8_t *)(((uintptr_t)s->qu8 + 15) & ~(uintptr_t)15); s->qv8 = (int8_t *)(((uintptr_t)s->qv8 + 15) & ~(uintptr_t)15);
        s->squ = (float *)hot_alloc(s, sizeof(float) * C); s->sqv = (float *)hot_alloc(s, sizeof(float) * C);
#endif
    }
    s->x = (float *)hot_alloc(s, sizeof(float) * C * d);
    int npos = left + 2 * C - 1;

#ifdef KM_ATT_INT8
    int ks = s->ks;
#else
    int ks = d;
#endif
#ifdef KM_KV_INT8
    s->kc = (int8_t *)zalloc(s, (size_t)s->nl * (left + C) * ks + 16); s->ks_ = (float *)zalloc(s, sizeof(float) * s->nl * (left + C));
    s->vc = (int8_t *)zalloc(s, (size_t)s->nl * (left + C) * d); s->vs = (float *)zalloc(s, sizeof(float) * s->nl * (left + C));
    s->P = (int8_t *)zalloc(s, (size_t)s->nl * npos * ks + 16); s->Ps = (float *)zalloc(s, sizeof(float) * s->nl * npos);
    s->kc_raw = s->kc; s->P_raw = s->P;
    s->kc = (int8_t *)(((uintptr_t)s->kc + 15) & ~(uintptr_t)15); s->P = (int8_t *)(((uintptr_t)s->P + 15) & ~(uintptr_t)15);   /* 16 byte 境界 (free は元ポインタで要るので下で保持) */
#else
    s->kc = (float *)zalloc(s, sizeof(float) * s->nl * (left + C) * d);
    s->vc = (float *)zalloc(s, sizeof(float) * s->nl * (left + C) * d);
    s->P = (float *)zalloc(s, sizeof(float) * s->nl * npos * d);
#endif
    s->gc = (float *)zalloc(s, sizeof(float) * s->nl * (s->half + C) * d);
    s->kf = (float *)hot_alloc(s, sizeof(float) * (d + 64)); s->vf = (float *)hot_alloc(s, sizeof(float) * (d + 64));
#ifdef KM_ATT_INT8
    s->vt_raw = zalloc(s, (size_t)s->nl * d * s->LCp + 16); s->vt = (int8_t *)(((uintptr_t)s->vt_raw + 15) & ~(uintptr_t)15);
    s->vts = (float *)zalloc(s, sizeof(float) * s->nl * s->LCp);
    s->aq = (int8_t *)hot_alloc(s, (size_t)2 * s->LCp + 32); s->aq = (int8_t *)(((uintptr_t)s->aq + 15) & ~(uintptr_t)15);
#endif   /* ATT_INT8 ではヘッド詰めで ks (≤ d+64) 要素書く */
    {   /* conv1 の重みを float で RAM に (int8 なら戻す) */
        int rows = m->conv1.rows, cols = m->conv1.cols;
        s->conv1w = (float *)hot_alloc(s, sizeof(float) * rows * cols);
        s->dwT = (float *)hot_alloc(s, sizeof(float) * s->K * d);
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++)
            s->conv1w[r * cols + c] = m->conv1.q ? m->conv1.q[(size_t)r * cols + c] * m->conv1.s[r] : m->conv1.f[(size_t)r * cols + c];
    }
    {   /* 窓と疎メルを RAM に (フラッシュの 82 KB をフレームごとに読まない) */
        int nb = m->nfft / 2 + 1, nm_ = m->n_mel, total = 0;
        s->win = (float *)hot_alloc(s, sizeof(float) * m->win); memcpy(s->win, m->window, sizeof(float) * m->win);
        s->fb_off = (int *)hot_alloc(s, sizeof(int) * nm_); s->fb_len = (int *)hot_alloc(s, sizeof(int) * nm_); s->fb_pos = (int *)hot_alloc(s, sizeof(int) * nm_);
        for (int k = 0; k < nm_; k++) {
            const float *f = m->fb + (size_t)k * nb; int lo = 0, hi = nb - 1;
            while (lo < nb && f[lo] == 0.f) lo++;
            while (hi > lo && f[hi] == 0.f) hi--;
            s->fb_off[k] = lo; s->fb_len[k] = (lo < nb) ? hi - lo + 1 : 0; s->fb_pos[k] = total; total += s->fb_len[k];
        }
        s->fb_w = (float *)hot_alloc(s, sizeof(float) * (total + 1));
        for (int k = 0; k < nm_; k++) memcpy(s->fb_w + s->fb_pos[k], m->fb + (size_t)k * nb + s->fb_off[k], sizeof(float) * s->fb_len[k]);
    }
    s->tmp = (float *)hot_alloc(s, sizeof(float) * (d + 64)); s->big = (float *)hot_alloc(s, sizeof(float) * (2 * m->ff + 64));
    s->sc = (float *)hot_alloc(s, sizeof(float) * (left + C)); s->col = (float *)zalloc(s, sizeof(float) * ((s->Cch * 9 + 15) / 16 * 16 + 64));   /* col/flat/lg は書いて 1 回読むだけ → PSRAM (S3 の内部 RAM を A/Xn に回す) */
    s->o = (float *)hot_alloc(s, sizeof(float) * s->Cch); s->flat = (float *)zalloc(s, sizeof(float) * s->Cch * s->F2);
    s->lg = (float *)zalloc(s, sizeof(float) * m->vocab);
#ifdef ESP_PLATFORM
    s->qb = (int8_t *)heap_caps_aligned_alloc(16, (size_t)m->sub_out.cols * 3 + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA);
    if (!s->qb) s->qb = (int8_t *)heap_caps_aligned_alloc(16, (size_t)m->sub_out.cols * 3 + 64, MALLOC_CAP_8BIT);   /* 内部が足りなければ PSRAM */
#else
    s->qb = (int8_t *)km_aligned_alloc((size_t)m->sub_out.cols * 3 + 64);
#endif
    s->bytes += (size_t)m->sub_out.cols * 3 + 64;
    s->h1 = (float *)hot_alloc(s, sizeof(float) * 3 * s->Cch * s->F1);     /* conv2 の gather が毎フレーム読むので内部 RAM (84 KB) */
    s->conv2p = m->conv2; s->c2p = m->conv2.cols;
    if (m->conv2.q && m->conv2.bits == 8) {                            /* 列 c*9+kk → kk*Cch+c、列数は 16 の倍数に (端は 0) */
        int rows = m->conv2.rows, cols = m->conv2.cols, cp = (cols + 15) / 16 * 16;
        s->conv2p_raw = zalloc(s, (size_t)rows * cp + 16);
        int8_t *q = (int8_t *)(((uintptr_t)s->conv2p_raw + 15) & ~(uintptr_t)15);
        for (int r = 0; r < rows; r++) for (int c = 0; c < s->Cch; c++) for (int kk = 0; kk < 9; kk++)
            q[(size_t)r * cp + kk * s->Cch + c] = m->conv2.q[(size_t)r * cols + c * 9 + kk];
        s->conv2p.q = q; s->conv2p.cols = cp; s->c2p = cp;
    }
    s->cap_ids = 4096; s->ids = (int *)zalloc(s, sizeof(int) * s->cap_ids); s->id_frame = (int *)zalloc(s, sizeof(int) * s->cap_ids);
    s->dedup_prefix = 0;                                             /* 既定 off: 合成音でも実会話でも効果が無く (dev −0.06pt)、本物の繰り返し (ココナッツ) を壊す懸念のほうが大きい */
    /* 相対位置 (left+C-1) … -(C-1) の射影。rel_pos(T=left+C) の先頭 npos 行がちょうどそれ */
    float *pe = (float *)malloc(sizeof(float) * (2 * (left + C) - 1) * d);
    rel_pos(pe, left + C, d);
    for (int l = 0; l < s->nl; l++)
        for (int r = 0; r < npos; r++) {
#ifdef KM_ATT_INT8
            matvec(&m->layer[l].pos, pe + (size_t)r * d, NULL, s->kf, s->qb);
            pad_heads(s, s->kf, s->vf);
            s->Ps[(size_t)l * npos + r] = km_quant_vec(s->vf, s->ks, s->P + ((size_t)l * npos + r) * s->ks);
#elif defined(KM_KV_INT8)
            matvec(&m->layer[l].pos, pe + (size_t)r * d, NULL, s->kf, s->qb);
            s->Ps[(size_t)l * npos + r] = km_quant_vec(s->kf, d, s->P + ((size_t)l * npos + r) * d);
#else
            matvec(&m->layer[l].pos, pe + (size_t)r * d, NULL, s->P + ((size_t)l * npos + r) * d, s->qb);
#endif
        }
    free(pe);
    km_stream_reset(s);
    return s;
}

void km_stream_free(km_stream *s) {
    if (!s) return;
    free(s->pcm); free(s->mel); free(s->h1); free(s->x); free(s->vc); free(s->gc); free(s->kf); free(s->vf);
#ifdef KM_KV_INT8
    free(s->kc_raw); free(s->P_raw);
#else
    free(s->kc); free(s->P);
#endif
#ifdef KM_KV_INT8
    free(s->ks_); free(s->vs); free(s->Ps);
#endif
#ifdef KM_ATT_INT8
    free(s->vt_raw); free(s->vts);
#endif
    free(s->tmp); free(s->big); free(s->sc); free(s->col); free(s->o); free(s->flat); free(s->lg); free(s->fw); free(s->conv1w); free(s->dwT); free(s->conv2p_raw); free(s->win); free(s->fb_w); free(s->fb_off); free(s->fb_len); free(s->fb_pos); free(s->Xn); free(s->Q); free(s->B); free(s->O); free(s->A); free(s->S); free(s->Hd); free(s->sx);
#ifdef ESP_PLATFORM
    heap_caps_free(s->qb); heap_caps_free(s->xq);
#else
    km_aligned_free(s->qb); km_aligned_free(s->xq);
#endif
    free(s->ids); free(s->id_frame); free(s);
}

void km_stream_reset(km_stream *s) {
    s->n_total = 0; s->T_total = -1; s->T1_total = -1; s->frames_done = 0; s->prev = s->m->vocab - 1; s->finished = 0; s->n_ids = 0;
    s->ctx_start = 0; s->skipped = 0;
    memset(s->pcm, 0, sizeof(float) * s->pcm_cap);
    memset(s->gc, 0, sizeof(float) * s->nl * (s->half + s->C) * s->d);
    s->h1_next = 0;
}

void km_stream_set_tap(km_stream *s, void (*tap)(void *, int, const float *, int), void *ctx) { s->tap = tap; s->tap_ctx = ctx; }
void km_stream_set_silence_gate(km_stream *s, float dbfs) { s->silence_db = dbfs; }
int km_stream_skipped(const km_stream *s) { return s->skipped; }

static inline float sample(const km_stream *s, long i);

/* チャンク [t0, t0+n) に対応する音 (フレーム 4t0 .. 4(t0+n) ぶんのサンプル) の RMS (dBFS) */
static float chunk_dbfs(const km_stream *s, long t0, int n) {
    long a = 4 * t0 * (long)s->m->hop, b = 4 * (t0 + n) * (long)s->m->hop; if (b > s->n_total) b = s->n_total;
    double e = 0; long cnt = 0;
    for (long i = a; i < b; i++) { float v = sample(s, i); e += (double)v * v; cnt++; }
    if (cnt == 0) return -120.f;
    return (float)(10.0 * log10(e / cnt + 1e-12));
}

/* 無音チャンクを飛ばす: 出力フレームだけ進め、以後の左文脈をこのチャンクの終わりからにする */
static void skip_chunk(km_stream *s, long t0, int n) {
    s->frames_done = t0 + n; s->ctx_start = t0 + n; s->h1_next = 2 * (t0 + n) - 1; s->prev = s->m->vocab - 1; s->skipped++;
}
int km_stream_tokens(const km_stream *s, const int **ids) { *ids = s->ids; return s->n_ids; }
void km_stream_reset_tokens(km_stream *s) { s->n_ids = 0; }
void km_stream_set_dedup_prefix(km_stream *s, int on) { s->dedup_prefix = on; }
int km_stream_frames(const km_stream *s) { return (int)s->frames_done; }
size_t km_stream_bytes(const km_stream *s) { return s->bytes; }
size_t km_stream_hot_psram(const km_stream *s) { return s->hot_psram; }
int km_stream_conv2_batch(const km_stream *s) { return s->conv2_batch; }

static inline float sample(const km_stream *s, long i) {
    if (i < 0 || i >= s->n_total) return 0.f;
    return s->pcm[i % s->pcm_cap];
}

/* メル行 t (絶対) を out に。系列の末尾が確定していて t ≥ T_total なら 0。 */
static void mel_row(const km_stream *s, long t, float *out) {
    const km_model *m = s->m;
    if (s->T_total >= 0 && t >= s->T_total) { memset(out, 0, sizeof(float) * m->n_mel); return; }
    int half = m->win / 2, nfft = m->nfft, nb = nfft / 2 + 1;
    float *re = s->fw, *im = s->fw + nfft, *pw = s->fw + 2 * nfft;
    long start = t * m->hop - half;
    for (int i = 0; i < nfft; i++) { re[i] = 0.f; im[i] = 0.f; }
    {   /* リング添字を剰余なしで進める (1 サンプル 2 回の除算が 128 フレーム × 400 で効いていた) */
        long k = start - 1; long cap = s->pcm_cap;
        long idx = ((k % cap) + cap) % cap;
        float prev = (k >= 0 && k < s->n_total) ? s->pcm[idx] : 0.f;
        for (int i = 0; i < m->win; i++) {
            k++; idx++; if (idx == cap) idx = 0;
            float cur = (k >= 0 && k < s->n_total) ? s->pcm[idx] : 0.f;
            re[i] = (k >= -half) ? (cur - 0.97f * prev) * s->win[i] : 0.f;
            prev = cur;
        }
    }
    extern void km_fft_inplace(float *re, float *im, int n);
    km_fft_inplace(re, im, nfft);
    for (int i = 0; i < nb; i++) pw[i] = re[i] * re[i] + im[i] * im[i];
    for (int k = 0; k < m->n_mel; k++) {
        const float *f = s->fb_w + s->fb_pos[k]; const float *p = pw + s->fb_off[k]; float acc = 0.f;
        for (int i = 0, n_ = s->fb_len[k]; i < n_; i++) acc += f[i] * p[i];
        out[k] = (logf(acc + 5.9604644775390625e-08f) - m->norm_mean[k]) / (m->norm_std[k] + 1e-5f);
    }
}

static inline void kv_row_to_float(const km_stream *s, int l, int which, long j, float *dst) {
    int LC = s->left + s->C, d = s->d;
#ifdef KM_KV_INT8
    int stride = d;
#ifdef KM_ATT_INT8
    if (!which) stride = s->ks;
#endif
    const int8_t *src = (which ? s->vc : s->kc) + ((size_t)l * LC + (size_t)(j % LC)) * stride;
    float sc = (which ? s->vs : s->ks_)[(size_t)l * LC + (size_t)(j % LC)];
    for (int e = 0; e < stride; e++) dst[e] = src[e] * sc;
#else
    memcpy(dst, (which ? s->vc : s->kc) + ((size_t)l * LC + (size_t)(j % LC)) * d, sizeof(float) * d);
#endif
}

typedef struct { km_stream *s; const km_layer *L; int l; long t0; int n; long j0, j1; int nk; long base; int npos; long rmin; } att_job;

/* K 行を float に戻す (呼び出し側ごとに別バッファ: 並列時は 2 本) */
static inline void k_row(const km_stream *s, int l, long j, float *dst) { kv_row_to_float(s, l, 0, j, dst); }

static void att_scores_keys(void *ctx, int lo, int hi) {
    const att_job *a = (const att_job *)ctx; km_stream *s = a->s; const km_layer *L = a->L;
    int d = s->d, dk = s->dk, heads = s->heads, n = a->n, nk = a->nk;
#ifdef KM_ATT_INT8
    int LC = s->left + s->C, ks = s->ks, dkp = s->dkp; (void)d; (void)dk; (void)L;
    for (long j = a->j0 + lo; j < a->j0 + hi; j++) {
        const int8_t *krow = s->kc + ((size_t)a->l * LC + (size_t)(j % LC)) * ks; float sk = s->ks_[(size_t)a->l * LC + (size_t)(j % LC)];
        for (int h = 0; h < heads; h++) {
            const int8_t *kh = krow + h * dkp;
            for (int i = 0; i < n; i++)
                s->S[((size_t)h * n + i) * nk + (j - a->j0)] += (float)km_dot_s8(s->qu8 + (size_t)i * ks + h * dkp, kh, dkp) * s->squ[i] * sk;
        }
    }
    return;
#endif
    float kbuf[1024];                                                /* d ≤ 1024 */
    for (long j = a->j0 + lo; j < a->j0 + hi; j++) {
        k_row(s, a->l, j, kbuf);
        for (int h = 0; h < heads; h++) {
            const float *kh = kbuf + h * dk, *bu = L->bias_u + h * dk;
            for (int i = 0; i < n; i++) {
                const float *q = s->Q + (size_t)i * d + h * dk; float ac = 0.f;
                for (int e = 0; e < dk; e++) ac += (q[e] + bu[e]) * kh[e];
                s->S[((size_t)h * n + i) * nk + (j - a->j0)] += ac;
            }
        }
    }
}

static void att_scores_pos(void *ctx, int lo, int hi) {
    const att_job *a = (const att_job *)ctx; km_stream *s = a->s; const km_layer *L = a->L;
    int d = s->d, dk = s->dk, heads = s->heads, n = a->n, nk = a->nk;
#ifdef KM_ATT_INT8
    int ks = s->ks, dkp = s->dkp; (void)d; (void)dk; (void)L;
    for (long r = a->rmin + lo; r < a->rmin + hi; r++) {
        const int8_t *prow = s->P + ((size_t)a->l * a->npos + r) * ks; float sp = s->Ps[(size_t)a->l * a->npos + r];
        for (int i = 0; i < n; i++) {
            long j = (a->t0 + i) - (a->base - r);
            if (j < a->j0 || j >= a->j1) continue;
            for (int h = 0; h < heads; h++)
                s->S[((size_t)h * n + i) * nk + (j - a->j0)] += (float)km_dot_s8(s->qv8 + (size_t)i * ks + h * dkp, prow + h * dkp, dkp) * s->sqv[i] * sp;
        }
    }
    return;
#endif
    float pbuf[1024]; (void)pbuf;
    for (long r = a->rmin + lo; r < a->rmin + hi; r++) {
#ifdef KM_KV_INT8
        { const int8_t *pr = s->P + ((size_t)a->l * a->npos + r) * d; float sc = s->Ps[(size_t)a->l * a->npos + r]; for (int e = 0; e < d; e++) pbuf[e] = pr[e] * sc; }
        const float *prow = pbuf;
#else
        const float *prow = s->P + ((size_t)a->l * a->npos + r) * d;
#endif
        for (int i = 0; i < n; i++) {
            long j = (a->t0 + i) - (a->base - r);
            if (j < a->j0 || j >= a->j1) continue;
            for (int h = 0; h < heads; h++) {
                const float *q = s->Q + (size_t)i * d + h * dk, *bv = L->bias_v + h * dk, *ph = prow + h * dk; float bd = 0.f;
                for (int e = 0; e < dk; e++) bd += (q[e] + bv[e]) * ph[e];
                s->S[((size_t)h * n + i) * nk + (j - a->j0)] += bd;
            }
        }
    }
}

static void att_values_heads(void *ctx, int lo, int hi) {
    const att_job *a = (const att_job *)ctx; km_stream *s = a->s;
    int d = s->d, dk = s->dk, n = a->n, nk = a->nk;
#ifdef KM_ATT_INT8
    {   /* O[i, h, e] = Σ_j a_ij V_j[e] を、リング位置 p で並べた a''[p] = a_ij·s_V[p] (int8) と V^T[e][p] (int8) の内積で */
        int LC = s->left + s->C, LCp = s->LCp;
        int8_t *aq = s->aq + (lo == 0 ? 0 : LCp);                     /* 2 コアで別バッファ */
        float af[1024];                                                 /* LCp ≤ 1024 */
        const int8_t *vt = s->vt + (size_t)a->l * d * LCp; const float *vts = s->vts + (size_t)a->l * LCp;
        for (int h = lo; h < hi; h++) {
            for (int i = 0; i < n; i++) {
                for (int p = 0; p < LCp; p++) af[p] = 0.f;
                const float *srow = s->S + ((size_t)h * n + i) * nk;
                for (long j = a->j0; j < a->j1; j++) { int p = (int)(j % LC); af[p] = srow[j - a->j0] * vts[p]; }
                float sa = km_quant_vec(af, LCp, aq);
                float *oh = s->O + (size_t)i * d + h * dk;
                for (int e = 0; e < dk; e++) oh[e] = (float)km_dot_s8(aq, vt + (size_t)(h * dk + e) * LCp, LCp) * sa;
            }
        }
        return;
    }
#endif
    float vbuf[1024];
    for (long j = a->j0; j < a->j1; j++) {
        kv_row_to_float(s, a->l, 1, j, vbuf);
        for (int h = lo; h < hi; h++) {
            const float *vh = vbuf + h * dk;
            for (int i = 0; i < n; i++) {
                float av = s->S[((size_t)h * n + i) * nk + (j - a->j0)]; float *oh = s->O + (size_t)i * d + h * dk;
                for (int e = 0; e < dk; e++) oh[e] += av * vh[e];
            }
        }
    }
}

/* piece a の文字列 (先頭の ▁ を除く) が piece b の真の接頭辞か */
static int piece_is_prefix(const km_model *m, int a, int b) {
    if (a < 0 || b < 0 || a >= m->n_tokens || b >= m->n_tokens) return 0;
    const unsigned char *pa = (const unsigned char *)m->tok[a], *pb = (const unsigned char *)m->tok[b]; int la = m->tok_len[a], lb = m->tok_len[b];
    if (la >= 3 && pa[0] == 0xE2 && pa[1] == 0x96 && pa[2] == 0x81) { pa += 3; la -= 3; }
    if (lb >= 3 && pb[0] == 0xE2 && pb[1] == 0x96 && pb[2] == 0x81) { pb += 3; lb -= 3; }
    return la > 0 && la < lb && memcmp(pa, pb, (size_t)la) == 0;
}

/* 出力フレーム [t0, t0+n) を処理する (n ≤ C)。n 本のフレームをまとめて進め、重みの各行は 1 チャンクに 1 回だけ読む。 */

static void process(km_stream *s, long t0, int n) {
    const km_model *m = s->m; int C = s->C, d = s->d, Cch = s->Cch, F0 = s->F0, F1 = s->F1, F2 = s->F2;
    if (km_clock_us) s->t_last = km_clock_us();
    /* 1. メル行 [4t0-3, 4(t0+n)) */
    long m0 = 4 * t0 - 3; int nm = 4 * n + 3;
    for (int i = 0; i < nm; i++) {
        long t = m0 + i;
        if (t < 0) memset(s->mel + (size_t)i * F0, 0, sizeof(float) * F0); else mel_row(s, t, s->mel + (size_t)i * F0);
    }
    TICK(0);
    /* 2. サブサンプリング。conv1 行は 3 行のリングで逐次、conv2 は 1 フレームの F2 本の列をまとめて、sub_out は n フレームまとめて */
    int sub_cols = m->sub_out.cols;
    int c2 = m->conv2.cols, c2u = s->c2p;                              /* Cch*9 と、16 の倍数に詰めた列数 (並べ替え写しがあるとき) */
    int perm = (s->conv2p.q != m->conv2.q);                            /* 並べ替え済みなら列は [kk][c] (零詰めあり)、そうでなければ [c][kk] */
    for (int i0 = 0; i0 < n; i0 += s->conv2_batch) {                  /* conv2_batch フレームずつ: 重みを読む回数が減る */
        int nf = n - i0 < s->conv2_batch ? n - i0 : s->conv2_batch;
        for (int fi_ = 0; fi_ < nf; fi_++) {
            long t2 = t0 + i0 + fi_;
            while (s->h1_next <= 2 * t2 + 1) {
                long t1 = s->h1_next; float *dst = s->h1 + (size_t)(t1 % 3) * Cch * F1;
                if (s->T1_total >= 0 && t1 >= s->T1_total) memset(dst, 0, sizeof(float) * Cch * F1);
                else for (int f = 0; f < F1; f++) {
                    float col9[9];
                    for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                        long ti = 2 * t1 - 1 + kh; int fi = 2 * f - 1 + kw; long mi = ti - m0;
                        col9[kh * 3 + kw] = (ti >= 0 && mi >= 0 && mi < nm && fi >= 0 && fi < F0) ? s->mel[(size_t)mi * F0 + fi] : 0.f;
                    }
                    float *drow = dst + (size_t)f * Cch;
                    for (int c = 0; c < Cch; c++) {
                        const float *w = s->conv1w + c * 9; float acc = m->conv1b[c];
                        for (int k = 0; k < 9; k++) acc += w[k] * col9[k];
                        drow[c] = acc > 0.f ? acc : 0.f;
                    }
                }
                s->h1_next++;
            }
            /* このフレームの F2 本の列 → xq の [fi_*F2 .. ) */
            for (int f = 0; f < F2; f++) {
                for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                    long t1 = 2 * t2 - 1 + kh; int fi = 2 * f - 1 + kw; int kk = kh * 3 + kw;
                    const float *src = (t1 >= 0 && fi >= 0 && fi < F1) ? s->h1 + ((size_t)(t1 % 3) * F1 + fi) * Cch : NULL;
                    if (perm) {
                        if (src) memcpy(s->col + (size_t)kk * Cch, src, sizeof(float) * Cch); else memset(s->col + (size_t)kk * Cch, 0, sizeof(float) * Cch);
                    } else {
                        if (src) for (int c = 0; c < Cch; c++) s->col[c * 9 + kk] = src[c]; else for (int c = 0; c < Cch; c++) s->col[c * 9 + kk] = 0.f;
                    }
                }
                int v = fi_ * F2 + f;
                if (m->conv2.q) {
                    if (perm) { for (int k = c2; k < c2u; k++) s->col[k] = 0.f; s->sx[v] = km_quant_vec(s->col, c2u, s->xq + (size_t)v * c2u); }
                    else s->sx[v] = km_quant_vec(s->col, c2, s->xq + (size_t)v * c2);
                } else km_gemv_f32(m->conv2.f, Cch, c2, s->col, m->conv2b, s->A + (size_t)v * Cch);
            }
        }
        int nv = nf * F2;
        if (m->conv2.q) { int cc = perm ? c2u : c2; km_gemv_s8_multi(s->conv2p.q, m->conv2.s, m->conv2.bits, Cch, cc, s->xq, s->sx, nv, m->conv2b, s->A, s->xq + (size_t)nv * cc); }
        for (int fi_ = 0; fi_ < nf; fi_++) {
            for (int f = 0; f < F2; f++) for (int c = 0; c < Cch; c++) { float v = s->A[((size_t)fi_ * F2 + f) * Cch + c]; s->flat[(size_t)c * F2 + f] = v > 0.f ? v : 0.f; }
            memcpy(s->Hd + (size_t)(i0 + fi_) * sub_cols, s->flat, sizeof(float) * sub_cols);
        }
    }
    TICK(1);
    /* 3. sub_out: n フレームまとめて (入力は Hd に一時保存した flat。Hd のサイズは下の確保で n×max(vocab, sub_cols) にしてある) */
    for (int i0 = 0; i0 < n; i0 += 8) {                                 /* 8 本ずつ (xq の大きさの都合) */
        int g = n - i0 < 8 ? n - i0 : 8;
        matvec_multi(&m->sub_out, s->Hd + (size_t)i0 * sub_cols, g, m->sub_outb, s->x + (size_t)i0 * d, s->xq, s->sx);
    }
    { float xs = sqrtf((float)d); for (int k = 0; k < n * d; k++) s->x[k] *= xs; }
    TICK(1);
    /* 4. 層 */
    int LC = s->left + C, GC = s->half + C, npos = s->left + 2 * C - 1, ff = m->ff, heads = s->heads, dk = s->dk;
    float scale = 1.f / sqrtf((float)dk);
    long j0 = t0 - s->left; if (j0 < 0) j0 = 0; if (j0 < s->ctx_start) j0 = s->ctx_start; long j1 = t0 + n; int nk = (int)(j1 - j0);
    for (int l = 0; l < s->nl; l++) {
        const km_layer *L = &m->layer[l];
        float *gc = s->gc + (size_t)l * GC * d;
        /* FF1 */
        for (int i = 0; i < n; i++) layernorm(s->x + (size_t)i * d, L->ln_ff1[0], L->ln_ff1[1], s->Xn + (size_t)i * d, d);
        TICK(7);
        matvec_multi(&L->ff1_1, s->Xn, n, L->ff1_1b, s->A, s->xq, s->sx);
        TICK(8);
        km_silu_vec(s->A, n * ff);
        TICK(9);
        matvec_multi(&L->ff1_2, s->A, n, L->ff1_2b, s->B, s->xq, s->sx);
        TICK(10);
        for (int k = 0; k < n * d; k++) s->x[k] += 0.5f * s->B[k];
        TICK(11);
        /* 注意: K/V をリングへ、Q を手元に */
        for (int i = 0; i < n; i++) layernorm(s->x + (size_t)i * d, L->ln_att[0], L->ln_att[1], s->Xn + (size_t)i * d, d);
        TICK(27);
        matvec_multi(&L->k, s->Xn, n, L->kb, s->B, s->xq, s->sx);
        TICK(23);
        for (int i = 0; i < n; i++) {
            long t = t0 + i;
#ifdef KM_ATT_INT8
            pad_heads(s, s->B + (size_t)i * d, s->kf);
            s->ks_[(size_t)l * LC + (size_t)(t % LC)] = km_quant_vec(s->kf, s->ks, s->kc + ((size_t)l * LC + (size_t)(t % LC)) * s->ks);
#elif defined(KM_KV_INT8)
            s->ks_[(size_t)l * LC + (size_t)(t % LC)] = km_quant_vec(s->B + (size_t)i * d, d, s->kc + ((size_t)l * LC + (size_t)(t % LC)) * d);
#else
            memcpy(s->kc + ((size_t)l * LC + (size_t)(t % LC)) * d, s->B + (size_t)i * d, sizeof(float) * d);
#endif
        }
        TICK(24);
        matvec_multi(&L->v, s->Xn, n, L->vb, s->B, s->xq, s->sx);
        TICK(25);
        for (int i = 0; i < n; i++) {
            long t = t0 + i;
#ifdef KM_ATT_INT8
            {   /* V_t を int8 にして内部 RAM の [n][d] に溜める (転置リングへの書き込みは下でまとめて) */
                int p = (int)(t % LC);
                s->vts[(size_t)l * s->LCp + p] = km_quant_vec(s->B + (size_t)i * d, d, s->xq + (size_t)i * d);
            }
#elif defined(KM_KV_INT8)
            s->vs[(size_t)l * LC + (size_t)(t % LC)] = km_quant_vec(s->B + (size_t)i * d, d, s->vc + ((size_t)l * LC + (size_t)(t % LC)) * d);
#else
            memcpy(s->vc + ((size_t)l * LC + (size_t)(t % LC)) * d, s->B + (size_t)i * d, sizeof(float) * d);
#endif
        }
#ifdef KM_ATT_INT8
        {   /* 転置リングへ: 行 e ごとに、このチャンクの n フレームぶん (リングの連続区間、途中で巻き戻ることあり) をまとめて書く */
            int8_t *vt = s->vt + (size_t)l * d * s->LCp; int p0 = (int)(t0 % LC);
            for (int e = 0; e < d; e++) {
                int8_t *row = vt + (size_t)e * s->LCp;
                for (int i = 0; i < n; i++) { int p = p0 + i; if (p >= LC) p -= LC; row[p] = s->xq[(size_t)i * d + e]; }
            }
        }
#endif
        TICK(26);
        matvec_multi(&L->q, s->Xn, n, L->qb, s->Q, s->xq, s->sx);
        TICK(12);
#ifdef KM_ATT_INT8
        for (int i = 0; i < n; i++) {
            const float *q = s->Q + (size_t)i * d;
            for (int h = 0; h < heads; h++) for (int e = 0; e < dk; e++) { s->kf[h * dk + e] = q[h * dk + e] + L->bias_u[h * dk + e]; s->vf[h * dk + e] = q[h * dk + e] + L->bias_v[h * dk + e]; }
            pad_heads(s, s->kf, s->A); s->squ[i] = km_quant_vec(s->A, s->ks, s->qu8 + (size_t)i * s->ks);
            pad_heads(s, s->vf, s->A); s->sqv[i] = km_quant_vec(s->A, s->ks, s->qv8 + (size_t)i * s->ks);
        }
#endif
        TICK(13);
        /* スコア: S[(h*n + i)*nk + (j-j0)]。鍵 j の行を 1 回読んで全問いに、P の行 r を 1 回読んで該当する (i, j) 対に */
        for (int k = 0; k < heads * n * nk; k++) s->S[k] = 0.f;
        {
            att_job aj = { s, L, l, t0, n, j0, j1, nk, s->left + C - 1, npos, 0 };
            long rmin = aj.base - ((t0 + n - 1) - j0), rmax = aj.base - (t0 - (j1 - 1));
            if (rmin < 0) rmin = 0;
            if (rmax > npos - 1) rmax = npos - 1;
            aj.rmin = rmin;
            km_run(att_scores_keys, &aj, nk, 32);                      /* 鍵で分割 (S の列が別なので衝突しない) */
            TICK(14);
            km_run(att_scores_pos, &aj, (int)(rmax - rmin + 1), 32);   /* 相対位置 r で分割: 対 (i, j) は r ごとに一意 → これも衝突しない */
            TICK(15);
        }
        /* softmax (問いごと・ヘッドごと) */
        for (int h = 0; h < heads; h++) for (int i = 0; i < n; i++) {
            float *row = s->S + ((size_t)h * n + i) * nk; float mx = -1e30f;
            for (int j = 0; j < nk; j++) { row[j] *= scale; if (row[j] > mx) mx = row[j]; }
            float sum = 0.f; for (int j = 0; j < nk; j++) { row[j] = km_expneg(row[j] - mx); sum += row[j]; }
            float inv = 1.f / sum; for (int j = 0; j < nk; j++) row[j] *= inv;
        }
        TICK(16);
        /* 値: V の行 j を 1 回読んで全問いへ */
        for (int k = 0; k < n * d; k++) s->O[k] = 0.f;
        {
            att_job aj = { s, L, l, t0, n, j0, j1, nk, s->left + C - 1, npos, 0 };
            km_run(att_values_heads, &aj, heads, 2);                   /* ヘッドで分割 (O の列が別) */
        }
        TICK(17);
        matvec_multi(&L->o, s->O, n, L->ob, s->B, s->xq, s->sx);
        for (int k = 0; k < n * d; k++) s->x[k] += s->B[k];
        TICK(18);
        /* conv: pw1 → GLU → リングへ → 深さ方向 conv → SiLU → pw2 */
        for (int i = 0; i < n; i++) layernorm(s->x + (size_t)i * d, L->ln_conv[0], L->ln_conv[1], s->Xn + (size_t)i * d, d);
        matvec_multi(&L->pw1, s->Xn, n, L->pw1b, s->A, s->xq, s->sx);         /* n × 2d */
        TICK(19);
        /* GLU の窓 W (内部 RAM、A の pw1 出力の後ろ): 行 0..half-1 = 前チャンクの末尾 (リングから)、half..half+n-1 = 今回、以降 half 行は 0 (未来) */
        float *W = s->A + (size_t)n * 2 * d;
        for (int k = 0; k < s->half; k++) {
            long j = t0 - s->half + k;
            if (j < s->ctx_start) memset(W + (size_t)k * d, 0, sizeof(float) * d);
            else memcpy(W + (size_t)k * d, gc + (size_t)(j % GC) * d, sizeof(float) * d);
        }
        memset(W + (size_t)(s->half + n) * d, 0, sizeof(float) * s->half * d);
        for (int i = 0; i < n; i++) {
            float *g = W + (size_t)(s->half + i) * d; const float *a = s->A + (size_t)i * 2 * d;
            km_glu_vec(a, a + d, g, d);
        }
        TICK(20);
        for (int k = 0; k < s->K; k++) for (int c = 0; c < d; c++) s->dwT[(size_t)k * d + c] = L->dw_w[(size_t)c * s->K + k];   /* (d,K) → (K,d) */
        for (int i = 0; i < n; i++) {                                   /* out[i][c] = b[c] + Σ_k w[k][c] · W[i + k][c]  (W の行 i+k ↔ フレーム t0+i+k-half) */
            float *out = s->Xn + (size_t)i * d; const float *Wi = W + (size_t)i * d;
            int c = 0;
            for (; c + 8 <= d; c += 8) {                                /* 8 チャネル同時: 累積 8 本をレジスタに、w と W は k ごとに 32 byte 連続 */
                float a0 = L->dw_b[c], a1 = L->dw_b[c+1], a2 = L->dw_b[c+2], a3 = L->dw_b[c+3], a4 = L->dw_b[c+4], a5 = L->dw_b[c+5], a6 = L->dw_b[c+6], a7 = L->dw_b[c+7];
                const float *w = s->dwT + c, *x = Wi + c;
                for (int k = 0; k < s->K; k++, w += d, x += d) {
                    a0 += w[0] * x[0]; a1 += w[1] * x[1]; a2 += w[2] * x[2]; a3 += w[3] * x[3];
                    a4 += w[4] * x[4]; a5 += w[5] * x[5]; a6 += w[6] * x[6]; a7 += w[7] * x[7];
                }
                out[c] = a0; out[c+1] = a1; out[c+2] = a2; out[c+3] = a3; out[c+4] = a4; out[c+5] = a5; out[c+6] = a6; out[c+7] = a7;
            }
            for (; c < d; c++) {
                float acc = L->dw_b[c]; const float *w = s->dwT + c, *x = Wi + c;
                for (int k = 0; k < s->K; k++) acc += w[(size_t)k * d] * x[(size_t)k * d];
                out[c] = acc;
            }
            km_silu_vec(out, d);
        }
        for (int k = 0; k < s->half && k < n; k++) {                    /* 今回の末尾 half 行をリングへ (次チャンクの左文脈) */
            long j = t0 + n - s->half + k; if (j < t0) continue;
            memcpy(gc + (size_t)(j % GC) * d, W + (size_t)(s->half + (j - t0)) * d, sizeof(float) * d);
        }
        if (n < s->half) {                                               /* 短い最終チャンク: 前チャンク由来の行も含めて末尾 half 行を戻す */
            for (int k = 0; k < s->half; k++) { long j = t0 + n - s->half + k; if (j < 0) continue; memcpy(gc + (size_t)(j % GC) * d, W + (size_t)(s->half + (j - t0)) * d, sizeof(float) * d); }
        }
        TICK(21);
        matvec_multi(&L->pw2, s->Xn, n, L->pw2b, s->B, s->xq, s->sx);
        for (int k = 0; k < n * d; k++) s->x[k] += s->B[k];
        TICK(22);
        /* FF2 + LN out */
        for (int i = 0; i < n; i++) layernorm(s->x + (size_t)i * d, L->ln_ff2[0], L->ln_ff2[1], s->Xn + (size_t)i * d, d);
        matvec_multi(&L->ff2_1, s->Xn, n, L->ff2_1b, s->A, s->xq, s->sx);
        km_silu_vec(s->A, n * ff);
        matvec_multi(&L->ff2_2, s->A, n, L->ff2_2b, s->B, s->xq, s->sx);
        for (int i = 0; i < n; i++) {
            float *xt = s->x + (size_t)i * d; const float *bt = s->B + (size_t)i * d;
            for (int k = 0; k < d; k++) xt[k] += 0.5f * bt[k];
            layernorm(xt, L->ln_out[0], L->ln_out[1], s->Xn + (size_t)i * d, d); memcpy(xt, s->Xn + (size_t)i * d, sizeof(float) * d);
        }
        TICK(2);
    }
    /* 5. head (n フレームまとめて) + greedy */
    int blank = m->vocab - 1;
    matvec_multi(&m->head, s->x, n, m->headb, s->Hd, s->xq, s->sx);
    for (int i = 0; i < n; i++) {
        const float *lg = s->Hd + (size_t)i * m->vocab;
        if (s->tap) s->tap(s->tap_ctx, (int)(t0 + i), lg, m->vocab);
        int best = 0; for (int k = 1; k < m->vocab; k++) if (lg[k] > lg[best]) best = k;
        if (best != blank && best != s->prev && s->n_ids < s->cap_ids) {
            int fr = (int)(t0 + i);
            if (s->dedup_prefix && s->n_ids > 0 && fr - s->id_frame[s->n_ids - 1] <= 2 && piece_is_prefix(m, s->ids[s->n_ids - 1], best))
                s->n_ids--;                                              /* 「タ」の直後に「タイムズ」: 前のを捨てる */
            s->ids[s->n_ids] = best; s->id_frame[s->n_ids] = fr; s->n_ids++;
        }
        s->prev = best;
    }
    TICK(6);
    s->frames_done = t0 + n;
}

/* チャンク c の最後のメル行 4(c+1)C-1 に要るサンプル数 */
static long need_samples(const km_stream *s, long t0) { return (4 * (t0 + s->C) - 1) * (long)s->m->hop + s->m->win / 2; }

int km_stream_feed(km_stream *s, const float *pcm, int n) {
    int done = 0; int i = 0;
    while (i < n) {
        /* リングを溢れさせない: 次のチャンクが要る最古サンプルから pcm_cap 以内しか先に進めない */
        long oldest = (4 * s->frames_done - 3) * (long)s->m->hop - s->m->win / 2 - 1; if (oldest < 0) oldest = 0;
        long room = oldest + s->pcm_cap - s->n_total;
        if (room <= 0) {                                                 /* 処理できるはず (need ≤ n_total) */
            if (s->n_total >= need_samples(s, s->frames_done)) { if (s->silence_db != 0.f && chunk_dbfs(s, s->frames_done, s->C) < s->silence_db) skip_chunk(s, s->frames_done, s->C); else process(s, s->frames_done, s->C); done++; continue; }
            return done;                                                 /* 起きないはず */
        }
        int take = (int)(n - i < room ? n - i : room);
        for (int k = 0; k < take; k++) s->pcm[(s->n_total + k) % s->pcm_cap] = pcm[i + k];
        s->n_total += take; i += take;
        while (s->n_total >= need_samples(s, s->frames_done)) {
            if (s->silence_db != 0.f && chunk_dbfs(s, s->frames_done, s->C) < s->silence_db) skip_chunk(s, s->frames_done, s->C);
            else process(s, s->frames_done, s->C);
            done++;
        }
    }
    return done;
}

void km_stream_finish(km_stream *s) {
    if (s->finished) return;
    s->finished = 1;
    s->T_total = 1 + s->n_total / s->m->hop;
    s->T1_total = (s->T_total - 1) / 2 + 1;
    long Tout = (s->T1_total - 1) / 2 + 1;
    while (s->frames_done < Tout) {
        int n = (int)(Tout - s->frames_done < s->C ? Tout - s->frames_done : s->C);
        if (s->silence_db != 0.f && chunk_dbfs(s, s->frames_done, n) < s->silence_db) skip_chunk(s, s->frames_done, n);
        else process(s, s->frames_done, n);
    }
}
