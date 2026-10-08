/* 音響特徴: 16 kHz PCM → log-mel (n_mel × T)。NeMo FilterbankFeatures と同じ定義:
 * 前後に win/2 の零詰め → pre-emphasis 0.97 → hann (対称) 窓 → 実 FFT (nfft) → |X|² → mel → log(x + 2^-24) → 特徴ごとの標準化。
 * フレーム数 T = 1 + n / hop。 */
#include "km_feat.h"
#include <math.h>
#include <string.h>

/* 自前の反復 radix-2 複素 FFT (nfft は 2 のべき)。実入力なので虚部 0 から始める。 */
void km_fft_inplace(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {                 /* ビット反転 */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.f * (float)M_PI / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.f, ci = 0.f;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
                float ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
            }
        }
    }
}

int km_num_frames(const km_model *m, int n) { return 1 + n / m->hop; }

/* 1 フレーム (中心 c = t*hop) の log-mel を out[n_mel] に。pcm は零詰め前の信号 (範囲外は 0、pre-emphasis は零詰め後に掛ける)。 */
static void frame_logmel(const km_model *m, const float *pcm, int n, int t, float *out, float *re, float *im, float *pw) {
    int half = m->win / 2, nfft = m->nfft, nb = nfft / 2 + 1;
    int start = t * m->hop - half;                        /* 窓の先頭サンプル (負なら零詰め領域) */
    for (int i = 0; i < nfft; i++) { re[i] = 0.f; im[i] = 0.f; }
    for (int i = 0; i < m->win; i++) {
        int s = start + i;
        float x = (s >= 0 && s < n) ? pcm[s] : 0.f;
        float xp = (s - 1 >= 0 && s - 1 < n) ? pcm[s - 1] : 0.f;
        float y = (s >= -half) ? x - 0.97f * xp : 0.f;    /* 零詰め領域の先頭は 0 - 0.97*0 = 0 */
        re[i] = y * m->window[i];
    }
    km_fft_inplace(re, im, nfft);
    for (int i = 0; i < nb; i++) pw[i] = re[i] * re[i] + im[i] * im[i];
    for (int k = 0; k < m->n_mel; k++) {
        const float *f = m->fb + (size_t)k * nb; float acc = 0.f;
        for (int i = 0; i < nb; i++) acc += f[i] * pw[i];
        out[k] = logf(acc + 5.9604644775390625e-08f);    /* 2^-24 */
    }
}

/* feat: (T, n_mel) 行優先。work は nfft*3 float 以上。 */
void km_logmel(const km_model *m, const float *pcm, int n, float *feat, float *work) {
    int T = km_num_frames(m, n), nm = m->n_mel;
    float *re = work, *im = work + m->nfft, *pw = work + 2 * m->nfft;
    for (int t = 0; t < T; t++) frame_logmel(m, pcm, n, t, feat + (size_t)t * nm, re, im, pw);
    if (m->flags & 1) {
        for (int t = 0; t < T; t++) for (int k = 0; k < nm; k++)
            feat[(size_t)t * nm + k] = (feat[(size_t)t * nm + k] - m->norm_mean[k]) / (m->norm_std[k] + 1e-5f);
    } else {                                                /* 発話内で特徴ごとに平均 0・不偏標準偏差 1 */
        for (int k = 0; k < nm; k++) {
            double s = 0, ss = 0;
            for (int t = 0; t < T; t++) s += feat[(size_t)t * nm + k];
            double mean = s / T;
            for (int t = 0; t < T; t++) { double d = feat[(size_t)t * nm + k] - mean; ss += d * d; }
            float sd = T > 1 ? (float)sqrt(ss / (T - 1)) : 0.f;
            for (int t = 0; t < T; t++) feat[(size_t)t * nm + k] = (float)((feat[(size_t)t * nm + k] - mean) / (sd + 1e-5f));
        }
    }
}
