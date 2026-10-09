/* komimi: Conformer-CTC の小さな推論エンジン (独自実装)。ホスト C / ESP32-S3 / ESP32-P4 共通の中核。
 *
 * 層の構成は NeMo ConformerEncoder (公開仕様): striding ×4 → ×sqrt(d) → [FF/2, rel-pos MHSA, conv(k=31), FF/2, LN] ×N → CTC。
 * 重みは .kmm (komimi/kmm.py)。行列はすべて行ごと対称 int8 (scale は行ごと)、活性はベクトルごとに動的 int8 化して
 * int32 で積和 → float に戻す。それ以外 (LayerNorm, softmax, SiLU, GLU, 深さ方向 conv, 残差) は float。
 * 内積カーネルは km_kernels.h の 1 関数に集約してあり、S3 (Xtensa PIE) / P4 (RISC-V) はそこだけ差し替える。 */
#ifndef KOMIMI_H
#define KOMIMI_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KM_MAX_LAYERS 24
#define KM_MAX_TOKENS 4096

typedef struct { const int8_t *q; const float *s; const float *f; int rows, cols, bits; } km_mat;  /* int8/int4 (q,s,bits) か float (f) */

typedef struct {
    km_mat ff1_1, ff1_2, ff2_1, ff2_2, q, k, v, o, pos, pw1, pw2;
    const float *ff1_1b, *ff1_2b, *ff2_1b, *ff2_2b, *qb, *kb, *vb, *ob, *pw1b, *pw2b;
    const float *bias_u, *bias_v;                       /* (heads, dk) */
    const float *ln_ff1[2], *ln_att[2], *ln_conv[2], *ln_ff2[2], *ln_out[2];   /* weight, bias */
    const float *dw_w, *dw_b;                           /* (d, k), (d) */
} km_layer;

typedef struct {
    int d, heads, ff, kernel, n_layers, n_mel, vocab, sub_ch, flags, chunk, left, sr, hop, win, nfft, bits;
    const uint8_t *base; size_t size;
    km_mat conv1, conv2, sub_out, head;
    const float *conv1b, *conv2b, *sub_outb, *headb;
    const float *window, *fb, *norm_mean, *norm_std;    /* fb: (n_mel, nfft/2+1) */
    km_layer layer[KM_MAX_LAYERS];
    int n_tokens; const char *tok[KM_MAX_TOKENS]; uint16_t tok_len[KM_MAX_TOKENS];
} km_model;

/* メモリ上の .kmm を解釈する (コピーしない。フラッシュの mmap 先でもよい)。0 で成功。 */
int km_load(km_model *m, const uint8_t *buf, size_t size);

/* 全文脈推論: 16 kHz float PCM (n サンプル) → CTC の greedy 結果 (token id、blank/重複は除いた列)。
 * 戻り値は token 数 (最大 max_out)、負ならエラー。n_frames_out に出力フレーム数 T' (= 入力フレーム / 4) を書く (NULL 可)。 */
int km_recognize(const km_model *m, const float *pcm, int n, int *ids, int max_out, int *n_frames_out);
/* km_recognize と同じ計算で、各フレームの CTC logits (softmax 前、vocab 個、blank = vocab-1) も logits_out に書く
 * (先頭 max_frames フレームまで。T' は 1 + (n/hop)/4 程度なので n/640 + 2 フレームあれば足りる)。候補の強制採点 (CTC forward) 用。 */
int km_recognize_ex(const km_model *m, const float *pcm, int n, int *ids, int max_out, int *n_frames_out, float *logits_out, int max_frames);

/* 行列の重み (int8/int4 の本体と scale) を、大きい順に budget bytes まで alloc で確保した RAM に写して差し替える。
 * フラッシュ直読みの帯域が律速の端末向け (PSRAM に写す)。写した合計 bytes を返す。 */
size_t km_model_cache_weights(km_model *m, void *(*alloc)(size_t), size_t budget);

/* token id 列 → UTF-8 ("▁" は空白に)。戻り値は書いたバイト数。 */
int km_detok(const km_model *m, const int *ids, int n, char *out, int cap);

/* 作業メモリの見積り (bytes)。T' は出力フレーム数 (= 入力フレーム / 4)。 */
size_t km_workspace_bytes(const km_model *m, int t_out);

#ifdef __cplusplus
}
#endif
#endif
