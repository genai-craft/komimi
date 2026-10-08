/* ストリーミング推論 (チャンク幅 chunk 出力フレーム = chunk×40 ms、左文脈 left フレーム)。
 * 学習側 (komimi_train/model.py、非公開) の chunk/left マスク学習と**同じ値**を返す (境界の決め事はこのファイルと km_stream.c 冒頭)。
 * 音を km_stream_feed で足していくと、揃ったチャンクから順に処理され、確定した token が溜まる。 */
#ifndef KOMIMI_STREAM_H
#define KOMIMI_STREAM_H
#include "km.h"

typedef struct km_stream km_stream;

km_stream *km_stream_new(const km_model *m, int chunk, int left);
void km_stream_free(km_stream *s);
void km_stream_reset(km_stream *s);
/* 16 kHz float PCM を追加。処理できたチャンク数を返す。 */
int km_stream_feed(km_stream *s, const float *pcm, int n);
/* 入力の終わり: 残りのフレームを (系列末尾として) 処理する。 */
void km_stream_finish(km_stream *s);
/* 無音ゲート: チャンクの RMS が dbfs (例 -55) を下回ったら計算を飛ばし、左文脈はそのチャンクの後からにする。0 で無効 (既定)。
 * 飛ばした時間で遅れを取り戻せる (呼び出し側が音を溜めておけば、平均で実時間なら足りる)。 */
void km_stream_set_silence_gate(km_stream *s, float dbfs);
int km_stream_skipped(const km_stream *s);
/* 接頭辞重複の除去 (既定 on): 直前の piece が 2 フレーム以内に出た次の piece の接頭辞なら捨てる */
void km_stream_set_dedup_prefix(km_stream *s, int on);
/* 確定済み token を捨てる (表示側が受け取ったあと。長時間動かすとき用) */
void km_stream_reset_tokens(km_stream *s);
/* これまでに確定した token 列 (全体)。 */
int km_stream_tokens(const km_stream *s, const int **ids);
/* 統計: 処理した出力フレーム数、使った作業メモリ (bytes) */
int km_stream_frames(const km_stream *s);
size_t km_stream_bytes(const km_stream *s);
size_t km_stream_hot_psram(const km_stream *s);
int km_stream_conv2_batch(const km_stream *s);     /* conv2 を何フレームまとめているか (内部 RAM 次第で 2 か 1) */   /* 内部 RAM に入らず PSRAM に落ちた作業バッファ (端末) */
/* 計測: km_clock_us を設定すると、段階別の累計時間 (us) を km_stream_profile で読める。
 * 0 mel, 1 subsampling, 2 FF, 3 注意 (q/k/v), 4 注意 (score/値), 5 conv, 6 head */
extern int64_t (*km_clock_us)(void);
const int64_t *km_stream_profile(const km_stream *s);
/* 検算用: 処理した各フレームの logits (vocab 個) を受け取るコールバック */
void km_stream_set_tap(km_stream *s, void (*tap)(void *ctx, int frame, const float *logits, int vocab), void *ctx);
#endif
