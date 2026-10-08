/* エンジン内部で共有する小道具 (km_conformer.c で定義)。 */
#ifndef KOMIMI_INTERNAL_H
#define KOMIMI_INTERNAL_H
#include "km.h"
void matvec(const km_mat *w, const float *x, const float *bias, float *out, int8_t *qbuf);
/* n 本まとめて: X は n×cols (行優先 float)、Out は n×rows。xq は n×cols + cols byte 以上 (16 byte 境界)、sx は n 個。 */
extern long long km_gemv_us, km_gemv_calls;      /* 計測 (km_clock_us が設定されているとき) */
extern long long (*km_clock_us_ref)(void);
void matvec_multi(const km_mat *w, const float *X, int n, const float *bias, float *Out, int8_t *xq, float *sx);
#include "km_kernels.h"
static inline float silu(float x) { return km_silu(x); }
static inline float sigmoid(float x) { return km_sigmoid(x); }
void layernorm(const float *x, const float *g, const float *b, float *out, int d);
void rel_pos(float *pe, int T, int d);          /* 行 r ↔ 相対位置 T-1-r (2T-1 行) */
#endif
