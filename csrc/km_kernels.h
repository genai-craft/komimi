/* 内積カーネル。ここだけ機種ごとに差し替える (KM_KERNEL_GENERIC / KM_KERNEL_ESP32S3 / KM_KERNEL_ESP32P4 / KM_KERNEL_AVX2)。
 *
 * km_gemv_s8: out[r] = s_x * s_w[r] * Σ_c q_w[r,c] * q_x[c] + bias[r]   (r < rows, c < cols; cols は 16 の倍数が前提)
 * km_quant_vec: float ベクトル → 対称 int8 (scale = max|x|/127)。 */
#ifndef KOMIMI_KERNELS_H
#define KOMIMI_KERNELS_H
#include <math.h>
#include <stdint.h>

/* 範囲 [0, n) を分担する並列実行の差し込み口。km_par が NULL なら直列。
 * 端末 (2 コア) では main.c が自分の半分を実行しつつもう半分を別コアのタスクに投げる実装を入れる。 */
typedef void (*km_range_fn)(void *ctx, int lo, int hi);
extern void (*km_par)(km_range_fn fn, void *ctx, int n);
static inline void km_run(km_range_fn fn, void *ctx, int n, int min_parallel) {
    if (km_par && n >= min_parallel) km_par(fn, ctx, n); else fn(ctx, 0, n);
}

/* 速い expf: x = k·ln2 + r (|r| ≤ ln2/2) に分けて e^r を 6 次多項式、2^k は指数ビットに足す。相対誤差 ~2e-7。
 * 端末の libm の expf は ~1 µs かかり、SiLU/GLU/softmax でチャンクあたり 60 万回呼ぶので効く。 */
static inline float km_expf(float x) {
    if (x > 88.0f) x = 88.0f;
    if (x < -87.0f) return 0.f;
    float kf = x * 1.44269504088896341f;                      /* x / ln2 */
    int k = (int)(kf + (kf >= 0.f ? 0.5f : -0.5f));
    float r = x - (float)k * 0.693145751953125f - (float)k * 1.428606765330187e-06f;   /* ln2 を 2 つに分けて精度を保つ */
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f; p = p * r + 1.0f / 24.0f; p = p * r + 1.0f / 6.0f; p = p * r + 0.5f; p = p * r + 1.0f; p = p * r + 1.0f;
    union { float f; int32_t i; } u; u.f = p; u.i += k << 23;
    return u.f;
}

/* sigmoid の表引き (4096 点、[-12, 12]、線形補間; 最大誤差 ~4e-7)。端末では expf が高く、SiLU/GLU で 1 チャンク数十万回呼ぶ。 */
#define KM_SIG_N 1024                                   /* 4 KB: L1 に収まる大きさ (4096 だと活性のストリームと取り合って遅かった) */
#define KM_SIG_LO (-12.0f)
#define KM_SIG_HI (12.0f)
static inline const float *km_sigmoid_table(void) {
    static float tab[KM_SIG_N + 1]; static int init = 0;
    if (!init) { for (int i = 0; i <= KM_SIG_N; i++) { float x = KM_SIG_LO + (KM_SIG_HI - KM_SIG_LO) * i / KM_SIG_N; tab[i] = 1.0f / (1.0f + expf(-x)); } init = 1; }
    return tab;
}
static inline float km_sigmoid(float x) {
    const float *tab = km_sigmoid_table();
    if (x <= KM_SIG_LO) return tab[0];
    if (x >= KM_SIG_HI) return tab[KM_SIG_N];
    float f = (x - KM_SIG_LO) * (KM_SIG_N / (KM_SIG_HI - KM_SIG_LO)); int i = (int)f; float t = f - (float)i;
    return tab[i] + (tab[i + 1] - tab[i]) * t;
}
static inline float km_silu(float x) { return x * km_sigmoid(x); }

/* 配列版: 4 本の独立な連鎖を交互に進めてインオーダー core の FP レイテンシを隠す (1 要素ずつだと 40 cycle 近くかかる) */
static inline void km_silu_vec(float *x, int n) {
    const float *tab = km_sigmoid_table(); const float k = KM_SIG_N / (KM_SIG_HI - KM_SIG_LO);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float x0 = x[i], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
        float c0 = x0 < KM_SIG_LO ? KM_SIG_LO : (x0 > KM_SIG_HI ? KM_SIG_HI : x0);
        float c1 = x1 < KM_SIG_LO ? KM_SIG_LO : (x1 > KM_SIG_HI ? KM_SIG_HI : x1);
        float c2 = x2 < KM_SIG_LO ? KM_SIG_LO : (x2 > KM_SIG_HI ? KM_SIG_HI : x2);
        float c3 = x3 < KM_SIG_LO ? KM_SIG_LO : (x3 > KM_SIG_HI ? KM_SIG_HI : x3);
        float f0 = (c0 - KM_SIG_LO) * k, f1 = (c1 - KM_SIG_LO) * k, f2 = (c2 - KM_SIG_LO) * k, f3 = (c3 - KM_SIG_LO) * k;
        int i0 = (int)f0, i1 = (int)f1, i2 = (int)f2, i3 = (int)f3;
        if (i0 >= KM_SIG_N) i0 = KM_SIG_N - 1;
        if (i1 >= KM_SIG_N) i1 = KM_SIG_N - 1;
        if (i2 >= KM_SIG_N) i2 = KM_SIG_N - 1;
        if (i3 >= KM_SIG_N) i3 = KM_SIG_N - 1;
        float t0 = f0 - (float)i0, t1 = f1 - (float)i1, t2 = f2 - (float)i2, t3 = f3 - (float)i3;
        float a0 = tab[i0], a1 = tab[i1], a2 = tab[i2], a3 = tab[i3];
        float b0 = tab[i0 + 1], b1 = tab[i1 + 1], b2 = tab[i2 + 1], b3 = tab[i3 + 1];
        x[i] = x0 * (a0 + (b0 - a0) * t0); x[i + 1] = x1 * (a1 + (b1 - a1) * t1); x[i + 2] = x2 * (a2 + (b2 - a2) * t2); x[i + 3] = x3 * (a3 + (b3 - a3) * t3);
    }
    for (; i < n; i++) x[i] = km_silu(x[i]);
}
/* GLU: g[k] = a[k] · sigmoid(b[k])、4 本交互 */
static inline void km_glu_vec(const float *a, const float *b, float *g, int n) {
    const float *tab = km_sigmoid_table(); const float k = KM_SIG_N / (KM_SIG_HI - KM_SIG_LO);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float x0 = b[i], x1 = b[i + 1], x2 = b[i + 2], x3 = b[i + 3];
        float c0 = x0 < KM_SIG_LO ? KM_SIG_LO : (x0 > KM_SIG_HI ? KM_SIG_HI : x0);
        float c1 = x1 < KM_SIG_LO ? KM_SIG_LO : (x1 > KM_SIG_HI ? KM_SIG_HI : x1);
        float c2 = x2 < KM_SIG_LO ? KM_SIG_LO : (x2 > KM_SIG_HI ? KM_SIG_HI : x2);
        float c3 = x3 < KM_SIG_LO ? KM_SIG_LO : (x3 > KM_SIG_HI ? KM_SIG_HI : x3);
        float f0 = (c0 - KM_SIG_LO) * k, f1 = (c1 - KM_SIG_LO) * k, f2 = (c2 - KM_SIG_LO) * k, f3 = (c3 - KM_SIG_LO) * k;
        int i0 = (int)f0, i1 = (int)f1, i2 = (int)f2, i3 = (int)f3;
        if (i0 >= KM_SIG_N) i0 = KM_SIG_N - 1;
        if (i1 >= KM_SIG_N) i1 = KM_SIG_N - 1;
        if (i2 >= KM_SIG_N) i2 = KM_SIG_N - 1;
        if (i3 >= KM_SIG_N) i3 = KM_SIG_N - 1;
        float t0 = f0 - (float)i0, t1 = f1 - (float)i1, t2 = f2 - (float)i2, t3 = f3 - (float)i3;
        float p0 = tab[i0], p1 = tab[i1], p2 = tab[i2], p3 = tab[i3];
        float q0 = tab[i0 + 1], q1 = tab[i1 + 1], q2 = tab[i2 + 1], q3 = tab[i3 + 1];
        g[i] = a[i] * (p0 + (q0 - p0) * t0); g[i + 1] = a[i + 1] * (p1 + (q1 - p1) * t1); g[i + 2] = a[i + 2] * (p2 + (q2 - p2) * t2); g[i + 3] = a[i + 3] * (p3 + (q3 - p3) * t3);
    }
    for (; i < n; i++) g[i] = a[i] * km_sigmoid(b[i]);
}

/* exp(x) (x ≤ 0) の表引き: softmax 用。[-16, 0] を 4096 点、線形補間。x < -16 は 0。 */
#define KM_EXP_N 1024
static inline const float *km_expneg_table(void) {
    static float tab[KM_EXP_N + 1]; static int init = 0;
    if (!init) { for (int i = 0; i <= KM_EXP_N; i++) tab[i] = expf(-16.0f + 16.0f * i / KM_EXP_N); init = 1; }
    return tab;
}
static inline float km_expneg(float x) {
    if (x <= -16.0f) return 0.f;
    if (x >= 0.f) return 1.f;
    const float *tab = km_expneg_table();
    float f = (x + 16.0f) * (KM_EXP_N / 16.0f); int i = (int)f; float t = f - (float)i;
    return tab[i] + (tab[i + 1] - tab[i]) * t;
}

/* 16 byte 境界に揃えた確保 (PIE のベクタロード用)。free は km_aligned_free で。 */
#include <stdlib.h>
static inline void *km_aligned_alloc(size_t n) {
    uint8_t *p = (uint8_t *)malloc(n + 32); if (!p) return NULL;
    uint8_t *q = (uint8_t *)(((uintptr_t)p + 16 + 15) & ~(uintptr_t)15); q[-1] = (uint8_t)(q - p); return q;
}
static inline void km_aligned_free(void *q) { if (q) free((uint8_t *)q - ((uint8_t *)q)[-1]); }

/* 最近接丸め (RISC-V/x86 は fcvt 1 命令、分岐なし)。|v| ≤ 127 が前提 */
static inline int km_round(float v) {
#if defined(__riscv) || defined(__x86_64__) || defined(__aarch64__)
    return (int)__builtin_lrintf(v);
#else
    return v >= 0.f ? (int)(v + 0.5f) : -(int)(-v + 0.5f);
#endif
}
static inline float km_quant_vec(const float *x, int n, int8_t *q) {
    float m = 0.f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > m) m = a; }
    float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
    for (int i = 0; i < n; i++) q[i] = (int8_t)km_round(x[i] * inv);
    return s;
}

#if defined(KM_KERNEL_AVX2)
#include <immintrin.h>
static inline int32_t km_dot_s8(const int8_t *a, const int8_t *b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        /* 16bit に広げて madd (int8×int8 → int16 ペア和 → int32) */
        __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va)), a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb)), b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));
    }
    int32_t tmp[8]; _mm256_storeu_si256((__m256i *)tmp, acc);
    int32_t s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; i++) s += (int32_t)a[i] * b[i];
    return s;
}
#elif defined(KM_KERNEL_ESP32S3)
/* ESP32-S3 の PIE (Processor Instruction Extensions): 16 lane の int8 積和を 40 bit の ACCX に溜める。
 * a, b は 16 byte 境界、n は 16 の倍数が前提 (外れたら末尾をスカラで)。 */
static inline int32_t km_dot_s8_scalar(const int8_t *a, const int8_t *b, int n) {
    int32_t s = 0; for (int i = 0; i < n; i++) s += (int32_t)a[i] * b[i]; return s;
}
/* noinline: 大きなループに inline されると xtensa gcc がレジスタ割付で ICE を起こす (asm の制約が複雑になるため) */
int32_t km_dot_s8_pie(const int8_t *pa, const int8_t *pb, int n16);    /* km_kernels_esp.c (asm は別 TU: xtensa gcc の ICE 回避) */
int32_t km_dot_s8_pie2(const int8_t *pa, const int8_t *pb, int n16);
int32_t km_dot_s8_pie3(const int8_t *pa, const int8_t *pb);           /* 48 要素固定 (注意スコア) */
extern int km_dot_variant;                                           /* 0 = 単純版, 1 = 融合ロード版 (起動時の自己診断で決める) */
static inline int32_t km_dot_s8(const int8_t *a, const int8_t *b, int n) {
    if ((((uintptr_t)a | (uintptr_t)b) & 15) || n < 16) return km_dot_s8_scalar(a, b, n);
    int n16 = n >> 4; int32_t acc;
    if (n16 == 3) acc = km_dot_s8_pie3(a, b);
    else if (km_dot_variant == 1) acc = km_dot_s8_pie2(a, b, n16);
    else acc = km_dot_s8_pie(a, b, n16);
    int rem = n & 15;
    if (rem) acc += km_dot_s8_scalar(a + (n - rem), b + (n - rem), rem);
    return acc;
}
#elif defined(KM_KERNEL_ESP32P4)
/* ESP32-P4 (RISC-V) の SIMD 拡張: S3 の PIE と同じ構成の命令が esp. 接頭辞で使える (esp-dsp の P4 実装と同じ流儀)。
 * ハードウェアループは使わず普通の分岐で回す。a, b は 16 byte 境界、n16 = n/16。 */
static inline int32_t km_dot_s8_scalar(const int8_t *a, const int8_t *b, int n) {
    int32_t s = 0; for (int i = 0; i < n; i++) s += (int32_t)a[i] * b[i]; return s;
}
int32_t km_dot_s8_p4(const int8_t *pa, const int8_t *pb, int n16);
int32_t km_dot_s8_p4_2(const int8_t *pa, const int8_t *pb, int n16);
int32_t km_dot_s8_p4_3(const int8_t *pa, const int8_t *pb);
extern int km_dot_variant;
static inline int32_t km_dot_s8(const int8_t *a, const int8_t *b, int n) {
    if ((((uintptr_t)a | (uintptr_t)b) & 15) || n < 16) return km_dot_s8_scalar(a, b, n);
    int n16 = n >> 4; int32_t acc;
    if (n16 == 3) acc = km_dot_s8_p4_3(a, b);
    else if (km_dot_variant == 1) acc = km_dot_s8_p4_2(a, b, n16);
    else acc = km_dot_s8_p4(a, b, n16);
    int rem = n & 15;
    if (rem) acc += km_dot_s8_scalar(a + (n - rem), b + (n - rem), rem);
    return acc;
}
#else
static inline int32_t km_dot_s8(const int8_t *a, const int8_t *b, int n) {
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) { s0 += (int32_t)a[i] * b[i]; s1 += (int32_t)a[i+1] * b[i+1]; s2 += (int32_t)a[i+2] * b[i+2]; s3 += (int32_t)a[i+3] * b[i+3]; }
    for (; i < n; i++) s0 += (int32_t)a[i] * b[i];
    return s0 + s1 + s2 + s3;
}
#endif

/* rows × cols の int8 行列 (行優先) と int8 ベクトルの積。 */
static inline void km_gemv_s8(const int8_t *w, const float *sw, int rows, int cols, const int8_t *x, float sx,
                              const float *bias, float *out) {
    for (int r = 0; r < rows; r++) {
        int32_t acc = km_dot_s8(w + (size_t)r * cols, x, cols);
        out[r] = (float)acc * sw[r] * sx + (bias ? bias[r] : 0.f);
    }
}

/* int4 の展開表: byte → (下位ニブル, 上位ニブル) の int8 対 */
static inline const int8_t *km_nibble_lut(void) {
    static int8_t lut[512]; static int init = 0;
    if (!init) { for (int b = 0; b < 256; b++) { int lo = b & 15, hi = b >> 4; lut[2 * b] = (int8_t)(lo > 7 ? lo - 16 : lo); lut[2 * b + 1] = (int8_t)(hi > 7 ? hi - 16 : hi); } init = 1; }
    return lut;
}
static inline void km_unpack_s4_row(const uint8_t *pr, int cols, int8_t *row) {
    const int8_t *lut = km_nibble_lut();
    for (int c = 0; c < cols / 2; c++) { const int8_t *e = lut + 2 * pr[c]; row[2 * c] = e[0]; row[2 * c + 1] = e[1]; }
}

/* 複数ベクトル版: 重みの行を 1 回だけ読んで (int4 なら 1 回だけ展開して) n 本の入力と内積。
 * xs は n 本の量子化済み入力 (各 cols byte、連続、16 byte 境界)、sx はその scale。out[i*rows + r]。 */
typedef struct { const int8_t *w; const float *sw; int bits, rows, cols; const int8_t *xs; const float *sx; int n; const float *bias; float *out; int8_t *row; } km_gemv_job;
static void km_gemv_rows(void *ctx, int lo, int hi) {
    const km_gemv_job *j = (const km_gemv_job *)ctx;
    int8_t *row = j->row + (lo == 0 ? 0 : j->cols);                 /* 2 分割の後半は別の展開バッファ (row は 2×cols 確保) */
    /* 行外側・ベクトル内側: 重みの行は 1 回だけ読む (PSRAM 上の重みにはこれが肝心。S3 の 32 KB キャッシュでは
     * 行ブロック × ベクトルの順だと重みを読み直して 10 倍遅かった)。入力ベクトルは内部 RAM なので何度読んでもよい */
    for (int r = lo; r < hi; r++) {
        const int8_t *wr;
        if (j->bits == 4) { km_unpack_s4_row((const uint8_t *)j->w + (size_t)r * (j->cols / 2), j->cols, row); wr = row; }
        else wr = j->w + (size_t)r * j->cols;
        float b = j->bias ? j->bias[r] : 0.f, swr = j->sw[r];
        for (int i = 0; i < j->n; i++) j->out[(size_t)i * j->rows + r] = (float)km_dot_s8(wr, j->xs + (size_t)i * j->cols, j->cols) * swr * j->sx[i] + b;
    }
}
/* row は 2×cols byte (並列時に 2 本使う)。 */
static inline void km_gemv_s8_multi(const int8_t *w, const float *sw, int bits, int rows, int cols,
                                    const int8_t *xs, const float *sx, int n, const float *bias, float *out, int8_t *row) {
    km_gemv_job j = { w, sw, bits, rows, cols, xs, sx, n, bias, out, row };
    km_run(km_gemv_rows, &j, rows, 64);
}

/* int4 (1 byte に 2 値、下位 = 偶数列) の行列。行ごとに int8 へ展開してから内積 (展開バッファは cols byte)。 */
static inline void km_gemv_s4(const int8_t *w, const float *sw, int rows, int cols, const int8_t *x, float sx,
                              const float *bias, float *out, int8_t *row) {
    const uint8_t *p = (const uint8_t *)w;
    for (int r = 0; r < rows; r++) {
        km_unpack_s4_row(p + (size_t)r * (cols / 2), cols, row);
        int32_t acc = km_dot_s8(row, x, cols);
        out[r] = (float)acc * sw[r] * sx + (bias ? bias[r] : 0.f);
    }
}

/* float 重み版 (bits=32 の .kmm の検算用)。 */
static inline void km_gemv_f32(const float *w, int rows, int cols, const float *x, const float *bias, float *out) {
    for (int r = 0; r < rows; r++) {
        const float *wr = w + (size_t)r * cols; float acc = 0.f;
        for (int c = 0; c < cols; c++) acc += wr[c] * x[c];
        out[r] = acc + (bias ? bias[r] : 0.f);
    }
}
#endif
