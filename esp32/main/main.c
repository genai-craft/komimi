/* komimi ベンチ (ESP32-S3 / ESP32-P4): "model" パーティションに書いた .kmm をフラッシュ直読み (mmap) し、
 * 合成音 (正弦波 + 雑音) をチャンク送りして 1 チャンク (640 ms) あたりの処理時間と RTF、作業メモリを出す。 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "km.h"
#include "km_stream.h"
#include "km_kernels.h"
#include "km_internal.h"
#include "test_audio.h"
/* ホスト (komimi_cli_att8) の出力。モデル (層数) と chunk で選ぶ */
/* ホスト (csrc/komimi_cli_att8 --stream --left 128) の出力。端末はこれと一字一句一致するはず。試験音声は test_audio.h (Open JTalk で合成) */
#define EXPECTED_16L_C32  "コミミワマイコンデゴクチーサナオンセーニンシキエンジンデスキョーワイーテンキデスネ"   /* v10 (16 層) chunk 32 */
#define EXPECTED_16L_C16  "コミミワマイコンデゴクチーサナオンセーニンシキエンジンデスキョーワイーテンキデスネ"   /* v10 chunk 16 (同じ) */
#define EXPECTED_8L88_C32 "コミミワマイコンデルゴチーサナオンセーニンシケンジンデスキョーワイーテンキデスネ"     /* v10s (8 層 sub_ch 88) chunk 32 */
#define EXPECTED_8L88_C16 "コミミワマイコンデゴチーサナオンセーニンシケンジンデスキョーワイーテンキデスデスネ"   /* v10s chunk 16 */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

/* 2 コア: 範囲の後半を core 1 のワーカーに投げ、前半を自分で回し、合流する */
static SemaphoreHandle_t s_go, s_done; static km_range_fn s_fn; static void *s_ctx; static int s_lo, s_hi;
static void worker(void *arg) { for (;;) { xSemaphoreTake(s_go, portMAX_DELAY); s_fn(s_ctx, s_lo, s_hi); xSemaphoreGive(s_done); } }
static void par2(km_range_fn fn, void *ctx, int n) {
    int mid = n / 2;
    s_fn = fn; s_ctx = ctx; s_lo = mid; s_hi = n;
    xSemaphoreGive(s_go);
    fn(ctx, 0, mid);
    xSemaphoreTake(s_done, portMAX_DELAY);
}
static void noop_job(void *ctx, int lo, int hi) { (void)ctx; (void)lo; (void)hi; }
static void start_parallel(void) {
    s_go = xSemaphoreCreateBinary(); s_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker, "km_w1", 16384, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    km_par = par2;
}

#define SR 16000
#define SECONDS 8

/* 内積カーネルの自己診断: SIMD 版とスカラ版を乱数で突き合わせ、速度も出す */
static void kernel_selftest(void) {
    enum { N = 704, R = 2000 };
    int8_t *a = (int8_t *)km_aligned_alloc(N), *b = (int8_t *)km_aligned_alloc(N);
    unsigned seed = 7; int bad = 0;
    for (int t = 0; t < 50; t++) {
        for (int i = 0; i < N; i++) { seed = seed * 1103515245u + 12345u; a[i] = (int8_t)(seed >> 16); seed = seed * 1103515245u + 12345u; b[i] = (int8_t)(seed >> 16); }
        int32_t ref = 0; for (int i = 0; i < N; i++) ref += (int32_t)a[i] * b[i];
        int32_t got = km_dot_s8(a, b, N);
        if (ref != got) { bad++; if (bad < 4) printf("  dot mismatch: ref %ld got %ld\n", (long)ref, (long)got); }
    }
    int64_t t0 = esp_timer_get_time(); volatile int32_t sink = 0;
    for (int r = 0; r < R; r++) sink += km_dot_s8(a, b, N);
    int64_t dt = esp_timer_get_time() - t0;
    printf("kernel selftest: %s, %.1f MMAC/s (n=%d)\n", bad ? "FAIL" : "PASS", (double)N * R / (double)dt, N);
    /* 候補 2 (融合ロード・2 段展開) を同じ乱数で検算し、正しければ速さで選ぶ */
    {
        int bad2 = 0; int32_t ref = 0;
        km_dot_variant = 1;
        for (int t = 0; t < 50 && !bad2; t++) {
            for (int i = 0; i < N; i++) { seed = seed * 1103515245u + 12345u; a[i] = (int8_t)(seed >> 16); seed = seed * 1103515245u + 12345u; b[i] = (int8_t)(seed >> 16); }
            ref = 0; for (int i = 0; i < N; i++) ref += (int32_t)a[i] * b[i];
            if (km_dot_s8(a, b, N) != ref || km_dot_s8(a, b, 176) != km_dot_s8_scalar(a, b, 176) || km_dot_s8(a, b, 48) != km_dot_s8_scalar(a, b, 48)) bad2 = 1;
        }
        int64_t t1 = esp_timer_get_time(); for (int r = 0; r < R; r++) sink += km_dot_s8(a, b, N); int64_t dt2 = esp_timer_get_time() - t1;
        printf("kernel variant 2 (fused load): %s, %.1f MMAC/s\n", bad2 ? "FAIL" : "PASS", (double)N * R / (double)dt2);
        km_dot_variant = (!bad2 && dt2 < dt) ? 1 : 0;
        printf("  -> using variant %d\n", km_dot_variant);
    }
    /* 短い内積 (d=176) */
    t0 = esp_timer_get_time(); for (int r = 0; r < R * 4; r++) sink += km_dot_s8(a, b, 176); dt = esp_timer_get_time() - t0;
    printf("  dot n=176: %.1f MMAC/s (%.0f ns/call)\n", 176.0 * R * 4 / dt, dt * 1e3 / (R * 4));
    t0 = esp_timer_get_time(); for (int r = 0; r < R * 8; r++) sink += km_dot_s8(a, b, 48); dt = esp_timer_get_time() - t0;
    printf("  dot n=48: %.1f MMAC/s (%.0f ns/call)\n", 48.0 * R * 8 / dt, dt * 1e3 / (R * 8));
    /* int4 の 704×176 を 16 本まとめて (重みは内部 RAM / PSRAM) */
    for (int where = 0; where < 2; where++) {
        size_t nq = 704 * 176 / 2;
        uint8_t *w = (uint8_t *)(where ? heap_caps_aligned_alloc(16, nq, MALLOC_CAP_SPIRAM) : heap_caps_aligned_alloc(16, nq, MALLOC_CAP_INTERNAL));
        float *sw = (float *)malloc(704 * 4), *sx = (float *)malloc(16 * 4), *out = (float *)heap_caps_malloc(16 * 704 * 4, MALLOC_CAP_INTERNAL);
        int8_t *xs = (int8_t *)heap_caps_aligned_alloc(16, 17 * 176, MALLOC_CAP_INTERNAL);
        if (!w || !sw || !sx || !out || !xs) { printf("  (alloc failed for gemv bench %d)\n", where); continue; }
        for (size_t i = 0; i < nq; i++) w[i] = (uint8_t)(i * 31 + 7);
        for (int i = 0; i < 704; i++) sw[i] = 0.01f;
        for (int i = 0; i < 16; i++) sx[i] = 0.02f;
        for (int i = 0; i < 17 * 176; i++) xs[i] = (int8_t)(i * 13);
        t0 = esp_timer_get_time();
        for (int r = 0; r < 20; r++) km_gemv_s8_multi((const int8_t *)w, sw, 4, 704, 176, xs, sx, 16, NULL, out, xs + 16 * 176);
        dt = esp_timer_get_time() - t0;
        printf("  gemv int4 704x176 x16 (%s): %.1f MMAC/s, %.2f ms/call\n", where ? "PSRAM" : "internal", 704.0 * 176 * 16 * 20 / dt, dt / 20e3);
        /* 同じ行列を int8 として (展開なし) */
        t0 = esp_timer_get_time();
        for (int r = 0; r < 20; r++) km_gemv_s8_multi((const int8_t *)w, sw, 8, 352, 176, xs, sx, 16, NULL, out, xs + 16 * 176);
        dt = esp_timer_get_time() - t0;
        printf("  gemv int8 352x176 x16 (%s): %.1f MMAC/s\n", where ? "PSRAM" : "internal", 352.0 * 176 * 16 * 20 / dt);
        heap_caps_free(w); free(sw); free(sx); heap_caps_free(out); heap_caps_free(xs);
    }
    /* int4 展開だけ */
    { int8_t *row = (int8_t *)heap_caps_malloc(1600, MALLOC_CAP_INTERNAL); uint8_t *pk = (uint8_t *)heap_caps_malloc(800, MALLOC_CAP_INTERNAL);
      for (int i = 0; i < 800; i++) pk[i] = (uint8_t)i;
      t0 = esp_timer_get_time(); for (int r = 0; r < 2000; r++) { km_unpack_s4_row(pk, 1584, row); sink += row[r & 1023]; } dt = esp_timer_get_time() - t0;
      printf("  unpack int4: %.1f MB/s of packed bytes (%.1f ns/byte)\n", 792.0 * 2000 / dt, dt * 1e3 / (792.0 * 2000)); heap_caps_free(row); heap_caps_free(pk); }
    /* conv1 相当: 176×9 のスカラ内積 ×1360 */
    { int8_t w9[176 * 9]; for (int i = 0; i < 176 * 9; i++) w9[i] = (int8_t)i; float sw9[176]; for (int i = 0; i < 176; i++) sw9[i] = 0.01f; int8_t x9[16] = {1,2,3,4,5,6,7,8,9}; float o9[176];
      t0 = esp_timer_get_time(); for (int r = 0; r < 1360; r++) km_gemv_s8(w9, sw9, 176, 9, x9, 0.1f, NULL, o9); dt = esp_timer_get_time() - t0;
      printf("  conv1-like 176x9 x1360: %.1f ms\n", dt / 1e3); sink += (int)o9[3]; }
    km_aligned_free(a); km_aligned_free(b);
}

static void *psram_alloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }

void app_main(void) {
    printf("komimi bench: free heap %u, psram %u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_DEFAULT), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    kernel_selftest();
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) { printf("no 'model' partition\n"); return; }
    const void *map = NULL; esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map, &h) != ESP_OK) { printf("mmap failed\n"); return; }
    static km_model m;
    int rc = km_load(&m, (const uint8_t *)map, part->size);
    printf("km_load %d: d=%d layers=%d vocab=%d bits=%d tokens=%d\n", rc, m.d, m.n_layers, m.vocab, m.bits, m.n_tokens);
    if (rc) return;
    km_clock_us = esp_timer_get_time; km_clock_us_ref = (long long (*)(void))esp_timer_get_time;
    start_parallel(); printf("parallel: 2 cores\n");
    {   /* 2 コア同期のコスト: 空ジョブを 1000 回 */
        int64_t t0 = esp_timer_get_time(); for (int i = 0; i < 1000; i++) km_par(noop_job, NULL, 2); int64_t dt = esp_timer_get_time() - t0;
        printf("parallel sync cost: %.1f us/call\n", dt / 1000.0);
    }
    int chunk = m.chunk > 0 ? m.chunk : 16, left = m.left > 0 ? m.left : 128;
    printf("stream config: chunk %d (%d ms), left %d\n", chunk, chunk * 40, left);
    km_stream *s = km_stream_new(&m, chunk, left);
    if (!s) { printf("km_stream_new failed (fixed norm?)\n"); return; }
    printf("conv2 batch %d frames; largest free internal block %u\n", km_stream_conv2_batch(s), (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA));
    printf("stream workspace %u bytes (of which %u fell back to PSRAM); free psram %u, free internal %u\n", (unsigned)km_stream_bytes(s), (unsigned)km_stream_hot_psram(s),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    /* 重みをフラッシュから PSRAM へ (残りの PSRAM から 256 KB の余裕を引いた分まで) */
    size_t free_ps = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t cached = free_ps > 300 * 1024 ? km_model_cache_weights(&m, psram_alloc, free_ps - 256 * 1024) : 0;
    printf("weights cached in PSRAM: %.2f MB (free psram now %u)\n", cached / 1e6, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    /* 実音声 (dev の発話、6.1 s) を 100 ms ずつ流す。認識結果をホストの出力と突き合わせる */
    int piece = SR / 10; float *buf = (float *)malloc(sizeof(float) * piece);
    int64_t t_compute = 0; int chunks = 0; double audio_s = (double)TEST_AUDIO_N / SR;
    for (int i = 0; i < TEST_AUDIO_N; i += piece) {
        int n_ = TEST_AUDIO_N - i < piece ? TEST_AUDIO_N - i : piece;
        for (int k = 0; k < n_; k++) buf[k] = test_audio[i + k] / 32768.0f;
        int64_t t0 = esp_timer_get_time(); int done = km_stream_feed(s, buf, n_); t_compute += esp_timer_get_time() - t0; chunks += done;
    }
    int64_t t0 = esp_timer_get_time(); km_stream_finish(s); t_compute += esp_timer_get_time() - t0;
    const int *ids; int n = km_stream_tokens(s, &ids); char text[512]; km_detok(&m, ids, n, text, sizeof text);
    printf("%d chunks, frames %d, compute %.3f s for %.2f s audio: RTF %.3f (%.0f ms / %d ms chunk)\n", chunks, km_stream_frames(s), t_compute / 1e6, audio_s,
           t_compute / 1e6 / audio_s, chunks ? t_compute / 1e3 / chunks : 0.0, chunk * 40);
    const char *expected = m.sub_ch == 88 ? (chunk == 32 ? EXPECTED_8L88_C32 : EXPECTED_8L88_C16)
                         : (chunk == 32 ? EXPECTED_16L_C32 : EXPECTED_16L_C16);   /* 公開モデル以外は参考値 */
    printf("text:     %s\nexpected: %s\nmatch: %s\n", text, expected, strcmp(text, expected) == 0 ? "YES" : "NO (ホストと不一致)");

    const int64_t *pf = km_stream_profile(s); const char *names[7] = {"mel", "subsample", "ff", "att_qkv", "att_score", "conv", "head"};
    for (int i = 0; i < 7; i++) printf("  %-10s %8.1f ms/chunk\n", names[i], chunks ? pf[i] / 1e3 / chunks : 0.0);
    printf("  gemv total %8.1f ms/chunk (%lld calls/chunk, incl. quantization)\n", chunks ? km_gemv_us / 1e3 / chunks : 0.0, chunks ? km_gemv_calls / chunks : 0LL);
    const char *sub[] = {"ff1:LN", "ff1:gemv1", "ff1:silu", "ff1:gemv2", "ff1:resid", "att:LN+KV", "att:Q+quant", "att:scores_k", "att:scores_p", "att:softmax", "att:values", "att:outproj", "conv:LN+pw1", "conv:GLU", "conv:dw+silu", "conv:pw2+res"};
    for (int i = 0; i < 16; i++) printf("    %-14s %8.1f ms/chunk\n", sub[i], chunks ? pf[7 + i] / 1e3 / chunks : 0.0);
    const char *sub2[] = {"att:K gemv", "att:K store", "att:V gemv", "att:V store+T", "att:LN"};
    for (int i = 0; i < 5; i++) printf("    %-14s %8.1f ms/chunk\n", sub2[i], chunks ? pf[23 + i] / 1e3 / chunks : 0.0);
    printf("    (att:LN+KV は上の 5 つの合計、att:Q+quant は Q gemv だけ、att:scores_k に Q の量子化が含まれる)\n");
    km_stream_free(s);
}
