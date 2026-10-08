/* komimi デモ (M5StickS3): マイク → komimi (端末内で日本語カナ認識) → LCD に文字起こしを流す。
 *
 *   capture task (core 0, 高優先)  I2S 16 kHz → PSRAM リング (20 s)
 *   main task    (core 0)          リングから 100 ms ずつ → km_stream_feed → 新しい token を LCD へ
 *   worker       (core 1)          km_par (gemv の行分割・注意の分割)
 *
 * ヘッダ: 入力レベル (dBFS バー、無音ゲートのしきい値マーク)、RTF、溜まっている音の長さ。
 * ボタン A (正面, GPIO11): 表示をクリア。ボタン B (側面, GPIO12): 充電の on/off (USB 給電中は充電器のスイッチングがキーキー鳴くので既定 off)。
 * ボタンは別タスクで 10 ms ごとに読む (メインループはチャンク計算で数百 ms 止まるので短押しを落とす)。 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/ringbuf.h"
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "audio.h"
#include "lcd.h"
#include "pm1.h"
#include "km.h"
#include "km_stream.h"
#include "km_kernels.h"

static const char *TAG = "demo";
#define PIN_BTN_A GPIO_NUM_11
#define PIN_BTN_B GPIO_NUM_12
#define RING_SECONDS 20
#define GATE_DB (-58.0f)                             /* 初期値。起動後は床 (静かなときのレベル) + GATE_MARGIN に適応 */
#define GATE_MARGIN 6.0f
#define LAG_CAP_S 2.5f                                /* 溜まった音がこれを超えたら古い音を捨てて追いつく (遅延を累積させない) */
#define COLS 15                                   /* 16px × 15 = 240 */
#define ROWS 6                                    /* ヘッダ 26px + 18px × 6 = 134 */
#define MAX_CHARS (COLS * ROWS * 2)

/* ---- 2 コア ---- */
static SemaphoreHandle_t s_go, s_done; static km_range_fn s_fn; static void *s_ctx; static int s_lo, s_hi;
static void worker(void *arg) { for (;;) { xSemaphoreTake(s_go, portMAX_DELAY); s_fn(s_ctx, s_lo, s_hi); xSemaphoreGive(s_done); } }
static void par2(km_range_fn fn, void *ctx, int n) { int mid = n / 2; s_fn = fn; s_ctx = ctx; s_lo = mid; s_hi = n; xSemaphoreGive(s_go); fn(ctx, 0, mid); xSemaphoreTake(s_done, portMAX_DELAY); }
static void *psram_alloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }

/* ---- 前処理: 80 Hz ハイパス (2 次、@16 kHz。StickS3 のマイクの低域ゴロゴロ・DC を落とす。Aunvox クライアントと同じ考え) とクリップ計数 ---- */
static float s_hp[4]; static volatile int s_clips = 0;
static inline int16_t hpf_sample(int16_t in) {
    /* butter(2, 80/8000, 'high'): b = [0.97803, -1.95606, 0.97803], a = [1, -1.95558, 0.95654] */
    const float b0 = 0.97803048f, b1 = -1.95606096f, b2 = 0.97803048f, a1 = -1.95557824f, a2 = 0.95654368f;
    float x = in, y = b0 * x + b1 * s_hp[0] + b2 * s_hp[1] - a1 * s_hp[2] - a2 * s_hp[3];
    s_hp[1] = s_hp[0]; s_hp[0] = x; s_hp[3] = s_hp[2]; s_hp[2] = y;
    if (y > 32767.f) y = 32767.f; if (y < -32768.f) y = -32768.f;
    return (int16_t)y;
}

/* ---- 音声キュー ---- */
static RingbufHandle_t s_ring;
static void capture_task(void *arg) {
    int16_t *buf = heap_caps_malloc(320 * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    for (;;) {
        size_t n = audio_read(buf, 320);                                /* 20 ms */
        if (!n) continue;
        for (size_t i = 0; i < n; i++) { if (buf[i] >= 32000 || buf[i] <= -32000) s_clips++; buf[i] = hpf_sample(buf[i]); }
        while (xRingbufferSend(s_ring, buf, n * sizeof(int16_t), 0) == pdFALSE) {   /* 満杯なら古い方を捨てる */
            size_t got = 0; uint8_t *old = xRingbufferReceiveUpTo(s_ring, &got, 0, n * sizeof(int16_t));
            if (!old) { vTaskDelay(1); continue; }
            vRingbufferReturnItem(s_ring, old);
        }
    }
}

static volatile int s_btn_a = 0, s_btn_b = 0;      /* 押下イベント (button_task が立て、メインが消費) */

/* ---- ボタン (10 ms ポーリング、30 ms デバウンス、押した瞬間だけイベント) ---- */
static void button_task(void *arg) {
    int pa = 1, pb = 1, ca = 0, cb = 0;
    for (;;) {
        int a = gpio_get_level(PIN_BTN_A), b = gpio_get_level(PIN_BTN_B);
        ca = (a == 0) ? ca + 1 : 0; cb = (b == 0) ? cb + 1 : 0;
        if (ca == 3 && pa) { s_btn_a = 1; pa = 0; } if (a) pa = 1;
        if (cb == 3 && pb) { s_btn_b = 1; pb = 0; } if (b) pb = 1;
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* ---- 表示 ---- */
static uint32_t s_text[MAX_CHARS]; static int s_ntext;                   /* 0x0A = 改行 */
static float s_level_db = -90.f, s_rtf = 0.f, s_lag = 0.f, s_floor_db = -60.f, s_gate_db = GATE_DB; static int s_gate_on = 1, s_charge_on = 0, s_dropped = 0;
static const km_model *s_model;

static int64_t s_draw_us = 0; static int s_draws = 0;
static void draw_(void);
static void draw(void) { int64_t t0 = esp_timer_get_time(); draw_(); s_draw_us += esp_timer_get_time() - t0; s_draws++; }
static void draw_(void) {
    lcd_clear(LCD_BLACK);
    lcd_fill_rect(0, 0, LCD_W, 24, LCD_DGRAY);
    lcd_text(4, 4, 2, LCD_CYAN, "komimi");
    /* レベルメータ (-80..0 dBFS を 70px に) としきい値の印 */
    int bx = 112, bw = 70; lcd_fill_rect(bx, 8, bw, 8, LCD_BLACK);
    int lv = (int)((s_level_db + 80.f) / 80.f * bw); if (lv < 0) lv = 0; if (lv > bw) lv = bw;
    lcd_fill_rect(bx, 8, lv, 8, s_level_db > s_gate_db ? LCD_GREEN : LCD_GRAY);
    if (s_gate_on) { int gx = bx + (int)((s_gate_db + 80.f) / 80.f * bw); if (gx < bx) gx = bx; if (gx > bx + bw) gx = bx + bw; lcd_fill_rect(gx, 6, 1, 12, LCD_YELLOW); }
    char st[32]; snprintf(st, sizeof st, "x%.2f %3.1fs%s", s_rtf, s_lag, s_dropped ? "!" : ""); lcd_text(LCD_W - 8 * (int)strlen(st) - 3, 8, 1, s_rtf > 1.f ? LCD_ORANGE : LCD_WHITE, st);
    if (s_charge_on) { lcd_fill_rect(80, 6, 28, 12, LCD_ORANGE); lcd_text(82, 8, 1, LCD_BLACK, "CHG"); }
    /* 本文: 折り返して最後の ROWS 行 */
    static int lines[ROWS * 4][2]; int nl = 0, start = 0;             /* [行][開始, 長さ] */
    for (int i = 0; i <= s_ntext && nl < ROWS * 4; i++) {
        int brk = (i == s_ntext) || s_text[i] == 0x0A || (i - start) >= COLS;
        if (brk) { lines[nl][0] = start; lines[nl][1] = i - start; nl++; start = (i < s_ntext && s_text[i] == 0x0A) ? i + 1 : i; if (i == s_ntext) break; }
    }
    int first = nl > ROWS ? nl - ROWS : 0;
    for (int r = first; r < nl; r++) {
        int y = 26 + (r - first) * 18, x = 0;
        uint16_t fg = (r == nl - 1) ? LCD_WHITE : LCD_GRAY;
        for (int k = 0; k < lines[r][1]; k++) x += lcd_glyph16(x, y, fg, s_text[lines[r][0] + k]);
    }
    if (s_ntext == 0) { lcd_text(16, 60, 2, LCD_GRAY, "say something"); lcd_text(16, 84, 1, LCD_GRAY, "A: clear   B: charge on/off"); }
    lcd_flush();
}

static int nid_total(km_stream *s) { const int *ids; return km_stream_tokens(s, &ids); }

static void append_utf8(const char *s) {
    while (*s) {
        uint32_t cp; int n = lcd_utf8_next(s, &cp); if (!n) break; s += n;
        if (cp == ' ') continue;
        if (s_ntext >= MAX_CHARS) {                                     /* 先頭の 1 行ぶん捨てる */
            memmove(s_text, s_text + COLS, sizeof(uint32_t) * (MAX_CHARS - COLS)); s_ntext -= COLS;
        }
        s_text[s_ntext++] = cp;
    }
}

void app_main(void) {
    gpio_config_t btn = { .pin_bit_mask = (1ULL << PIN_BTN_A) | (1ULL << PIN_BTN_B), .mode = GPIO_MODE_INPUT, .pull_up_en = GPIO_PULLUP_ENABLE };
    gpio_config(&btn);
    bool audio_ok = audio_init() == ESP_OK;                            /* I2C + PM1 (LCD バックライト電源) もここで */
    lcd_init();
    lcd_clear(LCD_BLACK); lcd_text(40, 40, 3, LCD_CYAN, "komimi"); lcd_text(40, 76, 1, LCD_GRAY, "loading model..."); lcd_flush();
    if (!audio_ok) { lcd_text(16, 100, 1, LCD_RED, "mic init failed"); lcd_flush(); }

    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const void *map = NULL; esp_partition_mmap_handle_t h;
    if (!part || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map, &h) != ESP_OK) { lcd_text(16, 100, 1, LCD_RED, "no model partition"); lcd_flush(); return; }
    static km_model m; int rc = km_load(&m, (const uint8_t *)map, part->size);
    if (rc) { char e[40]; snprintf(e, sizeof e, "km_load failed %d", rc); lcd_text(16, 100, 1, LCD_RED, e); lcd_flush(); return; }
    s_model = &m;
    s_go = xSemaphoreCreateBinary(); s_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker, "km_w1", 16384, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    km_par = par2;
    int chunk = m.chunk > 0 ? m.chunk : 16, left = m.left > 0 ? m.left : 128;
    km_stream *s = km_stream_new(&m, chunk, left);
    if (!s) { lcd_text(16, 100, 1, LCD_RED, "stream init failed"); lcd_flush(); return; }
    km_stream_set_silence_gate(s, GATE_DB);
    km_clock_us = esp_timer_get_time;
    /* 音声リング (PSRAM) を先に確保してから、残りで重みをキャッシュ */
    StaticRingbuffer_t *rbs = heap_caps_calloc(1, sizeof(StaticRingbuffer_t), MALLOC_CAP_SPIRAM);
    uint8_t *rbm = heap_caps_malloc(AUDIO_SAMPLE_RATE * RING_SECONDS * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    s_ring = xRingbufferCreateStatic(AUDIO_SAMPLE_RATE * RING_SECONDS * sizeof(int16_t), RINGBUF_TYPE_BYTEBUF, rbm, rbs);
    size_t free_ps = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t cached = free_ps > 512 * 1024 ? km_model_cache_weights(&m, psram_alloc, free_ps - 384 * 1024) : 0;
    ESP_LOGI(TAG, "model: %d layers, sub_ch %d, chunk %d; cached %.2f MB; workspace %u (psram fallback %u)", m.n_layers, m.sub_ch, chunk, cached / 1e6,
             (unsigned)km_stream_bytes(s), (unsigned)km_stream_hot_psram(s));
    if (audio_ok) xTaskCreatePinnedToCore(capture_task, "cap", 4096, NULL, 10, NULL, 0);
    xTaskCreate(button_task, "btn", 2048, NULL, 5, NULL);
    draw();

    float *fbuf = heap_caps_malloc(1600 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int consumed = 0; int64_t last_draw = 0, last_token_us = 0; double acc_audio = 0, acc_comp = 0;
    int prev_sk = 0, prev_frames = 0, silent_run = 0, pending_break = 0;
    for (;;) {
        size_t got = 0;
        int16_t *pcm = xRingbufferReceiveUpTo(s_ring, &got, pdMS_TO_TICKS(50), 1600 * sizeof(int16_t));
        if (pcm) {
            int n = got / sizeof(int16_t); double e = 0;
            for (int i = 0; i < n; i++) { float v = pcm[i] / 32768.0f; fbuf[i] = v; e += (double)v * v; }
            vRingbufferReturnItem(s_ring, pcm);
            float db = n ? (float)(10.0 * log10(e / n + 1e-12)) : -90.f;
            s_level_db = s_level_db * 0.6f + db * 0.4f;
            /* 床の追従: 下がるときは速く、上がるときはごくゆっくり (喋っている間に床が上がらないように) */
            if (db < s_floor_db) s_floor_db = s_floor_db * 0.7f + db * 0.3f;            /* 下がるときは速く */
            else if (db < s_floor_db + 10.f) s_floor_db += 0.05f * (db - s_floor_db);  /* 周囲音 (床の近く) には追従 */
            else s_floor_db += 0.002f * (db - s_floor_db);                               /* 声には追従しない */
            float g = s_floor_db + GATE_MARGIN; if (g < -70.f) g = -70.f; if (g > -30.f) g = -30.f;
            if (fabsf(g - s_gate_db) > 1.0f) { s_gate_db = g; if (s_gate_on) km_stream_set_silence_gate(s, s_gate_db); }
            int64_t t0 = esp_timer_get_time();
            km_stream_feed(s, fbuf, n);
            double dt = (esp_timer_get_time() - t0) / 1e6;
            acc_audio = acc_audio * 0.9 + (n / (double)AUDIO_SAMPLE_RATE) * 0.1; acc_comp = acc_comp * 0.9 + dt * 0.1;
            s_rtf = acc_audio > 0 ? (float)(acc_comp / acc_audio) : 0.f;
            UBaseType_t waiting = 0; vRingbufferGetInfo(s_ring, NULL, NULL, NULL, NULL, &waiting);
            s_lag = waiting / (float)(AUDIO_SAMPLE_RATE * sizeof(int16_t));
            if (s_lag > LAG_CAP_S) {                                     /* 追いつけない: 古い音を 1 秒ぶん残して捨て、文脈もそこから */
                size_t keep = AUDIO_SAMPLE_RATE * sizeof(int16_t);
                while (waiting > keep) { size_t got2 = 0; uint8_t *old = xRingbufferReceiveUpTo(s_ring, &got2, 0, waiting - keep); if (!old) break; vRingbufferReturnItem(s_ring, old); waiting -= got2; }
                s_dropped++; pending_break = 1;
            }
            /* 段落の区切り: (a) 無音ゲートで 2 チャンク以上続けて飛ばした、(b) token が 1.5 秒以上出ていない (ゲートが効かない騒がしい場所用) */
            int sk = km_stream_skipped(s), fr = km_stream_frames(s);
            if (sk > prev_sk) silent_run += sk - prev_sk; else if (fr > prev_frames) silent_run = 0;
            prev_sk = sk; prev_frames = fr;
            if (silent_run >= 2 && s_ntext > 0 && s_text[s_ntext - 1] != 0x0A) pending_break = 1;
            const int *ids; int nid = km_stream_tokens(s, &ids);
            if (nid > consumed) {
                int64_t now = esp_timer_get_time();
                if (last_token_us && now - last_token_us > 1500000 && s_ntext > 0 && s_text[s_ntext - 1] != 0x0A) pending_break = 1;
                last_token_us = now;
                if (pending_break) { if (s_ntext < MAX_CHARS) s_text[s_ntext++] = 0x0A; pending_break = 0; }
                char txt[256]; km_detok(&m, ids + consumed, nid - consumed, txt, sizeof txt); append_utf8(txt);
                consumed = nid; if (nid > 2048) { km_stream_reset_tokens(s); consumed = 0; }
                draw(); last_draw = esp_timer_get_time();
            }
        }
        { static int64_t last_log = 0; if (esp_timer_get_time() - last_log > 5000000) { last_log = esp_timer_get_time();
            const int64_t *pf = km_stream_profile(s); int fr = km_stream_frames(s);
            ESP_LOGI(TAG, "rtf %.2f lag %.1fs level %.0f floor %.0f gate %.0f dB skipped %d dropped %d clips %d | draw %lld ms (%d) | per chunk: mel %.0f sub %.0f ff %.0f att %.0f conv %.0f ms",
                     s_rtf, s_lag, s_level_db, s_floor_db, s_gate_db, km_stream_skipped(s), s_dropped, s_clips, (long long)(s_draw_us / 1000), s_draws,
                     fr ? pf[0] / 1e3 / fr * chunk : 0.0, fr ? pf[1] / 1e3 / fr * chunk : 0.0, fr ? (pf[2] + pf[7] + pf[8] + pf[9] + pf[10] + pf[11]) / 1e3 / fr * chunk : 0.0,
                     fr ? (pf[3] + pf[4] + pf[12] + pf[13] + pf[14] + pf[15] + pf[16] + pf[17] + pf[18] + pf[23] + pf[24] + pf[25] + pf[26] + pf[27]) / 1e3 / fr * chunk : 0.0,
                     fr ? (pf[5] + pf[19] + pf[20] + pf[21] + pf[22]) / 1e3 / fr * chunk : 0.0); s_draw_us = 0; s_draws = 0; } }
        if (s_btn_a) { s_btn_a = 0; s_ntext = 0; consumed = nid_total(s); draw(); last_draw = esp_timer_get_time(); }
        if (s_btn_b) { s_btn_b = 0; s_charge_on = !s_charge_on; pm1_charge_enable(s_charge_on); draw(); last_draw = esp_timer_get_time(); }
        if (esp_timer_get_time() - last_draw > 1000000) { draw(); last_draw = esp_timer_get_time(); }   /* 描画も CPU を食うので token が無ければ 1 秒ごと */
    }
}
