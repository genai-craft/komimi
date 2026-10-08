// StickS3 の LCD (ST7789 135x240)。Aunvox クライアントの lcd.c (Apache-2.0、同作者) を元に、
// 転送を帯に分けて内部 DMA バッファを小さくし (komimi の作業領域に内部 RAM を回す)、カタカナ 16px を足した。
#include "lcd.h"
#include <string.h>
#include "driver/spi_master.h"
#include "driver/ledc.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "font8x8.h"
#include "font_kana16.h"

static const char *TAG = "lcd";
#define PIN_MOSI 39
#define PIN_SCLK 40
#define PIN_DC   45
#define PIN_CS   41
#define PIN_RST  21
#define PIN_BL   38
#define PANEL_W 135
#define PANEL_H 240
#define GAP_X 52
#define GAP_Y 40
#define BAND 16                                   /* 一度に送るパネル行数 (135×16×2 = 4.3 KB の内部 DMA バッファ) */
#define NBAND_BUF 6                               /* 転送はキュー深さ 4 の DMA (非同期)。送り終わる前に上書きしないよう 6 本をリングで使う (depth+2) */

static esp_lcd_panel_handle_t s_panel;
static uint16_t *s_fb;                            /* 論理 240x135 (PSRAM) */
static uint16_t *s_band[NBAND_BUF];               /* 回転済みの帯 (内部 DMA)、リング */

esp_err_t lcd_init(void) {
    spi_bus_config_t bus = { .mosi_io_num = PIN_MOSI, .miso_io_num = -1, .sclk_io_num = PIN_SCLK, .quadwp_io_num = -1, .quadhd_io_num = -1,
                             .max_transfer_sz = PANEL_W * BAND * 2 + 16 };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(SPI3_HOST, &bus, SPI_DMA_CH_AUTO), TAG, "spi bus");
    esp_lcd_panel_io_handle_t io;
    esp_lcd_panel_io_spi_config_t io_cfg = { .dc_gpio_num = PIN_DC, .cs_gpio_num = PIN_CS, .pclk_hz = 40 * 1000 * 1000,
                                             .lcd_cmd_bits = 8, .lcd_param_bits = 8, .spi_mode = 0, .trans_queue_depth = 4 };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)SPI3_HOST, &io_cfg, &io), TAG, "panel io");
    esp_lcd_panel_dev_config_t dev = { .reset_gpio_num = PIN_RST, .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB, .bits_per_pixel = 16 };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_st7789(io, &dev, &s_panel), TAG, "st7789");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_reset(s_panel), TAG, "reset");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_init(s_panel), TAG, "init");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_invert_color(s_panel, true), TAG, "invert");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_set_gap(s_panel, GAP_X, GAP_Y), TAG, "gap");
    s_fb = heap_caps_calloc(LCD_W * LCD_H, 2, MALLOC_CAP_SPIRAM);
    for (int i = 0; i < NBAND_BUF; i++) { s_band[i] = heap_caps_malloc(PANEL_W * BAND * 2, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL); ESP_RETURN_ON_FALSE(s_band[i], ESP_ERR_NO_MEM, TAG, "band alloc"); }
    ESP_RETURN_ON_FALSE(s_fb, ESP_ERR_NO_MEM, TAG, "fb alloc");
    lcd_flush();
    ESP_RETURN_ON_ERROR(esp_lcd_panel_disp_on_off(s_panel, true), TAG, "disp on");
    ledc_timer_config_t tcfg = { .speed_mode = LEDC_LOW_SPEED_MODE, .duty_resolution = LEDC_TIMER_8_BIT, .timer_num = LEDC_TIMER_0,
                                 .freq_hz = 44100, .clk_cfg = LEDC_AUTO_CLK };    /* 可聴域の PWM はマイクに乗る → 44.1 kHz */
    ESP_RETURN_ON_ERROR(ledc_timer_config(&tcfg), TAG, "ledc timer");
    ledc_channel_config_t ccfg = { .gpio_num = PIN_BL, .speed_mode = LEDC_LOW_SPEED_MODE, .channel = LEDC_CHANNEL_0, .timer_sel = LEDC_TIMER_0, .duty = 0, .hpoint = 0 };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&ccfg), TAG, "ledc ch");
    lcd_brightness(100);
    ESP_LOGI(TAG, "LCD ready (%dx%d)", LCD_W, LCD_H);
    return ESP_OK;
}

void lcd_brightness(uint8_t percent) {
    if (percent > 100) percent = 100;
    uint32_t duty = (percent == 100) ? 256 : 255 * percent / 100;    /* 100% は完全 DC (PWM ノイズ無し) */
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, duty); ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
}

void lcd_clear(uint16_t color) { if (s_fb) for (int i = 0; i < LCD_W * LCD_H; i++) s_fb[i] = color; }

void lcd_fill_rect(int x, int y, int w, int h, uint16_t color) {
    if (!s_fb) return;
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > LCD_W) w = LCD_W - x;
    if (y + h > LCD_H) h = LCD_H - y;
    for (int j = 0; j < h; j++) { uint16_t *row = s_fb + (y + j) * LCD_W + x; for (int i = 0; i < w; i++) row[i] = color; }
}

static void put_px(int x, int y, uint16_t c) { if (x >= 0 && x < LCD_W && y >= 0 && y < LCD_H) s_fb[y * LCD_W + x] = c; }

void lcd_text(int x, int y, int scale, uint16_t fg, const char *str) {
    if (!s_fb || scale < 1) return;
    for (; *str; str++) {
        unsigned c = (unsigned char)*str; if (c < 0x20 || c > 0x7F) c = '?';
        const uint8_t *g = font8x8[c - 0x20];
        for (int row = 0; row < 8; row++) { uint8_t bits = g[row]; if (!bits) continue;
            for (int col = 0; col < 8; col++) if (bits >> col & 1)
                for (int sy = 0; sy < scale; sy++) for (int sx = 0; sx < scale; sx++) put_px(x + col * scale + sx, y + row * scale + sy, fg); }
        x += 8 * scale; if (x >= LCD_W) break;
    }
}

/* UTF-8 を 1 文字デコード。返り値は進めたバイト数 */
int lcd_utf8_next(const char *s, uint32_t *cp) {
    unsigned c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return c ? 1 : 0; }
    if ((c & 0xE0) == 0xC0) { *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((c & 0xF0) == 0xE0) { *cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3; }
    *cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4;
}

/* 1 文字 (コードポイント) を 16px で描く。カタカナは kana16、ASCII は 8x8 を 2 倍。戻り値は進めた幅 */
int lcd_glyph16(int x, int y, uint16_t fg, uint32_t cp) {
    const uint16_t *g = kana16_lookup(cp);
    if (g) {
        for (int row = 0; row < 16; row++) { uint16_t bits = g[row]; if (!bits) continue;
            for (int col = 0; col < 16; col++) if (bits >> (15 - col) & 1) put_px(x + col, y + row, fg); }
        return 16;
    }
    char a[2] = { (cp >= 0x20 && cp < 0x7F) ? (char)cp : '?', 0 };
    lcd_text(x, y, 2, fg, a);
    return 16;
}

void lcd_flush(void) {
    if (!s_panel || !s_fb) return;
    /* 論理 (lx,ly) → パネル (px,py) = (ly, LCD_W-1-lx)。パネル行 py を BAND 行ずつ回転して送る */
    int bi = 0;
    for (int py0 = 0; py0 < PANEL_H; py0 += BAND, bi = (bi + 1) % NBAND_BUF) {
        uint16_t *band = s_band[bi];                               /* draw_bitmap は非同期 (DMA)。この帯のバッファは次の 5 帯のあいだ触らない */
        for (int k = 0; k < BAND; k++) {
            int py = py0 + k; int lx = LCD_W - 1 - py;            /* この帯の行 py は論理列 lx */
            uint16_t *dst = band + k * PANEL_W;
            for (int px = 0; px < PANEL_W; px++) dst[px] = s_fb[px * LCD_W + lx];   /* px = ly */
        }
        esp_lcd_panel_draw_bitmap(s_panel, 0, py0, PANEL_W, py0 + BAND, band);
    }
}
