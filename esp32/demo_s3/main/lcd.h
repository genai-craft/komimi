// StickS3 内蔵 LCD (ST7789, 1.14" 135x240) の最小描画層。
// ピン/オフセットは M5GFX の board_M5StickS3 自動検出コードから写した:
//   SPI3: MOSI=39 SCLK=40 DC=45 CS=41 RST=21, BL=GPIO38 (PWM),
//   135x240 offset(52,40) 色反転あり。バックライト電源は M5PM1 PYG2 (audio 側で on 済み)。
// 論理座標は横向き 240x135 (lcd.c 内で 90 度回転して転送する)。
#pragma once
#include <stdint.h>
#include "esp_err.h"

#define LCD_W 240   // 論理 (横向き)
#define LCD_H 135

// RGB565。SPI へはビッグエンディアンで流すのでここでスワップしておく
#define LCD_RGB(r, g, b) ((uint16_t)__builtin_bswap16( \
    (((r) & 0xF8) << 8) | (((g) & 0xFC) << 3) | ((b) >> 3)))

#define LCD_BLACK  LCD_RGB(0, 0, 0)
#define LCD_WHITE  LCD_RGB(255, 255, 255)
#define LCD_GRAY   LCD_RGB(128, 128, 128)
#define LCD_DGRAY  LCD_RGB(48, 48, 48)
#define LCD_RED    LCD_RGB(230, 60, 50)
#define LCD_GREEN  LCD_RGB(40, 200, 90)
#define LCD_BLUE   LCD_RGB(40, 120, 230)
#define LCD_YELLOW LCD_RGB(240, 200, 40)
#define LCD_CYAN   LCD_RGB(60, 200, 220)
#define LCD_ORANGE LCD_RGB(240, 140, 40)

esp_err_t lcd_init(void);                       // I2C (PM1) 初期化後に呼ぶ
void lcd_clear(uint16_t color);
void lcd_fill_rect(int x, int y, int w, int h, uint16_t color);
// 8x8 フォント x scale。ASCII のみ。はみ出しはクリップ
void lcd_text(int x, int y, int scale, uint16_t fg, const char *str);
void lcd_flush(void);                           // フレームバッファをパネルへ転送
void lcd_brightness(uint8_t percent);           // 0-100

// 追加 (komimi デモ): UTF-8 とカタカナ 16px
int lcd_utf8_next(const char *s, uint32_t *cp);
int lcd_glyph16(int x, int y, uint16_t fg, uint32_t cp);
