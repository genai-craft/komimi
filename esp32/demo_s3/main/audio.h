// M5StickS3 のマイク入力 (ES8311 + I2S)。16 kHz mono s16。Aunvox クライアントの audio.c から入力側だけ (Apache-2.0、同作者)
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#define AUDIO_SAMPLE_RATE 16000
esp_err_t audio_init(void);
size_t audio_read(int16_t *dst, size_t n);     // n サンプル読む (ブロック)
