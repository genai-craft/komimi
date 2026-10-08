#include "audio.h"
#include "driver/i2c.h"
#include "driver/i2s_std.h"
#include "es8311.h"
#include "esp_check.h"
#include "pm1.h"

static const char *TAG = "audio";
#define PIN_I2C_SDA 47
#define PIN_I2C_SCL 48
#define PIN_I2S_MCLK 18
#define PIN_I2S_BCLK 17
#define PIN_I2S_WS   15
#define PIN_I2S_DOUT 14
#define PIN_I2S_DIN  16
#define MCLK_MULTIPLE 256

static i2s_chan_handle_t s_tx, s_rx;

esp_err_t audio_init(void) {
    i2c_config_t i2c = { .mode = I2C_MODE_MASTER, .sda_io_num = PIN_I2C_SDA, .scl_io_num = PIN_I2C_SCL,
                         .sda_pullup_en = GPIO_PULLUP_ENABLE, .scl_pullup_en = GPIO_PULLUP_ENABLE, .master.clk_speed = 100000 };
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_NUM_0, &i2c), TAG, "i2c cfg");
    ESP_RETURN_ON_ERROR(i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0), TAG, "i2c drv");
    pm1_ext_output(false);                                              /* 外部 5V ブースト off (コイル鳴きの発生源) */
    pm1_charge_enable(false);                                           /* 充電器も off (USB 給電中のキーキー音の疑い。デモ中は充電しない) */
    if (pm1_init() != ESP_OK || pm1_speaker_enable(false) != ESP_OK)
        ESP_LOGE(TAG, "M5PM1 に書けない (周辺電源 MIC/BL が入らないかも)");
    /* I2S 全二重 master (TX は使わないがクロック源として立てる: ES8311 の初期化は MCLK/BCLK が要る) */
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan, &s_tx, &s_rx), TAG, "i2s chan");
    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = { .mclk = PIN_I2S_MCLK, .bclk = PIN_I2S_BCLK, .ws = PIN_I2S_WS, .dout = PIN_I2S_DOUT, .din = PIN_I2S_DIN },
    };
    std.clk_cfg.mclk_multiple = MCLK_MULTIPLE;
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_tx, &std), TAG, "i2s tx");
    std.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;                         /* mono 受信は左スロットのみ (両方だと同じ値が 2 個並ぶ) */
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx, &std), TAG, "i2s rx");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_tx), TAG, "tx en");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_rx), TAG, "rx en");
    es8311_handle_t dev = es8311_create(I2C_NUM_0, ES8311_ADDRRES_0);
    ESP_RETURN_ON_FALSE(dev, ESP_FAIL, TAG, "es8311 create");
    const es8311_clock_config_t clk = { .mclk_inverted = false, .sclk_inverted = false, .mclk_from_mclk_pin = true,
                                        .mclk_frequency = AUDIO_SAMPLE_RATE * MCLK_MULTIPLE, .sample_frequency = AUDIO_SAMPLE_RATE };
    ESP_RETURN_ON_ERROR(es8311_init(dev, &clk, ES8311_RESOLUTION_16, ES8311_RESOLUTION_16), TAG, "es8311 init");
    ESP_RETURN_ON_ERROR(es8311_sample_frequency_config(dev, clk.mclk_frequency, AUDIO_SAMPLE_RATE), TAG, "es8311 fs");   /* 無いと ADC が 1/4 速になる (Aunvox で実証) */
    ESP_RETURN_ON_ERROR(es8311_microphone_config(dev, false), TAG, "es8311 mic");
    /* Aunvox の遠い声向け (アナログ +24 dB + デジタル +8 dB) だと、手元で喋るデモでは声が −7 dBFS まで来てクリップする
     * (クリップ計数で確認、子音の立ち上がりが潰れて「タタイムズ」のような重複の原因)。アナログ 12 dB、デジタル 0 dB (既定 0xC8) に */
    es8311_microphone_gain_set(dev, ES8311_MIC_GAIN_12DB);
    ESP_LOGI(TAG, "mic ready (16 kHz mono)");
    return ESP_OK;
}

size_t audio_read(int16_t *dst, size_t n) {
    size_t got = 0;
    if (i2s_channel_read(s_rx, dst, n * sizeof(int16_t), &got, portMAX_DELAY) != ESP_OK) return 0;
    return got / sizeof(int16_t);
}
