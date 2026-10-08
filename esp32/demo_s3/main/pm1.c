#include "pm1.h"
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"

#define PM1_ADDR 0x6E
#define REG_PWR_CFG 0x06     // bit0: CHG_EN, bit2: LDO_EN, bit3: BOOST_EN (外部 5V)
#define REG_GPIO_MODE 0x10   // bit n: 1=出力
#define REG_GPIO_OUT  0x11   // bit n: 出力レベル
#define REG_GPIO_DRIVE 0x13  // bit n: 0=プッシュプル 1=オープンドレイン
#define REG_GPIO_FUNC0 0x16  // GPIO0-3 の機能 (2bit ずつ, 00=GPIO)

static const char *TAG = "pm1";

// バスが一時的に詰まることがある (リセット直後など) ので軽くリトライする
#define RETRIES 3

static esp_err_t rd(uint8_t reg, uint8_t *val)
{
    esp_err_t err = ESP_FAIL;
    for (int i = 0; i < RETRIES; i++) {
        err = i2c_master_write_read_device(I2C_NUM_0, PM1_ADDR, &reg, 1, val, 1, pdMS_TO_TICKS(100));
        if (err == ESP_OK) return ESP_OK;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    return err;
}

static esp_err_t wr(uint8_t reg, uint8_t val)
{
    esp_err_t err = ESP_FAIL;
    for (int i = 0; i < RETRIES; i++) {
        uint8_t buf[2] = {reg, val};
        err = i2c_master_write_to_device(I2C_NUM_0, PM1_ADDR, buf, 2, pdMS_TO_TICKS(100));
        if (err == ESP_OK) return ESP_OK;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    return err;
}

static esp_err_t gpio_out(int pin, bool on)
{
    uint8_t v;
    esp_err_t err = rd(REG_GPIO_FUNC0, &v);
    if (err != ESP_OK) return err;
    v &= ~(0x03 << (pin * 2));               // 00 = 普通の GPIO
    if ((err = wr(REG_GPIO_FUNC0, v)) != ESP_OK) return err;
    if ((err = rd(REG_GPIO_MODE, &v)) != ESP_OK) return err;
    if ((err = wr(REG_GPIO_MODE, v | (1 << pin))) != ESP_OK) return err;
    // プッシュプルにしないと High が出せない (オープンドレイン + 基板の 100k プルダウンで
    // AW8737 の SHDN が上がらずアンプが鳴らない。M5Unified も 0x13 を明示クリアしている)
    if ((err = rd(REG_GPIO_DRIVE, &v)) != ESP_OK) return err;
    if ((err = wr(REG_GPIO_DRIVE, v & ~(1 << pin))) != ESP_OK) return err;
    if ((err = rd(REG_GPIO_OUT, &v)) != ESP_OK) return err;
    return wr(REG_GPIO_OUT, on ? (v | (1 << pin)) : (v & ~(1 << pin)));
}

esp_err_t pm1_init(void)
{
    uint8_t v = 0;
    esp_err_t err = rd(REG_GPIO_MODE, &v);
    if (err == ESP_OK) ESP_LOGI(TAG, "M5PM1 応答あり (GPIO_MODE=0x%02x)", v);
    return err;
}

esp_err_t pm1_ext_output(bool on)
{
    uint8_t v;
    esp_err_t err = rd(REG_PWR_CFG, &v);
    if (err != ESP_OK) return err;
    return wr(REG_PWR_CFG, on ? (v | (1 << 3)) : (v & ~(1 << 3)));
}

esp_err_t pm1_speaker_enable(bool on)
{
    esp_err_t err = gpio_out(2, true);       // PYG2: 周辺電源 (MIC/SPK/BL) は常に on
    if (err != ESP_OK) return err;
    return gpio_out(3, on);                  // PYG3: アンプ
}

esp_err_t pm1_charge_enable(bool on)
{
    uint8_t v;
    esp_err_t err = rd(REG_PWR_CFG, &v);
    if (err != ESP_OK) return err;
    ESP_LOGI(TAG, "PWR_CFG 0x%02x -> charge %s", v, on ? "on" : "off");
    return wr(REG_PWR_CFG, on ? (v | 1) : (v & ~1));
}
