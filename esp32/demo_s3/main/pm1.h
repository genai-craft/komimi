// M5PM1 (StickS3 の電源管理チップ, I2C 0x6E) の最小操作。
// スピーカーアンプ (AW8737) と周辺電源 (マイク・LCD バックライト) は
// ESP の GPIO ではなく PM1 の GPIO にぶら下がっている:
//   PYG2 = LCD BL / MIC / SPK の電源、PYG3 = SPK アンプ有効
// レジスタは vendor の M5PM1 ライブラリ (m5stack/M5PM1) から写した。
#pragma once
#include <stdbool.h>
#include "esp_err.h"

esp_err_t pm1_init(void);                 // I2C は audio 側で初期化済みの前提
esp_err_t pm1_speaker_enable(bool on);    // PYG3 (+PYG2 の周辺電源も on にする)
// 外部 5V ブースト (SY7088, Grove/HAT 給電) の on/off。使わない時は off:
// USB 給電中に軽負荷で PFM 動作してコイル鳴き (~10.7kHz) の発生源になる
esp_err_t pm1_ext_output(bool on);
// 充電器 (CHG_EN) の on/off。USB 給電中の充電スイッチングが鳴く疑いがあるとき、デモ中だけ切る
esp_err_t pm1_charge_enable(bool on);
