#!/bin/bash
# ESP32-P4 向けビルド + 焼きイメージ (16MB flash: app 1MB @0x10000、model 14MB @0x110000)
set -e
cd "$(dirname "$0")"
. ~/esp/esp-idf/export.sh > /dev/null
idf.py -B build_esp32p4 -DSDKCONFIG=sdkconfig.esp32p4 set-target esp32p4 > build_esp32p4.log 2>&1
idf.py -B build_esp32p4 -DSDKCONFIG=sdkconfig.esp32p4 build >> build_esp32p4.log 2>&1 || { grep -E "error" build_esp32p4.log | head; exit 1; }
grep -E "binary size" build_esp32p4.log | tail -1
cd build_esp32p4
python -m esptool --chip esp32p4 merge_bin -o komimi_p4_app.bin --flash_size 16MB @flash_args | tail -1
for M in "$@"; do
  name=$(basename "$M" .kmm)
  python -m esptool --chip esp32p4 merge_bin -o komimi_p4_full_$name.bin --flash_size 16MB @flash_args 0x110000 "$M" | tail -1
done
