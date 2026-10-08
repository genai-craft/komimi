#!/bin/bash
# komimi デモ (M5StickS3) のビルド。モデルは komimi_m5sticks3_full_* で焼いた区画 (0x90000) をそのまま使う
set -e
cd "$(dirname "$0")"
. ~/esp/esp-idf/export.sh > /dev/null
idf.py -B build set-target esp32s3 > build.log 2>&1
idf.py -B build build >> build.log 2>&1 || { grep -E "error" build.log | grep -v Wno-error | head; exit 1; }
grep -E "binary size" build.log | tail -1
cd build && python -m esptool --chip esp32s3 merge_bin -o komimi_demo_s3_app.bin --flash_size 8MB @flash_args | tail -1
