#!/bin/bash
# M5StickS3 向けビルド (8MB flash の区画)。書き込み: ./flash_m5sticks3.sh <model.kmm> [port]
set -e
cd "$(dirname "$0")"
. ~/esp/esp-idf/export.sh > /dev/null
idf.py -B build_m5sticks3 -DSDKCONFIG=sdkconfig.m5sticks3 -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.m5sticks3" set-target esp32s3 > build_m5sticks3.log 2>&1
idf.py -B build_m5sticks3 -DSDKCONFIG=sdkconfig.m5sticks3 build >> build_m5sticks3.log 2>&1 || { grep -E "error" build_m5sticks3.log | head; exit 1; }
grep -E "binary size" build_m5sticks3.log | tail -1
