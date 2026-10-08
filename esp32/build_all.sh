#!/bin/bash
# S3 と P4 の両方をビルドする (コンパイル確認)。実機書き込みは idf.py -p PORT flash、モデルは parttool で model 区画へ。
set -e
cd "$(dirname "$0")"
. ~/esp/esp-idf/export.sh > /dev/null
for t in esp32s3 esp32p4; do
  echo "== $t =="
  idf.py -B build_$t -DSDKCONFIG=sdkconfig.$t set-target $t > build_$t.log 2>&1 || { tail -20 build_$t.log; exit 1; }
  idf.py -B build_$t -DSDKCONFIG=sdkconfig.$t build >> build_$t.log 2>&1 || { grep -E "error|Error" build_$t.log | head -20; exit 1; }
  grep -E "komimi_bench.bin binary size|Used static|bytes" build_$t.log | tail -3
done
