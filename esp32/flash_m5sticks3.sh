#!/bin/bash
# アプリと .kmm を焼く。port は rfc2217://localhost:4000?ign_set_control (手元 PC の esp_rfc2217_server 経由) か /dev/ttyACM0
set -e
cd "$(dirname "$0")"
MODEL=${1:?model.kmm}; PORT=${2:-'rfc2217://localhost:4000?ign_set_control'}
. ~/esp/esp-idf/export.sh > /dev/null
idf.py -B build_m5sticks3 -DSDKCONFIG=sdkconfig.m5sticks3 -p "$PORT" flash
python -m esptool --chip esp32s3 -p "$PORT" -b 460800 write_flash 0x90000 "$MODEL"
