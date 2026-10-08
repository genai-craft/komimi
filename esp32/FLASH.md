# 焼き方

## 推奨: PC 側エージェント経由 (サーバーから直接)

手元 PC (端末が刺さっている方) で一度だけ:

```powershell
scp <user>@<server>:<komimi の場所>/tools/pc_agent.py .
python pc_agent.py            # 自分で ssh -R 4010 を張る (サーバーのパスワードを 1 回聞かれる)。以後は放置
```

サーバー側 (このリポジトリ):

```bash
python tools/device.py ports
python tools/device.py flash esp32/build_esp32p4/komimi_p4_full_ja_v5_i8_c32.bin --port COM7 --chip esp32p4 --baud 921600
python tools/device.py monitor --port COM7 --seconds 30
python tools/device.py flash esp32/build_m5sticks3/komimi_m5sticks3_full_ja_v8_i8_c16.bin --port COM4 --chip esp32s3 --baud 460800
python tools/device.py monitor --port COM4 --seconds 40
```

アプリだけ差し替えるときは `komimi_p4_app.bin` / `komimi_m5sticks3_app.bin` (300 KB)。モデル区画 (P4: 0x110000、S3: 0x90000) は残る。
RFC2217 (esp_rfc2217_server) は、ESP32 の USB-Serial/JTAG がリセットで再列挙されて握りが死ぬため使えなかった。

## 手動 (PC の esptool で直接)

```powershell
python -m esptool --chip esp32p4 -p COM7 -b 921600 write_flash 0x0 komimi_p4_full_ja_v5_i8_c32.bin
python -m esptool --chip esp32s3 -p COM4 -b 460800 write_flash 0x0 komimi_m5sticks3_full_ja_v8_i8_c16.bin
python -m serial.tools.miniterm COM7 115200
```

## いまの推奨イメージ

| 端末 | イメージ | モデル | RTF |
|---|---|---|---|
| ESP32-P4 (16 MB flash / 32 MB PSRAM) | `build_esp32p4/komimi_p4_full_ja_v10_i8_c32.bin` | v10 16 層 int8、chunk 32 (遅延 1.3 s)、dev300 CER 23.25% | 0.85 |
| M5StickS3 (8 MB flash / 8 MB PSRAM) | `build_m5sticks3/komimi_m5sticks3_full_ja_v10s_i8_c16.bin` | v10s 8 層 sub_ch 88 int8、chunk 16 (遅延 0.65 s)、dev300 CER 28.82% | 0.93 |
