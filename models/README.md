# 学習済み重み (.kmm、int8)

| ファイル | 構成 | 大きさ | 想定チャンク | dev 300 カナ CER (チャンク逐次) | 全文脈 | 想定端末 (実測 RTF) |
|---|---|---|---|---|---|---|
| `ja_v12m_i8_c32.kmm` (`_c16` も) | 16 層 / d=176 / ff 704 / 4 ヘッド (約 13M) | 13.4 MB | 32 (1,280 ms) | **21.33%** (chunk 32) / 22.61% (chunk 16) | 18.8% | ESP32-P4: RTF 0.85 (chunk 32)。v10 と同じ大きさで 7,000 時間 + カリキュラム + 発話前の増強、ゼロから (Apache-2.0) |
| `ja_v12a_i8_c32.kmm` (`_c16` も) | 16 層 / d=256 / ff 1024 / 4 ヘッド (約 30M) | 28.1 MB | 32 (1,280 ms) | **19.69%** (chunk 32) / 21.08% (chunk 16) | 17.4% | PC、ブラウザ、(NEON があれば) ラズパイ級。v11a と同じ大きさで 7,000 時間 + 増強、ゼロから (Apache-2.0) |
| `ja_v12s_i8_c16.kmm` (`_c32` は既定チャンクが違うだけ) | 8 層 / d=176 / sub_ch 88 (約 6.5M) | 7.1 MB | 16 (640 ms) | **24.38%** (chunk 16) / 22.08% (chunk 32) | 20.2% | ESP32-S3: RTF 0.93 (chunk 16)。v10s と同じ大きさで 7,000 時間 + カリキュラム学習、ゼロから (Apache-2.0)。発話の前に無音・雑音を足す増強つき (実機の出だしに強い) |
| `ja_v12_i8_c32.kmm` (`_c16` も。106 MB は GitHub の上限を超えるので [Releases](https://github.com/genai-craft/komimi/releases/tag/v12) から) | 16 層 / d=512 / ff 2048 / 8 ヘッド (約 100M) | 106 MB | 32 (1,280 ms) | **18.65%** (chunk 32) / 20.11% (chunk 16) | 16.8% | PC (AVX2 で RTF 0.09)、ブラウザ (取得 106 MB)。7,000 時間、ゼロから (Apache-2.0)、発話前の増強つき |


## 公開モデルとの比較 (人手の正解、2026-10-17)

上の dev CER は教師 (kana-whisper) との一致率なので、人手の正解がある公開セットで他のモデルと並べた。正解も出力も pyopenjtalk の g2p でカナに直した
**カナ CER** (表記ゆれを除いた「音」の誤り)。FLEURS ja は読み上げ 300 文、ReazonSpeech test は TV 400 文。ReazonSpeech test の字幕には音声にある語が
無いことがあり、ReazonSpeech で学習したモデル (komimi、zipformer、moonshine-ja) が有利に出るので、地力は FLEURS の列で読む。

| モデル | 大きさ | 動かし方 | FLEURS ja | ReazonSpeech test |
|---|---|---|---|---|
| komimi `ja_v12s` | 6.5M | ストリーミング chunk 16 / left 128 (CPU) | 23.1% | 31.3% |
| komimi `ja_v12s` | 6.5M | 全文脈 | 18.2% | 28.6% |
| komimi `ja_v12m` | 13M | ストリーミング chunk 32 / left 128 | 16.5% | 29.5% |
| komimi `ja_v12m` | 13M | 全文脈 | 13.6% | 28.3% |
| komimi `ja_v12a` | 30M | ストリーミング chunk 32 / left 128 | 15.4% | 28.9% |
| komimi `ja_v12a` | 30M | 全文脈 | 13.1% | 27.8% |
| [Ralomi](https://huggingface.co/ayousanz/ralomi) 1.0 (ESP32-S3 向け DFSMN-CTC、ひらがな出力) | 2.05M | 1 秒ずつ処理 (入力 ≤15 s。FLEURS は 76 文を 15 s で切断、≤15 s の 224 文では 29.4% 対 ja_v12s 22.2%) | 31.9% | **22.2%** |
| moonshine-tiny-ja | 27M | encoder-decoder、非ストリーミング | **7.1%** | 8.0% |
| moonshine-base-ja | 62M | 同上 | 7.6% | 5.0% |
| ReazonSpeech zipformer (sherpa-onnx int8) | 80M | ストリーミング (CPU) | 5.0% | **4.6%** |
| SenseVoiceSmall (sherpa-onnx int8) | 234M | 非ストリーミング (CPU) | 5.8% | 19.6% |
| kotoba-whisper-v2.0 | 756M | 非ストリーミング (GPU) | **4.0%** | 8.7% |

Ralomi (2026-10-17 公開、ESP32-S3 で RTF 0.45、16 bit 量子化) との比較は土俵で割れる: TV 音声の ReazonSpeech test では Ralomi 22.2% 対 `ja_v12s` 31.2% で Ralomi が良く、
読み上げの FLEURS では `ja_v12s` 22.2% 対 Ralomi 29.4% (15 s 以下の 224 文) で komimi が良い。差の主因は正解ラベルの作り方 (Ralomi は字幕の読み、komimi は kana-whisper の疑似ラベル)。

読み方: **komimi は同じ大きさの moonshine-tiny-ja に大差で負ける** (30M の `ja_v12a` を全文脈で使っても 13.1% 対 7.1%)。komimi の値打ちは精度ではなく、
ESP32-S3 / P4 で int8 のストリーミング CTC が動くことと、候補の CTC 採点 (決まった言い方から選ぶ用途) にある。PC やブラウザで自由認識の精度が要るなら
moonshine-ja や zipformer を使う方がよい。差の主因は学習データ (疑似ラベル約 7,000 時間) と大きさ。改善の要望は Issue へ。

- 出力はカタカナ (長音「ー」・促音「ッ」を含む)。語彙は SentencePiece 1,024 で、token 表はファイルの中にある。
- 動的チャンクで学習しているので、同じ重みを `--chunk 0` (全文脈) / 16 / 32 のどれでも使える。ヘッダの chunk/left は既定値 (CLI も端末も省略時はこれを使う)。
- 学習データは ReazonSpeech 約 7,000 時間、教師 (読みの正解) は sbintuitions/kana-whisper。dev も同じ教師の読みを正解にした発話なので、
  上の CER は「教師との一致率」であって人手の正解に対する値ではない。
- 実機では発話の前に音 (無音・雑音) が続いているぶん、dev の数字より 1 ポイントほど悪くなる (`docs/design.md` の位相の節)。
- 由来と権利: 学習音声は ReazonSpeech (CDLA-Sharing-1.0、著作権法 30 条の 4 の情報解析目的で利用。音声・字幕そのものは再配布しない)、
  読みの教師は sbintuitions/kana-whisper (MIT)。学習して得た重みは CDLA の「結果 (Results)」で、CDLA 側の制約は受けない。
- **重みのライセンス**: ここにある重み (v12 / v12a / v12m / v12s) はすべて**ゼロから学習** (NeMo などの公開重みを使っていない) で **Apache-2.0**。コードも Apache-2.0。
- 旧世代 (v10 / v10s / v11a) は配布をやめた (2026-10-09)。数字の推移は `docs/design.md`。v10 / v10s は NVIDIA NeMo `stt_en_conformer_ctc_small` の公開重み
  (CC-BY-4.0) を初期値に微調整した系統で、git の履歴に残るファイルは CC-BY-4.0 (NVIDIA への帰属表示つき、NOTICE 参照) に従う。
- 学習コードとデータは公開していない。重みの改善 (データ追加・再学習) の要望は Issue へ。
