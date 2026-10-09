# komimi (小耳)

小さな Conformer-CTC の日本語 (カナ) 音声認識を **ESP32-S3 / ESP32-P4 / PC / ブラウザ** で動かすための、独自実装の推論エンジンと学習済み重み。コードも重みも Apache-2.0 ([models/README.md](models/README.md))。

デモ: <https://komimi.aunvox.com> — S3 用 / P4 用の重みを WebAssembly で動かして比較 (結果は実機と同じ、速さだけ違う)。教師の kana-whisper (809M) も WebGPU で並べて動かせる。

- `csrc/` — C エンジン。`km_recognize` (全文脈) と `km_stream_*` (チャンク逐次、KV キャッシュ、無音ゲート)。
  内積カーネルは `km_kernels.h` に集約 (generic C / AVX2 / ESP32-S3 PIE / ESP32-P4 SIMD)
- `models/` — 学習済み重み `.kmm` (int8)。数字は [models/README.md](models/README.md)
- `esp32/` — ESP-IDF 5.3 のベンチ (S3 / P4) と、M5StickS3 のマイク → 認識 → LCD デモ (`esp32/demo_s3`)。焼き方は [esp32/FLASH.md](esp32/FLASH.md)
- `web/` — WebAssembly 版 (Emscripten) とデモページ
- `komimi/` — Python 側: `engine.py` (C エンジンの ctypes 束縛、CTC 行列・greedy・候補の CTC 採点・トークナイザ)、`kmm.py` (重みファイル形式の読み書き)、`ref_torch.py` (float の参照実装、1 発話)、`eval_dev.py` (dev のカナ CER、C CLI を回す)
- `tools/` — 手元 PC につないだ端末へサーバーから焼く/モニタするためのエージェント
- `tests/` — 重みファイルの往復、C エンジン == 参照実装 (先に `make -C csrc`。学習側パッケージがあれば学習モデルとの整合も)
- `docs/design.md` — 設計、由来、実測値の記録

## 使い方 (PC)

```bash
make -C csrc                      # komimi_cli (generic C) / komimi_cli_avx2 / komimi_cli_att8 (端末と同じ int8 注意)
csrc/komimi_cli_avx2 models/ja_v12m_i8_c32.kmm a.wav                                   # 全文脈。1 行 "path<TAB>text"
csrc/komimi_cli_att8 --stream --chunk 32 --left 128 models/ja_v12m_i8_c32.kmm a.wav    # 端末と同じチャンク逐次
python -m komimi.eval_dev --cli csrc/komimi_cli_att8 --cli_args "--stream --chunk 32 --left 128" \
    --kmm models/ja_v12m_i8_c32.kmm --manifest dev.jsonl --limit 300                   # カナ CER
```

wav は 16 kHz モノラル。マニフェストは 1 行 1 発話の JSON `{"wav", "text", "duration"}` (text はカタカナ)。

## 使い方 (Python)

torch は要らない (numpy と ctypes だけ)。共有ライブラリは初回に `make -C csrc libkomimi.so` で作る。

```python
from komimi.engine import Engine
e = Engine("models/ja_v12a_i8_c32.kmm")
lp = e.logprobs(wav)                    # 16 kHz float → (T', 1025) の CTC log-softmax (40 ms ごと、blank = 1024)
e.decode(e.greedy(lp))                  # 自由認識 (カタカナ)
e.ctc_score(lp, [e.tokenize("トーキョーエキ"), e.tokenize("シンジュクエキ")])   # 候補ごとの log p(候補 | 音声)
```

候補を CTC で採点できるので、決まった言い方の中から選ぶ用途 (音声コマンド) では、自由認識の CER より正確に当たる。
[openvons](https://github.com/genai-craft/openvons) の音声デモは komimi をこの形で使い、ブラウザでは振り分けまで含めて WebAssembly で動かす。
`models/ja1024_vocab.json` は語彙 (SentencePiece unigram の piece と対数確率、全 `ja_*` モデル共通) で、候補カナ列の分割に使う。
C からは `km_recognize_ex()` で各フレームの logits を受け取れる (`csrc/km.h`)。WebAssembly で SIMD を使うなら `-msimd128 -DKM_KERNEL_WASM_SIMD`。

## 学習について

学習コード・学習データ・教師ラベルは公開していない (Aunvox の一環として非公開で管理)。公開しているのはランタイムと重み。
学習の方針と結果の数字は `docs/design.md` に記録してある。

## 改善要望・不具合

Issue / PR はこのリポジトリへ。エンジン・端末・デモページだけでなく、重みの改善 (データ追加・再学習) の要望も Issue で受け付ける。

## 由来と権利

`docs/design.md` を参照。エンジンは komimi 独自の実装で、アーキテクチャは NVIDIA NeMo の
Conformer-CTC Small (`stt_en_conformer_ctc_small`、設定と重みは CC-BY-4.0) の公開仕様に従う。
他実装のソースはコピーしていない (検証に黒箱として出力を比べただけ)。
