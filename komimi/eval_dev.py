"""dev マニフェストでカナ CER を測る。参照実装 (float / int8 模擬、GPU 可) か、C エンジンの CLI か。

  python -m komimi.eval_dev --kmm out/ja_v1.kmm --manifest dev.jsonl --limit 300 [--quant] [--device cuda:N]
  python -m komimi.eval_dev --cli csrc/komimi_cli --kmm out/ja_v1.kmm --manifest dev.jsonl --limit 300 [--cli_args "--stream --chunk 16 --left 128"]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata

import jiwer
import numpy as np
import soundfile as sf
import torch


def clean_ja(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in t)
    return re.sub(r"[^ァ-ヺー]", "", t)


def pieces_to_text(ids: list[int], tokens: list[str]) -> str:
    return "".join(tokens[i] for i in ids if i < len(tokens)).replace("▁", " ").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmm", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=300); ap.add_argument("--quant", action="store_true")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--cli", default=""); ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--left", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=16, help="torch 側のバッチ (chunk 指定時は学習側 komimi_train/model.py のバッチ版を使う)")
    ap.add_argument("--offset", type=int, default=0, help="torch chunk 経路: チャンク格子のずらし (フレーム、学習時の --chunk_shift の検証用)")
    ap.add_argument("--pad_noise_db", type=float, default=None, help="--pad_ms の詰め物を 0 でなくこの dBFS の白色雑音にする")
    ap.add_argument("--pad_ms", type=int, default=0, help="先頭に無音を足してチャンク境界の位相をずらす (ms、CLI 経路)")
    ap.add_argument("--cli_args", default="", help='CLI のオプション (例 "--stream --chunk 16 --left 128")')
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8")][:a.limit]
    refs = [clean_ja(it["text"]) for it in items]
    audio = sum(float(it["duration"]) for it in items)
    hyps: list[str] = []
    t0 = time.time()
    if a.cli:
        tmp = tempfile.mkdtemp(prefix="komimi_eval_")
        paths = []
        for k, it in enumerate(items):
            x, sr = sf.read(it["wav"], dtype="int16")
            if a.pad_ms:
                n = sr * a.pad_ms // 1000
                pad = np.zeros(n, dtype=x.dtype) if a.pad_noise_db is None else (np.random.default_rng(k).standard_normal(n) * 32768 * 10 ** (a.pad_noise_db / 20)).astype(x.dtype)
                x = np.concatenate([pad, x])
            p = os.path.join(tmp, f"{k:05d}.wav"); sf.write(p, x, sr, subtype="PCM_16"); paths.append(p)
        t0 = time.time()
        got: dict[str, str] = {}
        cpu_s = 0.0
        for i in range(0, len(paths), 32):
            out = subprocess.run([a.cli] + a.cli_args.split() + [a.kmm] + paths[i:i + 32], capture_output=True, text=True, timeout=3600)
            for line in out.stdout.splitlines():
                if "\t" in line:
                    p, txt = line.split("\t", 1); got[p.strip()] = txt.strip()
            m = re.search(r"compute_s=([0-9.]+)", out.stderr)
            if m: cpu_s += float(m.group(1))
        hyps = [clean_ja(got.get(p, "")) or "-" for p in paths]
        shutil.rmtree(tmp, ignore_errors=True)
        wall = time.time() - t0
        extra = f" compute RTF {cpu_s/audio:.3f}" if cpu_s else ""
    else:
        from . import kmm, ref_torch as R
        hdr, w, tokens = kmm.read(a.kmm)
        blank = hdr["vocab"] - 1
        if a.chunk > 0:
            try:
                from komimi_train.model import Komimi                           # 学習側 (非公開パッケージ) があるときだけ
            except ImportError:
                raise SystemExit("torch の chunk 経路には学習側パッケージ komimi_train が必要です。C エンジンで測るには --cli を使ってください")
            m = Komimi(vocab=hdr["vocab"], n_layers=hdr["n_layers"]).eval(); m.load_dict(w); m.set_fake_quant(a.quant); m.to(a.device)
            with torch.no_grad():
                for i in range(0, len(items), a.batch):
                    b = items[i:i + a.batch]
                    wavs = [torch.from_numpy(sf.read(it["wav"], dtype="float32")[0]) for it in b]
                    lens = torch.tensor([x.numel() for x in wavs]).to(a.device)
                    x = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(a.device)
                    lp, olen = m(x, lens, a.chunk, a.left, a.offset)
                    for k in range(len(b)):
                        hyps.append(clean_ja(pieces_to_text(R.greedy(lp[k, :olen[k]], blank), tokens)) or "-")
        else:
            w = {k: v.to(a.device) for k, v in w.items()}
            R.QUANT["on"] = a.quant
            with torch.no_grad():
                for it in items:
                    wav = torch.from_numpy(sf.read(it["wav"], dtype="float32")[0]).to(a.device)
                    lp = R.logits(wav, w)
                    hyps.append(clean_ja(pieces_to_text(R.greedy(lp, blank), tokens)) or "-")
        wall = time.time() - t0; extra = f" chunk{a.chunk}/left{a.left}" if a.chunk > 0 else ""
    cer = jiwer.cer(refs, hyps)
    print(f"n={len(refs)} kana-CER {cer:.2%}  wall RTF {wall/audio:.3f}{extra} (audio {audio/60:.1f} min, {wall:.0f}s)"
          + (" [int8 sim]" if a.quant and not a.cli else ""))
    for r, h in list(zip(refs, hyps))[:a.show]:
        print(f"  ref {r[:34]}\n  hyp {h[:34]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
