"""Python 束縛 (komimi/engine.py、csrc/libkomimi.so) の試験。学習済みモデル (models/*.kmm) と合成音 (正弦波 + 雑音) で、
  1. km_recognize_ex の logits の greedy == km_recognize の greedy (CLI と同じ全文脈の経路)
  2. CTC forward (numpy) == torch.nn.functional.ctc_loss
  3. トークナイザ: 語彙表 (models/ja1024_vocab.json) の Viterbi 分割が piece を繋ぐと元に戻る
共有ライブラリは make -C csrc libkomimi.so で作る (無ければ skip)。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LIB = ROOT / "csrc" / "libkomimi.so"
MODEL = ROOT / "models" / "ja_v12s_i8_c16.kmm"
pytestmark = pytest.mark.skipif(not LIB.exists() or not MODEL.exists(), reason="make -C csrc libkomimi.so が要る")


def _wav(sec=2.0, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(int(16000 * sec)) / 16000
    return (0.1 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_logits_greedy_matches_cli(tmp_path):
    import soundfile as sf
    from komimi.engine import Engine
    e = Engine(MODEL); w = _wav()
    p = tmp_path / "a.wav"; sf.write(p, w, 16000, subtype="PCM_16")      # CLI は 16 bit PCM の wav を読む
    w = sf.read(p, dtype="float32")[0]                                     # 束縛にも同じ量子化後の音を渡す
    cli = ROOT / "csrc" / "komimi_cli_avx2"
    if not cli.exists():
        pytest.skip("make -C csrc komimi_cli_avx2")
    out = subprocess.run([str(cli), str(MODEL), str(p)], capture_output=True, text=True).stdout.strip().splitlines()
    want = next((l.split("\t")[1] if "\t" in l else "" for l in out if l.startswith(str(p))), "").replace(" ", "")
    lp = e.logprobs(w)
    assert lp.shape[1] == 1025 and np.allclose(np.exp(lp).sum(1), 1, atol=1e-4)
    assert e.decode(e.greedy(lp)) == want


def test_ctc_forward_matches_torch():
    torch = pytest.importorskip("torch")
    from komimi.engine import Engine
    e = Engine(MODEL); lp = e.logprobs(_wav(1.5, 1))
    seqs = [e.tokenize("ソシテ"), e.tokenize("トーキョーエキ"), [7, 7, 8], []]
    ours = e.ctc_score(lp, seqs)
    for s, o in zip(seqs, ours):
        ref = -torch.nn.functional.ctc_loss(torch.from_numpy(lp)[:, None], torch.tensor([s], dtype=torch.long), torch.tensor([lp.shape[0]]),
                                            torch.tensor([len(s)]), blank=e.blank, reduction="sum").item()
        assert abs(o - ref) < 1e-3 * max(1.0, abs(ref))


def test_tokenizer_roundtrip():
    from komimi.engine import Tokenizer
    t = Tokenizer()
    for s in ["ソシテトクテンオ", "ハイオセワニナッテオリマス", "キョーワイーテンキデスネ", "ヴァイオリン"]:
        ids = t.encode(s)
        assert ids[0] in (t.index["▁"], *[i for p, i in t.index.items() if p.startswith("▁")])
        assert t.decode(ids) == s
