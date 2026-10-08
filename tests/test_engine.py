"""komimi の整合性試験。重みは乱数 (学習済みモデルに依存しない) で、
  1. .kmm の往復、2. C エンジン (float、一括) == torch 参照 (ref_torch)、3. C ストリーミング (chunk ≥ 系列長) == torch 参照、
  4. バッチ版 == 単発話参照、5. チャンクマスク == 逐次計算 (torch)、6. C ストリーミング == torch チャンクマスク、
  7. random_weights() の鍵・形 == 学習用モデルの to_dict()。
1〜3 は公開パッケージ (komimi.kmm / komimi.ref_torch) だけで動く。4〜7 は学習用の nn.Module (非公開 repo komimi-train の
komimi_train.model.Komimi) が要るので、import できないときは skip する。C の試験は csrc/komimi_cli がビルド済みのときだけ
(make -C csrc komimi_cli)。"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from komimi import kmm, ref_torch as R  # noqa: E402

try:                                                           # 学習用モデルは非公開 repo (komimi-train) にある。無くても 1〜3 は動く
    from komimi_train.model import Komimi  # noqa: E402
except ImportError:
    Komimi = None

CLI = ROOT / "csrc" / "komimi_cli"
VOCAB = 65                                                     # 64 トークン + blank (最後の列)
TOKENS = [f"t{i}" for i in range(VOCAB - 1)]

needs_cli = pytest.mark.skipif(not CLI.exists(), reason="csrc/komimi_cli がビルドされていない (make -C csrc komimi_cli)")
needs_train = pytest.mark.skipif(Komimi is None, reason="学習用モデル komimi_train.model.Komimi が import できない (非公開 repo komimi-train を PYTHONPATH に置くと動く)")


def random_weights(n_layers: int = 2, vocab: int = VOCAB, seed: int = 0) -> dict:
    """komimi_train.model.Komimi(vocab, n_layers).to_dict() と同じ鍵・形・並びの重み辞書を、nn.Module なしに乱数から作る
    (kmm.write / ref_torch.logits が読む鍵をすべて含む)。学習パラメータは N(0, 0.08²) (LayerNorm の weight も含めて、
    旧 small_model と同じ)、mel.fb / mel.window は ref_torch と同じ定数、特徴の固定正規化は mean=-10, std=3。
    "_meta" は入れない (ストリーミング用の .kmm を書くときは呼び出し側で w["_meta"] = {"stream": True})。"""
    g = torch.Generator().manual_seed(seed)
    d, h, ff, k, nmel = R.D, R.H, R.FF, R.K, R.NMEL

    def rnd(*shape):
        return torch.randn(*shape, generator=g) * 0.08

    w: dict[str, torch.Tensor] = {}
    w["sub.conv1.weight"] = rnd(d, 1, 3, 3); w["sub.conv1.bias"] = rnd(d)
    w["sub.conv2.weight"] = rnd(d, d, 3, 3); w["sub.conv2.bias"] = rnd(d)
    w["sub.out.weight"] = rnd(d, d * ((nmel - 1) // 4 + 1)); w["sub.out.bias"] = rnd(d)      # 80 → 20 (striding ×4) × sub_ch
    for i in range(n_layers):
        p = f"layers.{i}."
        w[p + "att.bias_u"] = rnd(h, d // h); w[p + "att.bias_v"] = rnd(h, d // h)
        for n in ("norm_ff1", "norm_att", "norm_conv", "norm_ff2", "norm_out"):
            w[p + n + ".weight"] = rnd(d); w[p + n + ".bias"] = rnd(d)
        for n in ("ff1", "ff2"):
            w[p + n + ".1.weight"] = rnd(ff, d); w[p + n + ".1.bias"] = rnd(ff)
            w[p + n + ".2.weight"] = rnd(d, ff); w[p + n + ".2.bias"] = rnd(d)
        for n in ("q", "k", "v", "out"):
            w[p + f"att.{n}.weight"] = rnd(d, d); w[p + f"att.{n}.bias"] = rnd(d)
        w[p + "att.pos.weight"] = rnd(d, d)                                                    # bias なし
        w[p + "conv.pw1.weight"] = rnd(2 * d, d); w[p + "conv.pw1.bias"] = rnd(2 * d)          # GLU で半分に
        w[p + "conv.pw2.weight"] = rnd(d, d); w[p + "conv.pw2.bias"] = rnd(d)
        w[p + "conv.dw.weight"] = rnd(d, 1, k); w[p + "conv.dw.bias"] = rnd(d)                 # depthwise (groups=d)
    w["head.weight"] = rnd(vocab, d); w["head.bias"] = rnd(vocab)
    w["mel.fb"] = R.mel_filterbank()                                                           # (80, 257)
    w["mel.window"] = torch.hann_window(R.WIN, periodic=False)                                 # (400,)
    w["mel.norm_mean"] = torch.full((nmel,), -10.0); w["mel.norm_std"] = torch.full((nmel,), 3.0)
    return w


def small_model(seed=0, n_layers=2):
    """random_weights() を学習用 nn.Module に載せたもの (重みは辞書版と同一。mel.norm_* があるので fixed_norm になる)。"""
    m = Komimi(vocab=VOCAB, n_layers=n_layers).eval()
    m.load_dict(random_weights(n_layers, VOCAB, seed))
    return m


def wavs(n=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(16000 * (k + 1) + 777, generator=g) * 0.1 for k in range(n)]


def _wav16(path, x):
    import soundfile as sf
    sf.write(str(path), (x.numpy() * 32767).astype(np.int16), 16000, subtype="PCM_16")


def _quantized(x):
    """CLI と同じ PCM16 量子化を通した波形。"""
    return torch.from_numpy((x.numpy() * 32767).astype(np.int16).astype(np.float32) / 32768.0)


def _parse_cli_tokens(out: str) -> list[int]:
    """CLI の 1 行 "path<TAB>text" からトークン id 列へ。km_detok はトークンを区切り無しで連結する ("▁" だけが空白になる) ので、
    "t<番号>" を正規表現で切り出す (各トークンが t で始まるので一意に戻せる)。"""
    text = out.split("\t", 1)[1]
    return [TOKENS.index(t) for t in re.findall(r"t\d+", text)]


# ---- 重み辞書だけで動く試験 ----

def test_kmm_roundtrip(tmp_path):
    w = random_weights(); w["_meta"] = {"stream": True}
    path = tmp_path / "t.kmm"
    kmm.write(str(path), w, TOKENS, bits=32, chunk=4, left=8)
    hdr, w2, tok = kmm.read(str(path))
    assert hdr["d"] == 176 and hdr["n_layers"] == 2 and hdr["vocab"] == VOCAB and tok[3] == "t3"
    assert hdr["heads"] == 4 and hdr["ff"] == 704 and hdr["kernel"] == 31 and hdr["n_mel"] == 80 and hdr["sub_ch"] == 176
    assert hdr["flags"] == kmm.FLAG_FIXED_NORM | kmm.FLAG_STREAM and hdr["chunk"] == 4 and hdr["left"] == 8
    assert hdr["win"] == 400 and hdr["nfft"] == 512 and hdr["bits"] == 32
    assert set(w2) == {k for k in w if not k.startswith("_")}
    for k, v in w.items():
        if not k.startswith("_"):
            assert torch.allclose(v, w2[k], atol=1e-6), k
    kmm.write(str(path), w, TOKENS, bits=8)
    _, w8, _ = kmm.read(str(path))
    err = (w8["layers.0.ff1.1.weight"] - w["layers.0.ff1.1.weight"]).abs().max()
    assert err < w["layers.0.ff1.1.weight"].abs().max() / 100


@needs_cli
def test_c_engine_matches_torch(tmp_path):
    """C エンジン (float、一括認識 km_recognize) の貪欲復号 == ref_torch の貪欲復号。ヘッダの chunk/left は一括では使われない。"""
    w = random_weights(); w["_meta"] = {"stream": True}
    path = tmp_path / "m.kmm"; kmm.write(str(path), w, TOKENS, bits=32, chunk=4, left=8)
    x = wavs(1)[0]; wp = tmp_path / "a.wav"; _wav16(wp, x)
    with torch.no_grad():
        full = R.logits(_quantized(x), w)
    out = subprocess.run([str(CLI), str(path), str(wp)], capture_output=True, text=True, check=True).stdout
    c_ids = _parse_cli_tokens(out)
    assert c_ids == R.greedy(full, VOCAB - 1)


@needs_cli
def test_c_stream_full_context_matches_torch(tmp_path):
    """C ストリーミング (--stream) で chunk・left を系列長より大きく取ると全文脈と同じになるはずで、その logits == ref_torch。
    (1 秒の波形は 27 フレームなので chunk=32, left=64 で全部見える)。学習用モデル無しで C のストリーミング経路を検算する。"""
    w = random_weights(); w["_meta"] = {"stream": True}
    path = tmp_path / "m.kmm"; kmm.write(str(path), w, TOKENS, bits=32, chunk=32, left=64)
    x = wavs(1)[0]; wp = tmp_path / "a.wav"; _wav16(wp, x)
    with torch.no_grad():
        full = R.logits(_quantized(x), w)
    assert full.shape[0] <= 32
    dump = tmp_path / "lg.f32"
    subprocess.run([str(CLI), "--stream", "--chunk", "32", "--left", "64", "--piece", "1000", "--dump", str(dump), str(path), str(wp)],
                   capture_output=True, text=True, check=True)
    c = torch.log_softmax(torch.from_numpy(np.fromfile(dump, dtype=np.float32).reshape(-1, VOCAB)), -1)
    assert c.shape == full.shape
    assert (c - full).abs().max() < 2e-3


# ---- 学習用モデル (komimi_train) が要る試験 ----

@needs_train
def test_random_weights_matches_model_layout():
    """hard-code した鍵・形・並びが学習用 nn.Module の to_dict() と一致する (モデル側が変わったらここで気づく)。"""
    w = random_weights(); w2 = small_model().to_dict()
    assert list(w) == list(w2)
    for k in w:
        assert tuple(w[k].shape) == tuple(w2[k].shape), k
        assert torch.equal(w[k], w2[k]), k


@needs_train
def test_batched_matches_reference():
    m = small_model(); w = m.to_dict()
    xs = wavs(2)
    lens = torch.tensor([x.numel() for x in xs]); B = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    with torch.no_grad():
        lp, olen = m(B, lens)
        for i, x in enumerate(xs):
            ref = R.logits(x, w)
            assert ref.shape[0] == int(olen[i])
            assert (lp[i, :olen[i]] - ref).abs().max() < 1e-3


@needs_train
def test_chunk_mask_equals_incremental():
    m = small_model()
    x = wavs(1)[0][None]
    with torch.no_grad():
        feat, flen = m.features(x, torch.tensor([x.shape[1]]))
        full, _ = m.encode(feat, flen, chunk=4, left=8)
        xs = feat[:, None]; xs = torch.relu(m.sub["conv1"](xs)); xs = torch.relu(m.sub["conv2"](xs))
        Bn, C, T, Fd = xs.shape; xs = m.sub["out"](xs.transpose(1, 2).reshape(Bn, T, C * Fd)) * 176 ** 0.5
        hs = [xs] + [torch.zeros_like(xs) for _ in m.layers]; glu = [torch.zeros_like(xs) for _ in m.layers]
        half = 15
        for cs in range(0, T, 4):
            ce = min(T, cs + 4); s0 = max(0, cs - 8); a0 = cs - s0
            for l, layer in enumerate(m.layers):
                seg = hs[l][:, s0:ce]; Ts = seg.shape[1]
                pe = R.rel_pos_table(Ts, 176); mask = torch.ones(1, Ts, Ts, dtype=torch.bool)
                xx = seg + 0.5 * layer.ff1["2"](torch.nn.functional.silu(layer.ff1["1"](layer.norm_ff1(seg))))
                xx = xx + layer.attention(layer.norm_att(xx), pe, mask)
                xc = xx[:, a0:]
                g = torch.nn.functional.glu(layer.conv["pw1"](layer.norm_conv(xc)), dim=-1); glu[l][:, cs:ce] = g
                lg = glu[l][:, max(0, cs - half):cs]
                segc = torch.nn.functional.pad(torch.cat([lg, g], 1).transpose(1, 2), (half - lg.shape[1], half))
                xc = xc + layer.conv["pw2"](torch.nn.functional.silu(layer.conv_dw(segc).transpose(1, 2)))
                xc = xc + 0.5 * layer.ff2["2"](torch.nn.functional.silu(layer.ff2["1"](layer.norm_ff2(xc))))
                hs[l + 1][:, cs:ce] = layer.norm_out(xc)
        assert (full - hs[-1]).abs().max() < 1e-4


@needs_cli
@needs_train
@pytest.mark.parametrize("chunk,left", [(4, 8), (32, 64)])
def test_c_stream_matches_torch_chunk_mask(tmp_path, chunk, left):
    """C ストリーミング (--stream、1000 サンプルずつ供給) の logits == 学習用モデルのチャンクマスク版 (chunk, left)。"""
    m = small_model(); w = m.to_dict(); w["_meta"] = {"stream": True}
    path = tmp_path / "m.kmm"; kmm.write(str(path), w, TOKENS, bits=32, chunk=chunk, left=left)
    x = wavs(1)[0]; wp = tmp_path / "a.wav"; _wav16(wp, x)
    xq = _quantized(x)
    with torch.no_grad():
        lp_s, olen = m(xq[None], torch.tensor([xq.numel()]), chunk=chunk, left=left)
    dump = tmp_path / "lg.f32"
    subprocess.run([str(CLI), "--stream", "--chunk", str(chunk), "--left", str(left), "--piece", "1000", "--dump", str(dump), str(path), str(wp)],
                   capture_output=True, text=True, check=True)
    c = torch.log_softmax(torch.from_numpy(np.fromfile(dump, dtype=np.float32).reshape(-1, VOCAB)), -1)
    assert c.shape[0] == int(olen[0])
    assert (c - lp_s[0, :olen[0]]).abs().max() < 2e-3
