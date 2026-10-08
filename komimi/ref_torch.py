"""komimi の float 参照実装 (PyTorch)。NeMo Conformer-CTC Small の公開アーキテクチャ
(stt_en_conformer_ctc_small の model_config: 16 層 d=176 h=4 ff×4 k=31、striding ×4、rel_pos、xscaling)
を仕様から書き起こしたもの。C エンジン (csrc/) の正解値を作るのが役目で、速さは気にしない。

重みは komimi 自身の辞書 (.kmm を kmm.read で読んだもの、または学習側 komimi_train/export.py が作るもの) を受け取る。名前は NeMo の構成要素名に揃えてある。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

D, H, FF, K, NMEL, NFFT, WIN, HOP = 176, 4, 704, 31, 80, 512, 400, 160
LOG_GUARD = 2.0 ** -24
NORM_EPS = 1e-5
PREEMPH = 0.97


def mel_filterbank(sr=16000, n_fft=NFFT, n_mels=NMEL, fmin=0.0, fmax=None) -> torch.Tensor:
    """Slaney 流儀のメルフィルタ (librosa.filters.mel の既定と同じ定義。librosa には依存しない)。"""
    fmax = fmax or sr / 2
    def hz2mel(f):
        f = torch.as_tensor(f, dtype=torch.float64)
        m = 3.0 * f / 200.0
        min_log_hz, min_log_mel, logstep = 1000.0, 15.0, math.log(6.4) / 27.0
        return torch.where(f >= min_log_hz, min_log_mel + torch.log(f.clamp(min=1e-9) / min_log_hz) / logstep, m)
    def mel2hz(m):
        f = 200.0 * m / 3.0
        min_log_hz, min_log_mel, logstep = 1000.0, 15.0, math.log(6.4) / 27.0
        return torch.where(m >= min_log_mel, min_log_hz * torch.exp(logstep * (m - min_log_mel)), f)
    mels = torch.linspace(hz2mel(fmin).item(), hz2mel(fmax).item(), n_mels + 2, dtype=torch.float64)
    hz = mel2hz(mels)
    fft_f = torch.linspace(0, sr / 2, 1 + n_fft // 2, dtype=torch.float64)
    fdiff = hz[1:] - hz[:-1]
    ramps = hz[:, None] - fft_f[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    fb = torch.clamp(torch.minimum(lower, upper), min=0)
    enorm = 2.0 / (hz[2:n_mels + 2] - hz[:n_mels])
    return (fb * enorm[:, None]).to(torch.float32)


QUANT = {"on": False}


def q8_rows(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
    return torch.round(w / s).clamp(-127, 127) * s


def lin(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """F.linear。QUANT["on"] なら C エンジンと同じ int8 模擬: 重みは行ごと対称 int8、入力はベクトル (フレーム) ごと対称 int8。"""
    if QUANT["on"]:
        w = q8_rows(w)
        x = q8_rows(x)
    return F.linear(x, w, b)


def log_mel(wav: torch.Tensor, fb: torch.Tensor | None = None, window: torch.Tensor | None = None,
            norm_mean: torch.Tensor | None = None, norm_std: torch.Tensor | None = None) -> torch.Tensor:
    """(n,) 16kHz float → (T, 80) 正規化済み log-mel。NeMo の FilterbankFeatures と同じ手順:
    pre-emphasis 0.97 → STFT (n_fft 512, hop 160, hann 400 対称窓, center で前後 200 サンプルを零詰め) → |X|² → mel → log(x + 2^-24)
    → 特徴ごとに時間方向で標準化 (norm_mean/std を渡せば固定値で、ストリーミング用)。"""
    fb = mel_filterbank().to(wav.device) if fb is None else fb
    window = torch.hann_window(WIN, periodic=False, device=wav.device) if window is None else window
    # 前後 win/2 の零詰め → pre-emphasis の順 (C エンジンと同じ: 末尾の零詰め先頭は -0.97·x[n-1] になる)
    x = F.pad(wav, (NFFT // 2, NFFT // 2))                   # torch.stft は nfft 幅で切り出し、窓は中央 (前後 56) に置く
    x = torch.cat([x[:1], x[1:] - PREEMPH * x[:-1]])
    spec = torch.stft(x, NFFT, HOP, WIN, window=window, center=False, return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2                   # (257, T)
    mel = fb @ power                                          # (80, T)
    feat = torch.log(mel + LOG_GUARD)
    if norm_mean is not None:
        feat = (feat - norm_mean[:, None]) / (norm_std[:, None] + NORM_EPS)
    else:
        feat = (feat - feat.mean(1, keepdim=True)) / (feat.std(1, keepdim=True) + NORM_EPS)
    return feat.T.contiguous()                                # (T, 80)


def rel_pos_table(T: int, d: int = D, device=None) -> torch.Tensor:
    """相対位置 (T-1) … -(T-1) の正弦埋め込み (2T-1, d)。行 k の位置は T-1-k。"""
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float32, device=device)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(2 * T - 1, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def conv2d_s2(x: torch.Tensor, wt: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Conv2d(3×3, stride 2, pad 1)。int8 模擬では im2col して lin() に通す (C エンジンと同じ分解)。"""
    if not QUANT["on"]:
        return F.conv2d(x, wt, b, stride=2, padding=1)
    n, c, t, f = x.shape
    cols = F.unfold(x, 3, padding=1, stride=2)                # (n, c*9, L)
    to, fo = (t + 2 - 3) // 2 + 1, (f + 2 - 3) // 2 + 1
    y = lin(cols.transpose(1, 2), wt.reshape(wt.shape[0], -1), b)   # (n, L, Cout)
    return y.transpose(1, 2).reshape(n, wt.shape[0], to, fo)


def subsample(feat: torch.Tensor, w: dict) -> torch.Tensor:
    """striding ×4: Conv2d(1→C,3,s2,p1)+ReLU, Conv2d(C→C,3,s2,p1)+ReLU, (T',C,F') を C 優先で平らにして Linear。"""
    x = feat[None, None]                                      # (1,1,T,80)
    x = F.relu(conv2d_s2(x, w["sub.conv1.weight"], w["sub.conv1.bias"]))
    x = F.relu(conv2d_s2(x, w["sub.conv2.weight"], w["sub.conv2.bias"]))
    b, c, t, f = x.shape
    x = x.transpose(1, 2).reshape(b, t, c * f)
    return lin(x, w["sub.out.weight"], w["sub.out.bias"])[0]   # (T', D)


def attention(x: torch.Tensor, pe: torch.Tensor, w: dict, p: str) -> torch.Tensor:
    T = x.shape[0]; D = x.shape[1]; H = w[p + "att.bias_u"].shape[0]; dk = D // H     # 幅・ヘッド数は重みから (d=256 のモデルも検算できる)
    q = lin(x, w[p + "att.q.weight"], w[p + "att.q.bias"]).view(T, H, dk)
    k = lin(x, w[p + "att.k.weight"], w[p + "att.k.bias"]).view(T, H, dk)
    v = lin(x, w[p + "att.v.weight"], w[p + "att.v.bias"]).view(T, H, dk)
    pos = lin(pe, w[p + "att.pos.weight"]).view(2 * T - 1, H, dk)        # 行 r ↔ 相対位置 T-1-r
    qu = (q + w[p + "att.bias_u"]).transpose(0, 1)             # (H,T,dk)
    qv = (q + w[p + "att.bias_v"]).transpose(0, 1)
    ac = qu @ k.transpose(0, 1).transpose(1, 2)                # (H,T,T)
    bd_all = qv @ pos.transpose(0, 1).transpose(1, 2)          # (H,T,2T-1)
    # 問い i・鍵 j の相対位置は i-j → 行 r = T-1-(i-j) = T-1-i+j
    ar = torch.arange(T, device=x.device)
    idx = (ar[:, None] * -1 + ar[None, :] + (T - 1))
    bd = torch.gather(bd_all, 2, idx[None].expand(H, T, T))
    att = torch.softmax((ac + bd) / math.sqrt(dk), dim=-1)
    out = (att @ v.transpose(0, 1)).transpose(0, 1).reshape(T, D)
    return lin(out, w[p + "att.out.weight"], w[p + "att.out.bias"])


def conv_module(x: torch.Tensor, w: dict, p: str) -> torch.Tensor:
    y = lin(x, w[p + "conv.pw1.weight"], w[p + "conv.pw1.bias"])      # (T, 2D)
    y = F.glu(y, dim=-1)
    y = F.conv1d(y.T[None], w[p + "conv.dw.weight"], w[p + "conv.dw.bias"], padding=K // 2, groups=D)[0].T
    y = F.silu(y)
    return lin(y, w[p + "conv.pw2.weight"], w[p + "conv.pw2.bias"])


def ff(x: torch.Tensor, w: dict, p: str) -> torch.Tensor:
    return lin(F.silu(lin(x, w[p + "1.weight"], w[p + "1.bias"])), w[p + "2.weight"], w[p + "2.bias"])


def ln(x, w, p):
    return F.layer_norm(x, (x.shape[-1],), w[p + ".weight"], w[p + ".bias"], eps=1e-5)


def layer(x: torch.Tensor, pe: torch.Tensor, w: dict, i: int) -> torch.Tensor:
    p = f"layers.{i}."
    x = x + 0.5 * ff(ln(x, w, p + "norm_ff1"), w, p + "ff1.")
    x = x + attention(ln(x, w, p + "norm_att"), pe, w, p)
    x = x + conv_module(ln(x, w, p + "norm_conv"), w, p)
    x = x + 0.5 * ff(ln(x, w, p + "norm_ff2"), w, p + "ff2.")
    return ln(x, w, p + "norm_out")


def n_layers_of(w: dict) -> int:
    return 1 + max(int(k.split(".")[1]) for k in w if k.startswith("layers."))


def encode(feat: torch.Tensor, w: dict, n_layers: int | None = None) -> torch.Tensor:
    n_layers = n_layers_of(w) if n_layers is None else n_layers
    d = w["sub.out.weight"].shape[0]
    x = subsample(feat, w) * math.sqrt(d)                      # xscaling
    pe = rel_pos_table(x.shape[0], d, device=x.device)
    for i in range(n_layers):
        x = layer(x, pe, w, i)
    return x


def logits(wav: torch.Tensor, w: dict) -> torch.Tensor:
    """(n,) → (T', vocab+1) の log-softmax。blank は最後の列。"""
    feat = log_mel(wav, w.get("mel.fb"), w.get("mel.window"), w.get("mel.norm_mean"), w.get("mel.norm_std"))
    x = encode(feat, w)
    return torch.log_softmax(lin(x, w["head.weight"], w["head.bias"]), dim=-1)


def greedy(lp: torch.Tensor, blank: int) -> list[int]:
    ids = lp.argmax(-1).tolist()
    out, prev = [], blank
    for t in ids:
        if t != blank and t != prev:
            out.append(t)
        prev = t
    return out
