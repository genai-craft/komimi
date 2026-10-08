"""komimi の重みファイル `.kmm` の読み書き。

設計 (独自):
  "KMM1" | header (u32 ×16) | n_tensors | tensor table | token table | data
  tensor table の 1 行: name (48 bytes, NUL 詰め) | dtype (u32: 0=f32, 1=int8 行ごと対称, 2=int4 行ごと対称) | ndim | shape[4] | data_off | scale_off
  int8/int4 テンソルは 2 次元 (rows, cols) として扱い、行ごとに scale (f32 rows 個) を持つ。w ≈ q * scale[row]。
  int4 は 1 byte に 2 値 (下位ニブル = 偶数列、上位 = 奇数列、2 の補数 -7..7)。cols は偶数が前提。
  token table: n_tokens | (len u16 | utf-8 bytes) ×n。blank は id = n_tokens (表に無い最後の列)。
データは 64 byte 境界に置く (フラッシュから直接読む前提)。
"""
from __future__ import annotations

import struct

import numpy as np
import torch

MAGIC = b"KMM1"
HDR = ["d", "heads", "ff", "kernel", "n_layers", "n_mel", "vocab", "sub_ch", "flags", "chunk", "left", "sr", "hop", "win", "nfft", "bits"]
FLAG_FIXED_NORM = 1
FLAG_STREAM = 2
INT8_NAMES = ("sub.out.weight", "ff1.1.weight", "ff1.2.weight", "ff2.1.weight", "ff2.2.weight", "att.q.weight", "att.k.weight",
              "att.v.weight", "att.out.weight", "att.pos.weight", "conv.pw1.weight", "conv.pw2.weight", "head.weight",
              "sub.conv1.weight", "sub.conv2.weight")


def quantize_rows(w: torch.Tensor, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """2 次元 (rows, cols) を行ごとに対称 int8 (qmax 127) / int4 (qmax 7) へ。int4 は 2 値を 1 byte に詰める。"""
    qmax = 127 if bits == 8 else 7
    w2 = w.reshape(w.shape[0], -1).double()
    s = w2.abs().amax(1).clamp(min=1e-12) / qmax
    q = torch.round(w2 / s[:, None]).clamp(-qmax, qmax).to(torch.int8).numpy()
    if bits == 4:
        assert q.shape[1] % 2 == 0
        lo = q[:, 0::2].astype(np.uint8) & 0x0F; hi = q[:, 1::2].astype(np.uint8) & 0x0F
        q = (lo | (hi << 4)).astype(np.uint8)
    return q, s.float().numpy()


def unpack_int4(q: np.ndarray) -> np.ndarray:
    lo = (q & 0x0F).astype(np.int8); hi = (q >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo); hi = np.where(hi > 7, hi - 16, hi)
    out = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.int8); out[:, 0::2] = lo; out[:, 1::2] = hi
    return out


def _is_int8(name: str) -> bool:
    return any(name.endswith(n) for n in INT8_NAMES)


def write(path: str, w: dict, tokens: list[str], *, bits: int = 8, chunk: int = 0, left: int = 0) -> int:
    meta = w.get("_meta", {})
    flags = (FLAG_FIXED_NORM if "mel.norm_mean" in w else 0) | (FLAG_STREAM if meta.get("stream") else 0)
    n_layers = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("layers."))
    hdr = dict(d=w["head.weight"].shape[1], heads=w["layers.0.att.bias_u"].shape[0], ff=w["layers.0.ff1.1.weight"].shape[0], kernel=w["layers.0.conv.dw.weight"].shape[-1],
               n_layers=n_layers, n_mel=w["mel.fb"].shape[0], vocab=w["head.weight"].shape[0], sub_ch=w["sub.conv1.weight"].shape[0],
               flags=flags, chunk=chunk, left=left, sr=16000, hop=160, win=w["mel.window"].numel(), nfft=(w["mel.fb"].shape[1] - 1) * 2, bits=bits)
    names = [k for k in w if not k.startswith("_")]
    blobs: list[tuple[bytes, int]] = []           # (bytes, align)
    table = []
    def add(b: bytes) -> int:
        off = sum((len(x) + 63) // 64 * 64 for x, _ in blobs)
        blobs.append((b, 64)); return off
    for name in names:
        t = w[name]
        shape = list(t.shape) + [0] * (4 - t.dim())
        if _is_int8(name) and bits in (4, 8):
            b = bits if (bits == 8 or t.reshape(t.shape[0], -1).shape[1] % 2 == 0) else 8   # 列が奇数 (conv1 の 9) は int8 のまま
            q, s = quantize_rows(t, b)
            doff = add(q.tobytes()); soff = add(s.astype("<f4").tobytes())
            table.append((name, 1 if b == 8 else 2, t.dim(), shape, doff, soff))
        else:
            doff = add(t.contiguous().numpy().astype("<f4").tobytes())
            table.append((name, 0, t.dim(), shape, doff, 0))
    tok = struct.pack("<I", len(tokens)) + b"".join(struct.pack("<H", len(s.encode())) + s.encode() for s in tokens)
    head = MAGIC + struct.pack("<16I", *[int(hdr[k]) for k in HDR]) + struct.pack("<I", len(table))
    rows = b"".join(struct.pack("<48sIIiiiiII", n.encode().ljust(48, b"\0"), dt, nd, *sh, do, so) for n, dt, nd, sh, do, so in table)
    pre = head + rows + tok
    pre += b"\0" * ((64 - len(pre) % 64) % 64)
    data = b""
    for b, _ in blobs:
        data += b + b"\0" * ((64 - len(b) % 64) % 64)
    with open(path, "wb") as f:
        f.write(pre); f.write(data)
    return len(pre) + len(data)


def read(path: str) -> tuple[dict, dict, list[str]]:
    """(header, tensors (float に戻した torch.Tensor), tokens)。C エンジンの読み込みと同じ解釈 (試験用)。"""
    buf = open(path, "rb").read()
    assert buf[:4] == MAGIC
    vals = struct.unpack_from("<16I", buf, 4); hdr = dict(zip(HDR, vals))
    n, = struct.unpack_from("<I", buf, 4 + 64); p = 4 + 64 + 4
    table = []
    for _ in range(n):
        name, dt, nd, *rest = struct.unpack_from("<48sIIiiiiII", buf, p); p += 48 + 4 * 8
        table.append((name.rstrip(b"\0").decode(), dt, nd, rest[:4], rest[4], rest[5]))
    nt, = struct.unpack_from("<I", buf, p); p += 4
    tokens = []
    for _ in range(nt):
        ln, = struct.unpack_from("<H", buf, p); p += 2
        tokens.append(buf[p:p + ln].decode()); p += ln
    data0 = (p + 63) // 64 * 64
    w = {}
    for name, dt, nd, shape, doff, soff in table:
        shape = shape[:nd]; cnt = int(np.prod(shape))
        if dt == 0:
            w[name] = torch.from_numpy(np.frombuffer(buf, "<f4", cnt, data0 + doff).copy()).reshape(shape)
        elif dt == 1:
            q = np.frombuffer(buf, "i1", cnt, data0 + doff).reshape(shape[0], -1).astype(np.float32)
            s = np.frombuffer(buf, "<f4", shape[0], data0 + soff)
            w[name] = torch.from_numpy((q * s[:, None]).copy()).reshape(shape)
        else:
            q = unpack_int4(np.frombuffer(buf, "u1", cnt // 2, data0 + doff).reshape(shape[0], -1)).astype(np.float32)
            s = np.frombuffer(buf, "<f4", shape[0], data0 + soff)
            w[name] = torch.from_numpy((q * s[:, None]).copy()).reshape(shape)
    return hdr, w, tokens
