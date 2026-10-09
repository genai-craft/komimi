"""C エンジン (csrc/libkomimi.so) の Python 束縛 (ctypes)。torch は要らない。

    from komimi.engine import Engine
    e = Engine("models/ja_v12a_i8_c32.kmm")
    lp = e.logprobs(wav)            # (T', 1025) の log-softmax。T' は 40 ms ごと、blank = 1024
    ids = e.greedy(lp)              # CTC の greedy (blank と重複を除いた token 列)
    e.decode(ids)                   # "ソシテ..." (カタカナ)
    e.tokenize("ソシテ")             # 候補カナ列 → token 列 (unigram の Viterbi、SentencePiece と 99% 一致)
    e.ctc_score(lp, [ids1, ids2])   # 各 token 列の log p(列 | 音声) (CTC forward、全アラインメントの和)

共有ライブラリは `make -C csrc libkomimi.so` で作る (無ければ初回に make を試みる)。KOMIMI_LIB で場所を指定できる。
ctypes の呼び出しは GIL を外すので、複数スレッドから同時に logprobs() を呼べる (モデルは読み取り専用、作業メモリは呼び出しごと)。
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "models" / "ja1024_vocab.json"
_LIB = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    path = Path(os.environ.get("KOMIMI_LIB", ROOT / "csrc" / "libkomimi.so"))
    if not path.exists() and "KOMIMI_LIB" not in os.environ:
        subprocess.run(["make", "-C", str(ROOT / "csrc"), "libkomimi.so"], check=True, capture_output=True)
    lib = ctypes.CDLL(str(path))
    lib.km_load.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.km_load.restype = ctypes.c_int
    lib.km_recognize_ex.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int]
    lib.km_recognize_ex.restype = ctypes.c_int
    _LIB = lib
    return lib


class Tokenizer:
    """unigram の Viterbi 分割 (piece の対数確率の和が最大の分割)。先頭に dummy prefix「▁」を付ける (学習と同じ)。"""

    def __init__(self, vocab_path: str | Path = VOCAB):
        v = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.pieces: list[str] = v["pieces"]
        self.scores = np.asarray(v["scores"], dtype=np.float64)
        self.prefix: str = v.get("dummy_prefix", "▁")
        self.unk = int(v.get("unk_id", 0))
        self.blank = int(v.get("blank_id", len(self.pieces)))
        self.index = {p: i for i, p in enumerate(self.pieces) if i != self.unk}
        self.maxlen = max(len(p) for p in self.pieces)

    @lru_cache(maxsize=200_000)
    def _encode(self, text: str) -> tuple[int, ...]:
        s = self.prefix + text
        n = len(s)
        best = np.full(n + 1, -np.inf); best[0] = 0.0
        back: list[tuple[int, int]] = [(0, 0)] * (n + 1)
        for i in range(1, n + 1):
            for L in range(1, min(self.maxlen, i) + 1):
                pid = self.index.get(s[i - L:i])
                if pid is None or best[i - L] == -np.inf:
                    continue
                c = best[i - L] + self.scores[pid]
                if c > best[i] + 1e-12:
                    best[i] = c; back[i] = (i - L, pid)
            if best[i] == -np.inf:            # 語彙に無い文字は <unk> 1 つで飛ばす
                best[i] = best[i - 1] - 20.0; back[i] = (i - 1, self.unk)
        out = []; i = n
        while i > 0:
            j, pid = back[i]; out.append(pid); i = j
        return tuple(reversed(out))

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))

    def decode(self, ids) -> str:
        return "".join(self.pieces[i] for i in ids if 0 <= i < len(self.pieces) and i != self.unk).replace(self.prefix, "")


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


NEG = -1e30


def ctc_forward(lp: np.ndarray, seqs: list[list[int]], blank: int) -> np.ndarray:
    """各 token 列の log p(列 | 音声) (CTC、blank を挟む全アラインメントの和)。列はまとめて (K, 2L+1) の格子で回す。
    フレーム数が足りない列は -1e30 付近になる。"""
    K = len(seqs)
    if K == 0:
        return np.zeros(0)
    T = lp.shape[0]
    L = max(1, max(len(s) for s in seqs))
    S = 2 * L + 1
    lab = np.full((K, S), blank, dtype=np.int64)
    valid = np.zeros((K, S), dtype=bool)
    skip = np.zeros((K, S), dtype=bool)              # s-2 からの飛び越し (異なる非 blank 同士)
    ends = np.zeros(K, dtype=np.int64)
    for k, s in enumerate(seqs):
        n = len(s)
        lab[k, 1:2 * n:2] = s
        valid[k, :2 * n + 1] = True
        ends[k] = 2 * n
        for j in range(1, n):
            if s[j] != s[j - 1]:
                skip[k, 2 * j + 1] = True
    emit = lp[:, lab]                                 # (T, K, S)
    a = np.full((K, S), NEG)
    a[:, 0] = emit[0, :, 0]
    has = ends > 0
    a[has, 1] = emit[0, has, 1]
    a[~valid] = NEG
    for t in range(1, T):
        a1 = np.concatenate([np.full((K, 1), NEG), a[:, :-1]], 1)
        a2 = np.concatenate([np.full((K, 2), NEG), a[:, :-2]], 1)
        a2 = np.where(skip, a2, NEG)
        m = np.maximum(np.maximum(a, a1), a2)
        m_safe = np.where(m <= NEG / 2, 0.0, m)
        s_ = np.exp(a - m_safe) + np.exp(a1 - m_safe) + np.exp(a2 - m_safe)
        a = np.where(m <= NEG / 2, NEG, m_safe + np.log(np.maximum(s_, 1e-300))) + emit[t]
        a[~valid] = NEG
    idx = np.arange(K)
    last = a[idx, ends]
    prev = np.where(ends > 0, a[idx, np.maximum(ends - 1, 0)], NEG)
    m = np.maximum(last, prev)
    return np.where(m <= NEG / 2, NEG, m + np.log(np.exp(last - m) + np.exp(prev - m)))


class Engine:
    def __init__(self, kmm_path: str | Path, vocab_path: str | Path = VOCAB):
        self.path = str(kmm_path)
        self._buf = ctypes.create_string_buffer(Path(kmm_path).read_bytes())   # km_model は buf を指したまま使う
        self._m = ctypes.create_string_buffer(1 << 16)                          # km_model の実体 (sizeof は 30 KB 程度)
        lib = _lib()
        if lib.km_load(self._m, self._buf, len(self._buf) - 1) != 0:
            raise ValueError(f"km_load failed: {kmm_path}")
        self.tok = Tokenizer(vocab_path)
        self.vocab = len(self.tok.pieces) + 1
        self.blank = self.vocab - 1
        self.hop = 160

    def logits(self, wav: np.ndarray) -> np.ndarray:
        """全文脈の CTC logits (T', vocab)。wav は 16 kHz float。"""
        x = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
        max_frames = len(x) // (self.hop * 4) + 4
        out = np.zeros((max_frames, self.vocab), dtype=np.float32)
        ids = np.zeros(4096, dtype=np.int32)
        nf = ctypes.c_int(0)
        rc = _lib().km_recognize_ex(self._m, x.ctypes.data, len(x), ids.ctypes.data, len(ids), ctypes.byref(nf), out.ctypes.data, max_frames)
        if rc < 0:
            raise RuntimeError("km_recognize_ex failed")
        return out[: min(nf.value, max_frames)]

    def logprobs(self, wav: np.ndarray) -> np.ndarray:
        return log_softmax(self.logits(wav))

    def greedy(self, lp: np.ndarray) -> list[int]:
        best = lp.argmax(-1)
        out, prev = [], self.blank
        for b in best.tolist():
            if b != self.blank and b != prev:
                out.append(b)
            prev = b
        return out

    def decode(self, ids) -> str:
        return self.tok.decode(ids)

    def tokenize(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def ctc_score(self, lp: np.ndarray, seqs: list[list[int]]) -> np.ndarray:
        return ctc_forward(lp, seqs, self.blank)
