"""16 kHz モノラルの wav を C のヘッダ (int16 配列) にする。esp32/main/test_audio.h (ベンチの実音声) を作るのに使う。
  python tools/wav2header.py in.wav esp32/main/test_audio.h --name test_audio --comment "..."
"""
import argparse, numpy as np, soundfile as sf
ap = argparse.ArgumentParser(); ap.add_argument("wav"); ap.add_argument("out"); ap.add_argument("--name", default="test_audio"); ap.add_argument("--comment", default="")
a = ap.parse_args()
x, sr = sf.read(a.wav, dtype="int16"); assert sr == 16000, f"16 kHz expected, got {sr}"
if x.ndim > 1: x = x[:, 0]
N = a.name.upper()
with open(a.out, "w") as f:
    f.write(f"/* {a.comment or 'wav2header.py で生成'} (16 kHz PCM16、{len(x)/sr:.2f} s) */\n#include <stdint.h>\n#define {N}_N {len(x)}\nstatic const int16_t {a.name}[{N}_N] = {{\n")
    for i in range(0, len(x), 16): f.write("    " + ", ".join(str(int(v)) for v in x[i:i+16]) + ",\n")
    f.write("};\n")
print(f"{a.out}: {len(x)} samples, {len(x)/sr:.2f} s")
