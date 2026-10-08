#!/usr/bin/env python3
"""サーバー側から pc_agent.py (ssh -R で localhost:4010 に来ている) を叩く。

    python tools/device.py ports
    python tools/device.py flash build_esp32p4/komimi_p4_app.bin --port COM7 --chip esp32p4 --baud 921600
    python tools/device.py monitor --port COM7 --seconds 40 [--no-reset] [--raw]
"""
import argparse
import re
import sys
import urllib.request

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def call(url, data=None, timeout=1500):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ports", "flash", "monitor", "reset"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--base", default="http://127.0.0.1:4010")
    ap.add_argument("--port", default="COM7"); ap.add_argument("--chip", default="esp32p4"); ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--offset", default="0x0"); ap.add_argument("--seconds", type=float, default=30); ap.add_argument("--no-reset", action="store_true"); ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    if a.cmd == "ports":
        code, text = call(f"{a.base}/ports")
    elif a.cmd == "flash":
        data = open(a.file, "rb").read()
        code, text = call(f"{a.base}/flash?port={a.port}&chip={a.chip}&baud={a.baud}&offset={a.offset}", data)
    elif a.cmd == "monitor":
        code, text = call(f"{a.base}/monitor?port={a.port}&seconds={a.seconds}&baud=115200&reset={0 if a.no_reset else 1}")
    else:
        code, text = call(f"{a.base}/reset?port={a.port}")
    if not a.raw:
        text = ANSI.sub("", text)
    print(text)
    return 0 if code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
