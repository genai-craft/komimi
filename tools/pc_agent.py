#!/usr/bin/env python3
"""手元 PC で動かす端末エージェント。サーバーからの指示で esptool の書き込み・リセット・シリアル読みを代行する。

    pip install esptool pyserial
    python pc_agent.py --server <user>@<server>             # ssh -R でサーバーの localhost:4010 を自分に向ける (環境変数 KOMIMI_SERVER でも可)
    python pc_agent.py --no-tunnel                              # 自前でトンネルを張る場合

HTTP (127.0.0.1:4010、ssh -R 越しにサーバーから):
    GET  /ports                                   → COM ポート一覧
    POST /flash?port=COM7&chip=esp32p4&baud=921600&offset=0x0   (body = 書き込むバイナリ)
    GET  /monitor?port=COM7&seconds=30&baud=115200&reset=1       → リセットしてから seconds 秒ぶんのシリアル出力
    GET  /reset?port=COM7

RFC2217 を使わない理由: ESP32 の USB-Serial/JTAG はリセットのたびに COM が再列挙され、遠隔のシリアルサーバーの
握りが死ぬ。esptool をこの PC 上で動かせば、再列挙は esptool/pyserial が面倒を見る。
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import serial
from serial.tools import list_ports


def open_serial(port, baud, deadline):
    """再列挙中は数秒開けないので、deadline まで繰り返す。"""
    last = None
    while time.time() < deadline:
        try:
            return serial.Serial(port, baud, timeout=0.2)
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(0.3)
    raise RuntimeError(f"cannot open {port}: {last}")


def reset_device(ser):
    """USB-Serial/JTAG のリセット: RTS を立てて DTR を落とすと EN が下がる。戻すと起動する。"""
    try:
        ser.dtr = False; ser.rts = True; time.sleep(0.1); ser.rts = False; time.sleep(0.05)
    except Exception:  # noqa: BLE001
        pass


def monitor(port, baud, seconds, do_reset):
    out = []
    deadline = time.time() + seconds + 10
    ser = open_serial(port, baud, time.time() + 10)
    if do_reset:
        reset_device(ser)
    end = time.time() + seconds
    while time.time() < end:
        try:
            data = ser.read(4096)
            if data:
                out.append(data.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001  (再列挙で握りが死んだら開け直す)
            out.append(f"\n[agent] serial error: {e}; reopening\n")
            try:
                ser.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
            try:
                ser = open_serial(port, baud, deadline)
            except Exception as e2:  # noqa: BLE001
                out.append(f"[agent] reopen failed: {e2}\n"); break
    try:
        ser.close()
    except Exception:  # noqa: BLE001
        pass
    return "".join(out)


def flash(port, chip, baud, offset, data):
    fd, path = tempfile.mkstemp(suffix=".bin"); os.write(fd, data); os.close(fd)
    try:
        cmd = [sys.executable, "-m", "esptool", "--chip", chip, "-p", port, "-b", str(baud), "--before", "default_reset", "--after", "hard_reset",
               "write_flash", offset, path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    finally:
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, text):
        body = text.encode("utf-8", errors="replace")
        self.send_response(code); self.send_header("Content-Type", "text/plain; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/ports":
                self._send(200, "\n".join(f"{p.device}\t{p.description}" for p in list_ports.comports()) + "\n")
            elif u.path == "/monitor":
                text = monitor(q.get("port", ["COM7"])[0], int(q.get("baud", ["115200"])[0]), float(q.get("seconds", ["20"])[0]), q.get("reset", ["1"])[0] == "1")
                self._send(200, text)
            elif u.path == "/reset":
                ser = open_serial(q.get("port", ["COM7"])[0], 115200, time.time() + 10); reset_device(ser); ser.close(); self._send(200, "reset\n")
            else:
                self._send(404, "unknown\n")
        except Exception as e:  # noqa: BLE001
            self._send(500, f"error: {e}\n")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        n = int(self.headers.get("Content-Length", "0")); data = self.rfile.read(n)
        try:
            if u.path == "/flash":
                rc, text = flash(q.get("port", ["COM7"])[0], q.get("chip", ["esp32p4"])[0], int(q.get("baud", ["921600"])[0]), q.get("offset", ["0x0"])[0], data)
                self._send(200 if rc == 0 else 500, f"[agent] esptool exit {rc}, {len(data)} bytes\n" + text)
            else:
                self._send(404, "unknown\n")
        except Exception as e:  # noqa: BLE001
            self._send(500, f"error: {e}\n")

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=4010)
    ap.add_argument("--server", default=os.environ.get("KOMIMI_SERVER"), help="ssh -R を張る先 (user@host)。省略時は環境変数 KOMIMI_SERVER")
    ap.add_argument("--no-tunnel", action="store_true")
    a = ap.parse_args()
    tunnel = None
    if not a.no_tunnel:
        print(f"[agent] opening tunnel: ssh -N -R {a.listen}:127.0.0.1:{a.listen} {a.server}  (パスワードを聞かれたら入力)")
        tunnel = subprocess.Popen(["ssh", "-N", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes", "-R", f"{a.listen}:127.0.0.1:{a.listen}", a.server])
    print(f"[agent] ports: " + ", ".join(p.device for p in list_ports.comports()))
    print(f"[agent] listening on 127.0.0.1:{a.listen}  (Ctrl-C で終了)")
    srv = HTTPServer(("127.0.0.1", a.listen), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
