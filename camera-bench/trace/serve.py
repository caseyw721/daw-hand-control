#!/usr/bin/env python3
"""Serves the cursor trace test and the live pointer diagnostic, and logs trace results Claude can read.

    python3 trace/serve.py [--port 8765] [--diag-port 8766]
    http://localhost:8765/            the trace test (hold, lines, diagonals, box)
    http://localhost:8765/diag.html   the live 3-D-to-2-D diagnostic (bench.py --diag 8766 feeds it)

Trace results: ~/Library/Logs/DAW Control/trace-tests.jsonl (DAWCONTROL_LOG_DIR moves it), one JSON object
per task: task, rms (px off the path), max, jitter (px, RMS of the track's second difference), secs, speed,
samples, rate (cursor moves per second), when, screen.

Diagnostic: the tracker sends one UDP JSON packet per frame to --diag-port; the page subscribes to
/diag/stream (server-sent events) and gets the newest packet at up to 60 per second.
"""
import argparse
import http.server
import json
import os
import socket
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get("DAWCONTROL_LOG_DIR") or Path.home() / "Library/Logs/DAW Control") / "trace-tests.jsonl"

latest = {"seq": 0, "data": b"{}"}
changed = threading.Condition()


def udp_listener(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    while True:
        data, _ = s.recvfrom(65536)
        with changed:
            latest["seq"] += 1
            latest["data"] = data
            changed.notify_all()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def do_GET(self):
        if self.path.startswith("/diag/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen = latest["seq"]
            try:
                while True:
                    with changed:
                        changed.wait_for(lambda: latest["seq"] != seen, timeout=2.0)
                        seq, data = latest["seq"], latest["data"]
                    if seq == seen:
                        self.wfile.write(b": idle\n\n")          # keeps the connection alive
                    else:
                        seen = seq
                        self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1 / 60)
            except (BrokenPipeError, ConnectionResetError):
                return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/result":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            rec = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400)
            return
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"result: {rec.get('task')}: off path {rec.get('rms')} px rms, max {rec.get('max')}, jitter {rec.get('jitter')}", flush=True)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--diag-port", type=int, default=8766)
    args = ap.parse_args()
    threading.Thread(target=udp_listener, args=(args.diag_port,), daemon=True).start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    print(f"trace test on http://localhost:{args.port}/  diagnostic on http://localhost:{args.port}/diag.html  "
          f"(UDP {args.diag_port})  results -> {LOG}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
