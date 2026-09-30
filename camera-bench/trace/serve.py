#!/usr/bin/env python3
"""Serves the cursor trace test and appends each finished task's metrics to a log Claude can read.

    python3 trace/serve.py [--port 8765]      then open http://localhost:8765/

Results: ~/Library/Logs/DAW Control/trace-tests.jsonl (DAWCONTROL_LOG_DIR moves it), one JSON object per
task: task, rms (px off the path), max, jitter (px, RMS of the track's second difference), secs, speed,
samples, rate (cursor moves per second), when, screen.
"""
import argparse
import http.server
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get("DAWCONTROL_LOG_DIR") or Path.home() / "Library/Logs/DAW Control") / "trace-tests.jsonl"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

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
    args = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"trace test on http://localhost:{args.port}/  results -> {LOG}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
