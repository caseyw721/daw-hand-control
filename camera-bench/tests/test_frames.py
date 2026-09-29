"""Host tests for the frame writer (no camera, no MediaPipe): python3 tests/test_frames.py"""
import json
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frames import FrameWriter, frame_record, shared_clock  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    if not cond:
        fails += 1
        print("FAIL:", msg)


def fake_result(n_hands=1, label="Right"):
    """Looks enough like a MediaPipe HandLandmarkerResult: 21 points with .x/.y/.z per hand."""
    def pts(scale):
        return [SimpleNamespace(x=0.1 * i * scale, y=0.02 * i, z=-0.001 * i) for i in range(21)]
    return SimpleNamespace(
        hand_landmarks=[pts(1.0) for _ in range(n_hands)],
        hand_world_landmarks=[pts(0.01) for _ in range(n_hands)],
        handedness=[[SimpleNamespace(category_name=label, score=0.9876)] for _ in range(n_hands)],
    )


# --- the record ---
r = frame_record(5, None, "gone", 10.0, 10.012, 1280, 720)
check(r["v"] == 1 and r["seq"] == 5 and r["state"] == "gone" and r["hands"] == [], "no-hand frame is an empty hands list")
check(abs(r["det_ms"] - 12.0) < 1e-6, f"det_ms is t - t_read in ms ({r['det_ms']})")
check(r["w"] == 1280 and r["h"] == 720 and r["mirrored"] is False and r["t_capture"] is None, "frame carries size, mirror flag, null capture time")

r = frame_record(6, fake_result(), "tracked", 10.0, 10.01, 1280, 720)
h = r["hands"][0]
check(len(r["hands"]) == 1 and len(h["img"]) == 21 and len(h["world"]) == 21, "one hand, 21 image and 21 world points")
check(h["label"] == "Right" and h["score"] == 0.988, f"handedness label and rounded score ({h['label']}, {h['score']})")
check(h["img"][3] == [round(0.3, 4), 0.06, -0.003], f"points are [x, y, z] rounded to 4 places ({h['img'][3]})")
check(len(json.dumps(r, separators=(",", ":"))) < 3000, "a one-hand frame is under 3 KB")

r = frame_record(7, fake_result(2, "Left"), "tracked", 0.0, 0.01, 640, 480)
check(len(r["hands"]) == 2 and all(x["label"] == "Left" for x in r["hands"]), "two hands are both recorded")

partial = SimpleNamespace(hand_landmarks=fake_result().hand_landmarks, hand_world_landmarks=[], handedness=[])
r = frame_record(8, partial, "tracked", 0.0, 0.01, 640, 480)
check(r["hands"][0]["world"] is None and r["hands"][0]["label"] is None, "missing world points and handedness are null, not an error")

# --- the writer: every frame lands, in order, one JSON line each ---
d = tempfile.mkdtemp()
path = os.path.join(d, "frames.jsonl")
w = FrameWriter(path, 1280, 720)
for i in range(5):
    w.write(fake_result() if i % 2 == 0 else None, "tracked" if i % 2 == 0 else "lost", 1.0 + i / 30, 1.01 + i / 30)
written, dropped = w.close()
lines = open(path).read().splitlines()
check(len(lines) == 5 and written == 5 and dropped == 0, f"5 frames written ({len(lines)} lines, written {written}, dropped {dropped})")
recs = [json.loads(l) for l in lines]
check([x["seq"] for x in recs] == [1, 2, 3, 4, 5], "seq counts from 1 in order")
check(abs(recs[0]["t"] - (1.01 + w.offset)) < 1e-5 and abs(recs[0]["t_read"] - (1.0 + w.offset)) < 1e-5,
      "stamps are perf_counter values moved onto the shared clock")
check(abs((shared_clock() - time.perf_counter()) - w.offset) < 0.01, f"the offset is the live clock difference ({w.offset:.3f})")
check(abs(recs[0]["det_ms"] - 10.0) < 1e-6, "det_ms is unchanged by the offset")
check([len(x["hands"]) for x in recs] == [1, 0, 1, 0, 1], "hand and no-hand frames alternate as written")
check(recs[1]["state"] == "lost", "state is passed through")

# --- the writer never blocks: a stuck output drops frames and counts them ---
path2 = os.path.join(d, "stuck.jsonl")
w = FrameWriter(path2, 1280, 720, maxsize=2)
gate = threading.Event()
real = w.f


class Stuck:
    def write(self, s):
        gate.wait()
        real.write(s)

    def flush(self):
        real.flush()

    def close(self):
        real.close()


w.f = Stuck()
t0 = time.perf_counter()
for i in range(20):
    w.write(None, "gone", 0.0, 0.01)
elapsed = time.perf_counter() - t0
check(elapsed < 0.05, f"20 writes into a stuck output return at once ({elapsed * 1000:.1f} ms)")
check(w.dropped >= 15, f"frames beyond the queue are dropped and counted ({w.dropped})")
gate.set()
written, dropped = w.close()
check(written + dropped == 20, f"written + dropped accounts for every frame ({written} + {dropped})")

print(f"test_frames: {fails} failures")
sys.exit(1 if fails else 0)
