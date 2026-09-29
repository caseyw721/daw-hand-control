"""Host tests for the camera bench logic (no camera needed): python3 tests/test_bench.py"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import (AbsoluteMap, OneEuro, Pinch, RelativeMap, pinch_dist, pinch_ratio, point_of,  # noqa: E402
                   rect_from_sweep, INDEX_TIP, THUMB_TIP, WRIST, MIDDLE_MCP)

fails = 0


def check(cond, msg):
    global fails
    if not cond:
        fails += 1
        print("FAIL:", msg)


def close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


# --- 1€ filter: passes a constant, smooths noise when slow, follows fast moves ---
f = OneEuro(1.0, 0.02)
for i in range(100):
    y = f(100.0, i / 60)
check(abs(y - 100) < 1e-9, "constant input passes unchanged")
import random  # noqa: E402
random.seed(1)
f = OneEuro(1.0, 0.02)
out = [f(100 + random.gauss(0, 3), i / 60) for i in range(300)]
tail = out[120:]
rms_in, rms_out = 3.0, math.sqrt(sum((v - 100) ** 2 for v in tail) / len(tail))
check(rms_out < rms_in / 2, f"noise at rest is reduced ({rms_out:.2f} vs {rms_in})")
f = OneEuro(1.0, 0.02)
lag = None
for i in range(60):
    x = 0 if i < 10 else 500.0  # step at frame 10
    y = f(x, i / 60)
    if lag is None and x == 500 and y > 450:
        lag = i - 10
check(lag is not None and lag <= 3, f"fast step is followed within 3 frames (got {lag})")

# --- pinch hysteresis ---
p = Pinch(on=0.30, off=0.45, min_frames=2)
seq = [0.8, 0.8, 0.25, 0.25, 0.35, 0.35, 0.5, 0.5, 0.8]
events = [p.update(r) for r in seq]
check(events == [None, None, None, "down", None, None, None, "up", None], f"pinch events {events}")
p = Pinch(min_frames=2)
check(p.update(0.1) is None and p.update(0.9) is None and not p.closed, "one noisy frame never pinches")

# --- pinch from world landmarks: metres, all three axes, slower release than press ---
wlm = [(0.0, 0.0, 0.0)] * 21
wlm[THUMB_TIP] = (0.01, 0.02, 0.00)
wlm[INDEX_TIP] = (0.01, 0.02, 0.03)     # 3 cm apart along z: a 2-D view would call this touching
check(abs(pinch_dist(wlm) - 0.03) < 1e-9, "pinch distance uses all three world axes")
p = Pinch(on=0.035, off=0.055, min_frames=2, off_frames=3)
seq = [0.09, 0.03, 0.03, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06]
events = [p.update(d) for d in seq]
check(events == [None, None, "down", None, None, None, None, None, "up"], f"world pinch events {events}")

# --- pinch ratio and points ---
lm = [(0.0, 0.0, 0.0)] * 21
lm[WRIST] = (0.5, 0.9, 0)
lm[MIDDLE_MCP] = (0.5, 0.5, 0)      # hand size 0.4
lm[THUMB_TIP] = (0.40, 0.40, 0)
lm[INDEX_TIP] = (0.44, 0.40, 0)     # 0.04 apart -> ratio 0.1
check(abs(pinch_ratio(lm) - 0.1) < 1e-9, "pinch ratio normalised by hand size")
check(point_of(lm, "tip") == (0.44, 0.40), "tip point")

# --- absolute map: mirrored, centred 60 % box, clamped ---
m = AbsoluteMap(1000, 500, box=0.6)
check(close(m(0.5, 0.5), (499.5, 249.5)), "centre of frame -> centre of screen")
check(close(m(0.2, 0.2), (999.0, 0.0)), "left edge of box (mirrored) -> right edge of screen")
check(close(m(0.0, 1.0), (999.0, 499.0)), "outside the box clamps")

# --- absolute map with a calibrated box ---
m = AbsoluteMap(1000, 500, rect=(0.5, 0.4, 0.9, 0.8))
check(close(m(0.7, 0.6), (499.5, 249.5)), "centre of the box -> centre of the screen")
check(close(m(0.5, 0.4), (999.0, 0.0)), "box x0 (mirrored) -> right edge, y0 -> top")
check(close(m(0.95, 0.9), (0.0, 499.0)), "outside the calibrated box clamps")

# --- calibration sweep -> box: outliers trimmed, shrunk 5 % per side, no-motion rejected ---
pts = [(0.3 + 0.4 * i / 99, 0.5 + 0.3 * i / 99) for i in range(100)] + [(0.0, 0.0), (1.0, 1.0)]
rect = rect_from_sweep(pts)
check(rect is not None and 0.3 < rect[0] < 0.35 and 0.65 < rect[2] < 0.7
      and 0.5 < rect[1] < 0.55 and 0.75 < rect[3] < 0.8, f"sweep box trims outliers and shrinks a little ({rect})")
check(rect_from_sweep([(0.5, 0.5)] * 50) is None, "a sweep that did not move is rejected")
check(rect_from_sweep([(0.2, 0.2), (0.8, 0.8)]) is None, "too few sweep points are rejected")

# --- relative map: motion moves, loss does not jump ---
r = RelativeMap(1000, 500, gain=2.0, deadzone=0.0)
r(0.5, 0.5)
x, y = r(0.4, 0.5)   # hand moves left in camera = right on screen (mirrored)
check(x > 500 and y == 250, f"relative motion moves the cursor the mirrored way ({x},{y})")
r.lost()
x2, y2 = r(0.9, 0.9)  # reacquired somewhere else: no jump
check((x2, y2) == (x, y), "reacquiring the hand after loss does not jump")

print("camera-bench: %d failures" % fails)
sys.exit(1 if fails else 0)
