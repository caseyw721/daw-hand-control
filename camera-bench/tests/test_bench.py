"""Host tests for the camera bench logic (no camera needed): python3 tests/test_bench.py"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import (AbsoluteMap, Lock, Marker, OneEuro, Pinch, Presence, RelativeMap, jitter_rms, lead_gain,  # noqa: E402
                   pinch_dist, pinch_ratio, point_of, rect_from_sweep, still_windows, INDEX_TIP, THUMB_TIP,
                   WRIST, MIDDLE_MCP, POINTS)

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

# --- lead: exact for constant velocity, nothing at rest, gated by speed ---
f = OneEuro(1.0, 0.02)
for i in range(200):
    f(1000.0 * i / 60, i / 60)   # 1000 px/s, long enough for the velocity estimate to settle
check(abs(f.lead(0.04) - (f.x + 40.0)) < 0.5, f"40 ms lead at 1000 px/s is 40 px ahead ({f.lead(0.04) - f.x:.2f})")
f = OneEuro(1.0, 0.02)
for i in range(60):
    f(100.0, i / 60)
check(f.lead(0.04) == f.x, "no lead for a hand at rest")
check(lead_gain(0) == 0 and lead_gain(150) == 0 and lead_gain(325) == 0.5 and lead_gain(500) == 1 and lead_gain(5000) == 1,
      "lead gain ramps 0 -> 1 between 150 and 500 px/s")
check(lead_gain(300, 100, 500) == 0.5, "lead gain takes a custom gate")

# --- stillness: judged on the median-smoothed raw point over half-second windows ---
random.seed(2)
rest1 = [(500 + random.gauss(0, 5), 500 + random.gauss(0, 5)) for _ in range(15)]
move = [(500 + 1000 / 30 * i, 500.0) for i in range(1, 16)]
rest2 = [(1000 + random.gauss(0, 5), 500 + random.gauss(0, 5)) for _ in range(15)]
check(still_windows(rest1 + move + rest2) == [0, 30], f"still windows are the two rests, not the move ({still_windows(rest1 + move + rest2)})")
spiky = rest1[:7] + [(700.0, 700.0)] + rest1[8:]
check(still_windows(spiky) == [0], "a one-frame landmark glitch does not break a rest (median)")
check(jitter_rms([(1.0, 1.0)] * 15) == 0 and abs(jitter_rms([(0.0, 0.0), (2.0, 0.0)]) - 1.0) < 1e-9, "jitter rms about the mean")

# --- presence: short losses coast, a long one is lost exactly once, hold 0 is the old behaviour ---
pr = Presence(hold_s=0.17)
seq = [(True, 0.0), (False, 0.033), (False, 0.066), (True, 0.1), (False, 0.133), (False, 0.2), (False, 0.3),
       (False, 0.333), (True, 0.4)]
states = [pr.update(h, t) for h, t in seq]
check(states == ["tracked", "held", "held", "tracked", "held", "held", "lost", "gone", "tracked"], f"presence states {states}")
check(Presence(0.17).update(False, 0.0) == "gone", "no hand before the first sighting is nothing to lose")
pr = Presence(hold_s=0)
check(pr.update(True, 0) == "tracked" and pr.update(False, 0.033) == "lost", "hold 0 loses the hand on the first missing frame")

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
p = Pinch(on=0.035, off=0.055, min_frames=2, off_frames=3)
for d in (0.02, 0.02, 0.07, 0.07):  # pinched, then two of the three release frames counted
    p.update(d)
p.reset()
check(not p.closed and p.update(0.02) is None, "reset forgets the half-counted frames: one frame cannot pinch")

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

# --- marker: a 160 px square centred on Quartz (top-left) point -> AppKit (bottom-left) origin ---
check(Marker.origin(100, 50, 160, 1000) == (20, 870), "marker origin flips y and centres the square")
check(Marker(enabled=False).win is None, "disabled marker creates no window")

# --- anchor points: the thumb joint does not move when the index finger curls ---
lm = [(0.5, 0.5, 0.0)] * 21
lm = list(lm)
lm[3] = (0.40, 0.60, 0.0)          # thumb IP
lm[8] = (0.55, 0.30, 0.0)          # index tip
lm[2], lm[5] = (0.38, 0.70, 0.0), (0.50, 0.55, 0.0)
check(point_of(lm, "thumb") == (0.40, 0.60), "thumb = the thumb's middle joint")
check(point_of(lm, "tip") == (0.55, 0.30), "tip = the index fingertip")
check(point_of(lm, "web") == (0.44, 0.625), "web = between thumb base and index knuckle")
curled = list(lm)
curled[8] = (0.45, 0.55, 0.0)      # the index curls toward the thumb
check(point_of(curled, "thumb") == point_of(lm, "thumb"), "curling the index does not move the thumb anchor")
check(set(POINTS) == {"tip", "thumb", "thumbtip", "web", "palm"}, "all anchor names are known")

# --- hover lock: rest locks, small motion is ignored, a real move breaks out, a grab never breaks ---
k = Lock(dwell=0.2, still=120, breakout=50)
t = 0.0
out = None
for i in range(10):                 # resting at (500, 400) for 0.3 s
    out = k.update((500 + (i % 2) * 3, 400), t, 20.0)
    t += 1 / 30
check(k.at is not None, "resting 0.2 s locks")
held_at = k.at
out = k.update((530, 420), t, 300.0)
check(out == held_at, "a 36 px wobble stays locked")
out = k.update((600, 400), t + 0.03, 900.0)
check(k.at is None and out == (600, 400), "a 100 px move breaks out and the cursor follows")
k = Lock(dwell=0.2, still=120, breakout=50)
out = k.update((300, 300), 0.0, 400.0, held=True, grab_at=(290, 305))
check(out == (290, 305), "a grab locks where the pointer was just before the pinch")
out = k.update((500, 500), 0.1, 900.0, held=True)
check(out == (290, 305), "while the grab is held nothing breaks the lock")
out = k.update((500, 500), 0.2, 900.0, held=False)
check(out == (500, 500) and k.at is None, "after the grab a far hand breaks out")
k = Lock(dwell=0, still=120, breakout=50)
for i in range(20):
    out = k.update((100, 100), i / 30, 10.0)
check(k.at is None, "dwell 0 turns hover locking off")

print("camera-bench: %d failures" % fails)
sys.exit(1 if fails else 0)
