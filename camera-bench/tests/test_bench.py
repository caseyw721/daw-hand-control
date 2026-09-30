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
check(point_of(lm, "thumbip") == (0.40, 0.60), "thumbip = the thumb's middle joint")
tx, ty = point_of(lm, "thumb")
check(abs(tx - (0.38 + 0.40 + 0.50) / 3) < 1e-9 and abs(ty - (0.70 + 0.60 + 0.55) / 3) < 1e-9,
      "thumb = mean of thumb base, thumb middle joint and index knuckle")
check(point_of(lm, "tip") == (0.55, 0.30), "tip = the index fingertip")
check(point_of(lm, "web") == (0.44, 0.625), "web = between thumb base and index knuckle")
curled = list(lm)
curled[8] = (0.45, 0.55, 0.0)      # the index curls toward the thumb
check(point_of(curled, "thumb") == point_of(lm, "thumb"), "curling the index does not move the thumb anchor")
check(set(POINTS) == {"tip", "base", "thumb", "thumbip", "thumbtip", "web", "palm"}, "all anchor names are known")
lm2 = [(0.1 * i / 21, 0.5, 0.0) for i in range(21)]
bx, by = point_of(lm2, "base")
check(abs(bx - sum(lm2[k][0] for k in (0, 5, 9, 13, 17)) / 5) < 1e-12 and by == 0.5, "base = wrist + four finger knuckles")
bent = list(lm2)
for k in (6, 7, 8, 10, 11, 12, 3, 4):   # fingers and thumb curl: joints past the knuckles move
    bent[k] = (bent[k][0] + 0.05, bent[k][1] + 0.05, 0.0)
check(point_of(bent, "base") == point_of(lm2, "base"), "bending the fingers or the thumb does not move the base anchor")

# --- despike: a single-frame glitch never reaches the cursor; n=1 passes everything ---
from bench import Despike, Rate  # noqa: E402
d = Despike(3)
outs = [d(p) for p in [(100, 100), (102, 101), (400, 90), (104, 102), (106, 103)]]
check(all(abs(o[0] - 103) < 5 for o in outs[2:]), f"a one-frame jump to x=400 is dropped ({outs})")
d = Despike(1)
check(d((400, 90)) == (400, 90), "median 1 is off")
r = Rate()
for i in range(60):
    fps = r.tick(i / 30)
check(abs(fps - 30) < 0.5, f"frame rate reads 30 at 30 fps ({fps:.2f})")

# --- hover lock onto controls: resting on a control locks (small ones snap to their centre), resting on
#     nothing does not; leaving the control's frame releases; a grab never releases ---
k = Lock(dwell=0.2, still=120, breakout=20)
t = 0.0
for i in range(10):                 # resting over empty space for 0.3 s
    k.update((500 + (i % 2) * 3, 400), t, 20.0, target=None)
    t += 1 / 30
check(k.at is None, "resting where there is no control does not lock (user: it locked wherever it stopped)")
button = (480, 385, 40, 30)          # a 40 x 30 button under the pointer
for i in range(10):
    out = k.update((505 + (i % 2) * 3, 402), t, 20.0, target=button)
    t += 1 / 30
check(k.at == (500, 400), f"resting on a small control snaps to its centre ({k.at})")
out = k.update((530, 420), t, 300.0)
check(out == (500, 400), "motion inside the control plus margin stays locked")
out = k.update((600, 400), t + 0.03, 900.0)
check(k.at is None and out == (600, 400), "leaving the control breaks out and the cursor follows")
fader = (100, 100, 30, 300)          # a tall fader: no snap, lock where the pointer is
k = Lock(dwell=0.2, still=120, breakout=20)
for i in range(10):
    k.update((110, 250), i / 30, 10.0, target=fader)
check(k.at == (110, 250), f"a big control locks in place, no snap ({k.at})")
k = Lock(dwell=0.2, still=120, breakout=50, anywhere=True)
for i in range(10):
    k.update((700, 700), i / 30, 10.0, target=None)
check(k.at == (700, 700), "`anywhere` restores the rest-anywhere lock")
k = Lock(dwell=0.2, still=120, breakout=50)
out = k.update((300, 300), 0.0, 400.0, held=True, grab_at=(290, 305), target=None)
check(out == (300, 300) and k.at is None, "a pinch over nothing does not lock (user, live: it locked wherever the pinch was)")
k = Lock(dwell=0.2, still=120, breakout=50)
knob = (270, 285, 40, 40)
out = k.update((300, 300), 0.0, 400.0, held=True, grab_at=(290, 305), target=knob)
check(out == (290, 305), "a pinch over a control locks where the pointer was just before the pinch")
out = k.update((500, 500), 0.1, 900.0, held=True)
check(out == (290, 305), "while the grab is held nothing breaks the lock")
out = k.update((500, 500), 0.2, 900.0, held=False)
check(out == (500, 500) and k.at is None, "after the grab a far hand breaks out")
k = Lock(dwell=0, still=120, breakout=50)
for i in range(20):
    out = k.update((100, 100), i / 30, 10.0)
check(k.at is None, "dwell 0 turns hover locking off")

# --- grab session: a quick pinch never presses; a held one presses, drags by the hand, and returns ---
from bench import Grab  # noqa: E402
gb = Grab("drag", gain=0.5, hold_s=0.25)
gb.begin((400, 300), 0.0, 500.0, "AXSlider")
check(gb.step(0.1, 500.0) is None, "no press before the hold time (a tap stays a gesture)")
check(gb.end() is None, "a quick pinch ends with no press and no release")
gb.begin((400, 300), 0.0, 500.0, "AXSlider")
check(gb.step(0.3, 500.0) == ("press", 400, 300), "a held pinch presses where the grab began")
a = gb.step(0.35, 480.0)
check(a == ("drag", 400, 290.0, -10.0), f"20 px of hand movement up drags 10 px up at gain 0.5 ({a})")
check(gb.step(0.4, 480.2) is None, "sub-pixel changes send nothing")
check(gb.end() == ("release", 400, 290.0, 400, 300), "release lets go where the drag ended and returns to the grab point")
gb.begin((400, 300), 0.0, 500.0, "AXButton")
check(gb.step(1.0, 400.0) is None, "drag mode never presses a button")
gb = Grab("any")
gb.begin((10, 10), 0.0, 0.0, None)
check(gb.step(0.3, 0.0)[0] == "press", "any mode grabs where Accessibility sees nothing (plug-in GUIs)")
gb = Grab("off")
gb.begin((10, 10), 0.0, 0.0, "AXSlider")
check(gb.step(1.0, 0.0) is None, "off never grabs")

# --- metric mapping: 3-D hand position to a steady 2-D cursor (the parallax fix) ---
from bench import MetricMap, px_per_metre  # noqa: E402


def project(world_hand, centre_m, f=900.0, w=1280, h=720):
    """A pinhole camera: a hand whose shape is `world_hand` (metres, hand-centred) with its centre at
    `centre_m` = (X, Y, Z) metres from the camera -> normalised image landmarks."""
    out = []
    for x, y, z in world_hand:
        X, Y, Z = centre_m[0] + x, centre_m[1] + y, centre_m[2] + z
        out.append((0.5 + f * X / Z / w, 0.5 + f * Y / Z / h, 0.0))
    return out


hand = [(0.01 * (i % 5) - 0.02, -0.02 * (i // 5) + 0.04, 0.003 * (i % 3)) for i in range(21)]  # a 21-point hand shape
img = project(hand, (0.0, 0.0, 0.5))
s = px_per_metre(img, hand, 1280, 720)
check(abs(s - 900 / 0.5) / (900 / 0.5) < 0.02, f"px per metre = f / distance ({s:.0f} vs 1800)")
check(abs(px_per_metre(project(hand, (0.0, 0.0, 1.0)), hand, 1280, 720) - 900) < 20, "twice as far, half the pixels per metre")


def cursor_at(m, centre_m, anchor=0, frames=40):
    """Feed the same hand pose a few frames (the depth estimate is smoothed) and return the cursor."""
    out = None
    for _ in range(frames):
        lmk = project(hand, centre_m)
        out = m(lmk[anchor][0], lmk[anchor][1], lmk, hand, 1280, 720)
    return out


m0 = MetricMap(1728, 1117, pad=(0.24, 0.15))
first = cursor_at(m0, (0.0, 0.02, 0.50), anchor=10, frames=200)   # anchor 10 sits 4 cm above the hand's lowest point
anchor_x = -(0.0 + hand[10][0])                    # the anchor landmark, mirrored, in metres from the camera line
check(abs(first[0] - (1727 / 2 + anchor_x / 0.24 * 1727)) < 3, f"across is centred on the camera line ({first})")
check(abs(m0.below - 0.04) < 0.004, f"it learns how far the hand hangs below the anchor ({m0.below:.3f} m vs 0.04)")
edge_y = m0.centre[1] + 0.075                      # the anchor's height at the pad's bottom edge
at_edge = cursor_at(m0, (0.0, edge_y, 0.50), anchor=10, frames=200)
lowest = max(p[1] for p in project(hand, (0.0, edge_y, 0.50)))
check(abs(at_edge[1] - 1116) < 2 and abs(lowest - 0.95) < 0.01,
      f"at the screen's bottom edge the whole hand is still in view (lowest point at {lowest:.3f} of the frame)")
low = MetricMap(1728, 1117, pad=(0.24, 0.15))
cursor_at(low, (0.0, 0.15, 0.50), anchor=10, frames=200)
low.recenter_next = True
cursor_at(low, (0.0, 0.15, 0.50), anchor=10, frames=200)
check(low.centre[1] < 0.15 - 0.05, f"recenter with the hand at the frame's bottom still keeps the pad in view ({low.centre[1]:.3f} m)")
high = MetricMap(1728, 1117, pad=(0.24, 0.15))
cursor_at(high, (0.03, -0.05, 0.50), frames=60)
high.recenter_next = True
cursor_at(high, (0.03, -0.05, 0.50), frames=60)
check(abs(high.centre[1] - (-0.05 + hand[0][1])) < 0.002, f"recenter higher up is kept as said ({high.centre[1]:.3f} m)")
m = MetricMap(1728, 1117, pad=(0.24, 0.15))
start = cursor_at(m, (0.05, -0.02, 0.50))
m.recenter_next = True
start = cursor_at(m, (0.05, -0.02, 0.50))
check(abs(start[0] - 1727 / 2) < 1 and abs(start[1] - 1116 / 2) < 1, f"recenter makes the current position the centre ({start})")
toward = cursor_at(m, (0.05, -0.02, 0.35))
check(abs(toward[0] - start[0]) < 12 and abs(toward[1] - start[1]) < 12,
      f"moving 15 cm toward the camera, off-axis, barely moves the cursor ({start} -> {toward})")
m2 = MetricMap(1728, 1117, pad=(0.24, 0.15))
m2.recenter_next = True                        # measure from where the hand is, like saying "recenter"
cursor_at(m2, (0.0, 0.0, 0.40))
near = cursor_at(m2, (-0.02, 0.0, 0.40))      # 2 cm to the camera's left = the user's right (mirrored)
m3 = MetricMap(1728, 1117, pad=(0.24, 0.15))
m3.recenter_next = True
cursor_at(m3, (0.0, 0.0, 0.70))
far = cursor_at(m3, (-0.02, 0.0, 0.70))
check(abs((near[0] - 1727 / 2) - (far[0] - 1727 / 2)) < 8,
      f"2 cm sideways moves the cursor the same near and far ({near[0] - 863.5:.0f} vs {far[0] - 863.5:.0f} px)")
check(near[0] > 1727 / 2, "the user's right is the cursor's right")
check(abs((near[0] - 1727 / 2) - 0.02 / 0.24 * 1727) < 8, f"2 cm of a 24 cm pad is 1/12 of the screen ({near[0] - 863.5:.0f} px)")
a = MetricMap(1728, 1117)
check(a(0.5, 0.5, None, None, 1280, 720) is None, "no world landmarks, no metric position")

# --- precision: slow movement moves the cursor less, the cursor catches up on quick moves ---
from bench import Precision  # noqa: E402
off = Precision(1.0)
check(off((100.0, 100.0), 0.0) == (100.0, 100.0) and off((110.0, 100.0), 0.033) == (110.0, 100.0), "gain 1 is off")
pr = Precision(0.35)
t = 0.0
pr((500.0, 500.0), t)
for k in range(1, 31):                                  # 1 px per frame = 30 px/s: slow drift
    t += 1 / 30
    at = pr((500.0 + k, 500.0), t)
check(abs(at[0] - (500 + 30 * 0.35)) < 0.5, f"a slow 30 px drift moves the cursor about a third of it ({at[0] - 500:.1f} px)")
wob = Precision(0.35)
t = 0.0
wob((500.0, 500.0), t)
xs = []
for k in range(1, 61):                                  # a 3 px wobble back and forth (90 px/s, what gets past the filter)
    t += 1 / 30
    xs.append(wob((500.0 + (3 if k % 2 else 0), 500.0), t)[0])
check(max(xs) - min(xs) < 3 * 0.35 + 0.2, f"a 3 px wobble shrinks to about 1 px ({max(xs) - min(xs):.1f})")
for k in range(1, 16):                                  # then a quick 600 px move (40 px a frame = 1200 px/s)
    t += 1 / 30
    at = wob((503.0 + 40 * k, 500.0), t)
check(abs(at[0] - (503 + 600)) < 15, f"a quick move brings the cursor back to the hand ({at[0] - 1103:+.1f} px off)")
still = Precision(0.35)
t = 0.0
still((500.0, 500.0), t)
for k in range(1, 21):                                  # a slow 20 px drift builds an offset
    t += 1 / 30
    at = still((500.0 + k, 500.0), t)
held = []
for k in range(20):                                     # then the hand holds still
    t += 1 / 30
    held.append(still((520.0, 500.0), t)[0])
check(still.offset() > 10 and max(held) - min(held) < 1e-9, f"a still hand never pulls the cursor (offset {still.offset():.1f} px waits for a move)")
gap = Precision(0.35)
gap((100.0, 100.0), 0.0)
gap((105.0, 100.0), 0.033)
check(gap((300.0, 300.0), 0.5) == (300.0, 300.0), "after a gap in the frames the cursor starts at the hand's spot")

# --- zones: bigger targets (user, live: "widen the targets ... to the zones") ---
from bench import rect_distance, openness, Overlay, FINGER_JOINTS  # noqa: E402
check(rect_distance((5, 5), (0, 0, 10, 10)) == 0, "inside a control is distance 0")
check(abs(rect_distance((13, 14), (0, 0, 10, 10)) - 5) < 1e-9, "distance to the nearest corner")
check(rect_distance((-7, 5), (0, 0, 10, 10)) == 7, "distance to the nearest side")
z = Lock(dwell=0.2, still=120, breakout=15, zone=40)
t = 0.0
for _ in range(10):                                   # resting 30 px left of an 18 px send knob, within the zone
    out = z.update((470.0, 409.0), t, 10.0, target=(500, 400, 18, 18))
    t += 0.033
check(out == (509.0, 409.0), f"resting within the zone snaps onto the small control ({out})")
check(z.update((455.0, 409.0), t, 10.0, target=None) == (509.0, 409.0), "the lock holds anywhere inside the zone")
check(z.update((440.0, 409.0), t, 10.0, target=None) == (440.0, 409.0), "leaving zone + breakout lets go")
tall = Lock(dwell=0.2, still=120, breakout=15, zone=40)
out = tall.update((230.0, 950.0), 0.0, 10.0, held=True, grab_at=(230.0, 950.0), target=(197, 901, 20, 180))
check(out == (216.0, 950.0), f"a grab next to a tall fader track presses on the track, not beside it ({out})")
# --- a closed hand ---
flat = [(0.0, 0.0, 0.0)] * 21
straight = list(flat)
for m, p_, d, tip in FINGER_JOINTS:
    x = 0.02 * m
    straight[m], straight[p_], straight[d], straight[tip] = (x, 0.0, 0.0), (x, 0.04, 0.0), (x, 0.065, 0.0), (x, 0.085, 0.0)
check(openness(straight) > 0.99, f"straight fingers are open ({openness(straight):.2f})")
curled = list(straight)
for m, p_, d, tip in FINGER_JOINTS:
    x = 0.02 * m
    curled[p_], curled[d], curled[tip] = (x, 0.04, 0.0), (x, 0.04, 0.025), (x, 0.02, 0.03)
check(openness(curled) < 0.72, f"curled fingers are a closed hand ({openness(curled):.2f})")
half = list(curled)
m, p_, d, tip = FINGER_JOINTS[0]
half[p_], half[d], half[tip] = straight[p_], straight[d], straight[tip]
check(openness(half) > 0.9, "one straight finger is not a closed hand (pointing)")
Overlay(enabled=False).update((10, 10), (0, 0, 5, 5), locked=True)   # disabled: a no-op, no AppKit needed
# --- the grab state travels with the frame ---
import io, json as _json  # noqa: E402
from frames import FrameWriter  # noqa: E402
fw = FrameWriter("-", 1280, 720)
class Keep(io.StringIO):
    def close(self):                                  # the writer closes its file at the end; keep the text
        pass


buf = Keep()
fw.f = buf
fw.write(None, "tracked", 1.0, 1.01, extra={"grab": "active"})
fw.write(None, "tracked", 1.03, 1.04)
fw.close()
recs = [_json.loads(x) for x in buf.getvalue().splitlines()]
check(recs[0].get("grab") == "active" and "grab" not in recs[1], f"extra fields ride on the frame ({[r.get('grab') for r in recs]})")

print("camera-bench: %d failures" % fails)
sys.exit(1 if fails else 0)
