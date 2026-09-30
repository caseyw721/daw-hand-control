#!/usr/bin/env python3
"""Camera hand-tracking pointer bench (Phase 0, no wearable).

Tracks one hand with MediaPipe Hand Landmarker on the built-in camera and drives the
macOS cursor from it, so the same test app (controller/test-app) can measure it with
tests A–D exactly as it measures the ring. A thumb-to-index pinch is a *gesture*: it is
reported (P line, Tink sound) but never clicks unless --click is given.

    ./run.sh                     # absolute pointing, index fingertip, no preview
    ./run.sh --calibrate 5       # abs: the first 5 s with a hand in view set the box (sweep your range)
    ./run.sh --rect .3,.4,.9,.95 # abs: use this box of the raw camera frame (x0,y0,x1,y1)
    ./run.sh --mode rel          # relative ("air mouse") pointing
    ./run.sh --aim-calibrate     # aim from your eye through your fingertip; point at five dots first (saved)
    ./run.sh --point palm        # steadier point: palm centre instead of fingertip
    ./run.sh --preview           # show the camera with landmarks and the box (costs a few ms)
    ./run.sh --no-cursor         # measure only, leave the cursor alone
    ./run.sh --log run.csv       # per-frame timings and positions
    ./run.sh --marker            # big black/white square follows the cursor (for slow-mo latency filming)
    ./run.sh --hold-ms 180       # coast through hand losses shorter than this (MediaPipe flickers, mostly mid-pinch)
    ./run.sh --lead-ms 40        # predict this far ahead while the hand moves (buys back latency, costs overshoot)
    ./run.sh --frames -          # every frame's landmarks as one JSON line on stdout, for the gesture engine (frames.py)

Needs macOS Camera permission for the terminal (first run asks) and Accessibility
permission to move the cursor. Ctrl-C stops it.

Lines on stderr:
    I,...                                  information (camera, screen, calibrated box, warnings)
    P,<t s>,<down|up>,<pinch m>,<x>,<y>    pinch events (a Tink sound plays on "down")
    S,<fps>,<detect ms p50>,<detect ms p95>,<loop ms p50>,<hand %>,<still>,<jitter px rms>,<pinch>,<held gaps>,<lost>
                                           once a second. "still" = the hand rested for a half-second window this
                                           second (judged on the raw point, see still_windows), "jitter" = rms of the
                                           shown point in those windows; "held gaps" = short losses coasted through,
                                           "lost" = real ones
"""
import argparse
import collections
import csv
import math
import os
import subprocess
import sys
import time

from frames import FrameWriter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "hand_landmarker.task")

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 13, 17


class OneEuro:
    """1€ filter (Casiez et al. 2012). Units: whatever you feed it, per second. `dx` is the low-passed
    derivative of the *raw* input, as in the paper: taking it against the previous filtered value
    instead over-reads speed by lag/dt (a third too high at 1000 px/s here), which opened the cutoff
    early and made lead() overshoot."""

    def __init__(self, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x = self.dx = self.raw = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.raw, self.t = x, 0.0, x, t
            return x
        dt = max(t - self.t, 1e-4)
        self.t = t
        dx = (x - self.raw) / dt
        self.raw = x
        a_d = self._alpha(self.dcutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mincutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x

    def lead(self, seconds):
        """Where the filtered value will be `seconds` from now if it keeps its filtered velocity:
        a fixed-lead prediction that buys back that much of the pipeline's latency, paid for with
        overshoot when the hand stops."""
        return self.x + self.dx * seconds

    def reset(self):
        self.x = self.dx = self.raw = self.t = None


def lead_gain(speed, v0=150.0, v1=500.0):
    """How much of the lead to apply at this filtered speed (px/s): none at rest, all of it once the
    hand is clearly moving, a ramp between. The filtered velocity of a still hand reaches ~95 px/s
    (p99 in the recordings, 1€ 3 Hz / 0.05), and predicting from that only turns it into jitter."""
    return min(max((speed - v0) / (v1 - v0), 0.0), 1.0)


def gray_prep(bgr, clahe):
    """Greyscale plus a contrast lift (CLAHE) on a half-size frame, back to 3 channels for the model.
    Measured on a dim red-lit recording: the hand found in 95% of frames, against 93% raw and 88% for
    `enhance`; 0.3 ms. (Edges only, the other idea tried: 0%, the model needs a natural image.)"""
    import cv2
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(clahe.apply(g), cv2.COLOR_GRAY2BGR)


class Precision:
    """Slow hand movement moves the cursor less: below `v_lo` px/s the cursor moves `gain` of the hand's
    movement, from `v_hi` up all of it, a ramp between (the PRISM idea, Frees et al. 2007; a mouse's
    acceleration curve does the same). Hand tremor and drift are slow, so they shrink with it, and aiming
    is a fast move followed by slow corrections, so the corrections get finer. The offset this builds
    between hand and cursor is paid back only in the hand's direction of travel: `pull` of it per frame
    on fast moves, `bleed` on moderate ones, none while the hand is still. A gap in the frames starts
    over at the hand's spot. Replayed on the 2026-09-29 sessions at gain 0.35: still jitter p50 9.3 to
    4.0 px, p90 22 to 16 px; the cursor sits 22 px (p50) and 60 px (p90) from the hand's own spot.
    gain 1 is off."""

    def __init__(self, gain=1.0, v_lo=150.0, v_hi=600.0, pull=0.3, bleed=0.05):
        self.gain, self.v_lo, self.v_hi, self.pull, self.bleed = gain, v_lo, v_hi, pull, bleed
        self.reset()

    def reset(self):
        self.at = self.last = self.t = None

    def __call__(self, p, t):
        if self.gain >= 1.0:
            self.reset()
            return p
        if self.at is None or t - self.t > 0.3:
            self.at, self.last, self.t = p, p, t
            return p
        dx, dy = p[0] - self.last[0], p[1] - self.last[1]
        v = math.hypot(dx, dy) / max(t - self.t, 1e-3)
        k = self.gain + (1.0 - self.gain) * min(max((v - self.v_lo) / (self.v_hi - self.v_lo), 0.0), 1.0)
        x, y = self.at[0] + k * dx, self.at[1] + k * dy
        ox, oy = p[0] - x, p[1] - y
        pull = self.pull if v > self.v_hi else (self.bleed if v > self.v_lo else 0.0)
        if pull and ox * dx + oy * dy > 0:
            x, y = x + pull * ox, y + pull * oy
        self.at, self.last, self.t = (x, y), p, t
        return self.at

    def offset(self):
        """How far the cursor sits from the hand's own spot, px (for the preview and diagnostic page)."""
        return 0.0 if self.at is None else math.hypot(self.last[0] - self.at[0], self.last[1] - self.at[1])


class Nudge:
    """Aim, then fine-tune (user, live 2026-09-30: grabbing a control "without taking 30 seconds"; aim mode lands
    within ~3 cm, 7 cm at worst). While the hand moves, the cursor is where the aim is. Once the aim has rested (its
    speed under `still` px/s) for `dwell` s, the cursor lets go of the aim: from then on it moves `gain` of the aim's
    movement from where it was, so the last few cm are nudged away by a small deliberate move, and the aim's jitter
    shrinks by the same factor. A quick move (over `snap` px/s for two frames) or the cursor ending up `leash` px from
    the aim puts it back on the aim. Unlike Precision (always on, so the cursor slid away from where the user
    pointed: "there's some drift") this only starts on a pause and always snaps back.
    `freeze`: a pinch or a closing hand moves the fingertip, and with it the aim, without meaning to move the cursor;
    while one is forming the cursor holds still and nothing snaps, so a grab lands where the nudge put the cursor.
    Speeds are an average of the last few frames' aim movement (the 1-euro filter's own derivative is low-passed at
    1 Hz and would notice a quick move 150 ms late). gain 0 or 1 is off."""

    def __init__(self, gain=0.0, dwell=0.3, still=200.0, snap=1200.0, leash=500.0):
        self.gain, self.dwell, self.still, self.snap, self.leash = gain, dwell, still, snap, leash
        self.reset()

    def reset(self):
        self.on = False
        self.at = self.last = self.t = self.still_since = None
        self.v = 0.0
        self.fast = 0

    @property
    def enabled(self):
        return 0.0 < self.gain < 1.0

    def __call__(self, aim, t, freeze=False):
        if not self.enabled:
            self.reset()
            return aim
        if self.last is None or t - self.t > 0.3:          # the first frame, or a gap in the frames: start over
            self.reset()
            self.last, self.t = aim, t
            return aim
        dx, dy = aim[0] - self.last[0], aim[1] - self.last[1]
        v = math.hypot(dx, dy) / max(t - self.t, 1e-3)
        self.v += 0.4 * (v - self.v)
        self.last, self.t = aim, t
        if freeze:
            self.fast, self.still_since = 0, None
            return self.at if self.on else aim
        if self.on:
            self.fast = self.fast + 1 if self.v > self.snap else 0
            at = (self.at[0] + self.gain * dx, self.at[1] + self.gain * dy)
            if self.fast >= 2 or math.hypot(aim[0] - at[0], aim[1] - at[1]) > self.leash:
                self.on, self.at, self.still_since, self.fast = False, None, None, 0
                return aim
            self.at = at
            return at
        if self.v < self.still:
            if self.still_since is None:
                self.still_since = t
            elif t - self.still_since >= self.dwell:
                self.on, self.at = True, aim
        else:
            self.still_since = None
        return aim


GRAB_HAND_PX = 1.5           # aim-mode drags: screen px per camera px of the hand's base (x grab gain); ~6 cm = a fader
TAP_S = 0.4                  # the finger gun: a trigger press shorter than this that did not grab is a click
REWIND_S = 0.6               # aim mode: how far back a pinch or grip looks for where the cursor was before it began


def rewind_point(history, hand_hist, t, what="pinch", span=REWIND_S, noise=None):
    """Where the cursor was just before a pinch (or, with `grip`, a closing hand) began, for the grab and the click.
    In aim mode the ray runs through the fingertip, and closing the fingers moves it: over the 0.4 s before a pinch
    closed the aim moved 229 px (median) and 957 px (p90) in the first nudge session (user, live: "I can't grab
    anything because it freaks out and points everywhere"). The pinch's own speed does not separate it from pointing
    (the user's thumb rests ~6 cm from the tip and pinches close slowly), so instead of predicting it the tracker
    looks back from the close to where the closing began: the peak of the thumb-to-index distance (or of the hand's
    openness) within `span` s, walking back until the value falls again by more than `noise`. history: (t, x, y);
    hand_hist: (t, pinch m, openness[, thumb press m]); `what`: pinch, grip or thumb (the thumb trigger). None when
    there is nothing to go back to."""
    noise = noise if noise is not None else (0.03 if what == "grip" else 0.004)
    when, most = None, -1.0
    for row in reversed(hand_hist):
        th = row[0]
        if t - th > span:
            break
        v = {"pinch": row[1], "grip": row[2], "thumb": row[3] if len(row) > 3 else row[1]}[what]
        if v > most:
            when, most = th, v
        elif v < most - noise:                   # past the peak: the fingers were closing from there on
            break
    if when is None:
        return None
    best = None
    for (tp, x, y) in history:                   # the cursor at (or just before) that moment
        if tp <= when:
            best = (x, y)
        else:
            break
    return best


class Despike:
    """Median of the last n points: a single-frame landmark glitch never reaches the cursor. Costs
    (n-1)/2 frames of delay; n=1 is off."""

    def __init__(self, n=3):
        self.n = n
        self.buf = collections.deque(maxlen=max(1, n))

    def __call__(self, p):
        if self.buf.maxlen != max(1, self.n):
            self.buf = collections.deque(self.buf, maxlen=max(1, self.n))
        self.buf.append(p)
        xs = sorted(q[0] for q in self.buf)
        ys = sorted(q[1] for q in self.buf)
        return xs[len(xs) // 2], ys[len(ys) // 2]

    def reset(self):
        self.buf.clear()


class Rate:
    """Frames per second as the user sees it: an exponential average of the frame interval."""

    def __init__(self, k=0.1):
        self.k, self.dt, self.last = k, None, None

    def tick(self, t):
        if self.last is not None:
            d = t - self.last
            self.dt = d if self.dt is None else self.dt + self.k * (d - self.dt)
        self.last = t
        return 1.0 / self.dt if self.dt else 0.0


def enhance(bgr, clahe):
    """Low-light correction before detection: grey-world white balance (a red-lit room turns neutral),
    then CLAHE on the lightness channel (dim hands gain contrast without blowing out lamps).
    Works on a half-size frame: the landmark model downsamples to ~224 px anyway and landmarks are
    normalised, so nothing is lost, and it costs a few ms instead of ~20."""
    import cv2
    import numpy as np
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    means = cv2.mean(small)[:3]
    gains = np.array(means).mean() / (np.array(means) + 1e-3)
    balanced = cv2.transform(small, np.diag(gains).astype(np.float32))
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


class Pinch:
    """Hysteresis on a closeness measure: below `on` for `min_frames` frames closes, above `off`
    for `off_frames` frames opens. Works on the world distance (metres) or the 2-D ratio."""

    def __init__(self, on=0.30, off=0.45, min_frames=2, off_frames=None):
        self.on_thr, self.off_thr = on, off
        self.min_frames = min_frames
        self.off_frames = min_frames if off_frames is None else off_frames
        self.closed = False
        self._run = 0

    def update(self, v):
        """Returns 'down', 'up' or None."""
        if not self.closed:
            self._run = self._run + 1 if v < self.on_thr else 0
            if self._run >= self.min_frames:
                self.closed, self._run = True, 0
                return "down"
        else:
            self._run = self._run + 1 if v > self.off_thr else 0
            if self._run >= self.off_frames:
                self.closed, self._run = False, 0
                return "up"
        return None

    def reset(self):
        """Forget a pinch in progress *and* the half-counted frames: clearing only `closed` left a
        release count behind, and one sub-threshold frame later could fire 'down' on its own."""
        self.closed, self._run = False, 0


FINGER_JOINTS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))   # mcp, pip, dip, tip


THUMB_PRESS_POINTS = (INDEX_MCP, 6, MIDDLE_MCP, 10)    # the index and middle fingers' big knuckles and middle joints


def thumb_press(wlm):
    """The thumb trigger ("finger gun", aim mode): how close the thumb tip is to the side of the hand (the index and middle
    fingers' base and middle joints), world metres. Pressing the thumb down leaves the index fingertip, and so the aim,
    nearly still (user, live: "it moves my index finger just a tiny bit but the thumb moves more"), where a pinch swings
    it 229 px (median)."""
    return min(math.dist(wlm[THUMB_TIP], wlm[k]) for k in THUMB_PRESS_POINTS)


class ThumbTrigger:
    """The finger gun's trigger: aim with the thumb raised, drop it against the side of the hand to press, raise it to let
    go. The user's thumb rests against the hand much of the time (thumb-to-hand median 2.2 cm, often under 1 cm, live
    2026-09-30), so a thumb that is merely down must never press: the trigger arms only after the thumb has been up
    (over `off`) for `arm_s`, and a press is a drop under `on` for `min_frames` frames after that. Releasing (over `off`
    for `off_frames` frames) ends the press and starts arming again."""

    def __init__(self, on=0.015, off=0.03, arm_s=0.2, min_frames=2, off_frames=3):
        self.on_thr, self.off_thr, self.arm_s, self.min_frames, self.off_frames = on, off, arm_s, min_frames, off_frames
        self.reset()

    def reset(self):
        self.closed = False
        self.up_since = None
        self._run = 0

    def update(self, v, t):
        """'down', 'up' or None."""
        if not self.closed:
            if v > self.off_thr:
                self.up_since = t if self.up_since is None else self.up_since
                self._run = 0
                return None
            armed = self.up_since is not None and t - self.up_since >= self.arm_s
            if v < self.on_thr and armed:
                self._run += 1
                if self._run >= self.min_frames:
                    self.closed, self._run, self.up_since = True, 0, None
                    return "down"
            else:
                self._run = 0
                if v < self.on_thr:          # down without having been up long enough: stays disarmed
                    self.up_since = None
            return None
        self._run = self._run + 1 if v > self.off_thr else 0
        if self._run >= self.off_frames:
            self.closed, self._run, self.up_since = False, 0, t
            return "up"
        return None


def openness(wlm):
    """How straight the straightest of the four fingers is, on world landmarks: tip-to-knuckle distance over the
    finger's length, 1.0 straight, ~0.45 curled (the gesture engine's `ext`). A closed hand (every finger
    curled) is under ~0.7: the whole-hand grab (user, live: "just grab it")."""
    best = 0.0
    for m, p, d, t in FINGER_JOINTS:
        seg = math.dist(wlm[p], wlm[m]) + math.dist(wlm[d], wlm[p]) + math.dist(wlm[t], wlm[d])
        if seg > 1e-9:
            best = max(best, math.dist(wlm[t], wlm[m]) / seg)
    return best


def pinch_ratio(lm):
    """2-D fallback. lm: list of (x, y, z) normalised image landmarks. Thumb tip to index tip over
    hand size; inflates when the hand tilts towards the camera (the palm foreshortens)."""
    d = math.dist(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2])
    size = math.dist(lm[WRIST][:2], lm[MIDDLE_MCP][:2]) or 1e-6
    return d / size


def pinch_dist(wlm):
    """Thumb tip to index tip in MediaPipe's world landmarks: metres at the model's average hand
    size, so it does not change with hand orientation or distance from the camera."""
    return math.dist(wlm[THUMB_TIP], wlm[INDEX_TIP])


THUMB_MCP, THUMB_IP = 2, 3
POINTS = ("tip", "base", "thumb", "thumbip", "thumbtip", "web", "palm")
PREPS = ("none", "enhance", "gray")
MAPS = ("abs", "rel", "metric", "aim")


def point_of(lm, which):
    """The point that drives the cursor.
    tip       index fingertip: precise, but it moves whenever the index curls or pinches
    thumb     mean of the thumb base, the thumb's middle joint and the index knuckle: none of them move
              when the index finger grabs (user's design), and averaging three joints halves the noise
              (recorded in a dim bar: 29 jumps over 40 px in 20 s, against 66 for the fingertip)
    thumbip   the thumb's middle joint alone (38 jumps)
    thumbtip  thumb tip
    web       between the thumb base and the index knuckle
    palm      mean of the four finger knuckles"""
    if which == "tip":
        return lm[INDEX_TIP][0], lm[INDEX_TIP][1]
    if which == "base":
        # wrist + the four finger knuckles: they move only when the whole hand moves, never when fingers bend.
        # User, live: "the thumb ... does move probably forty percent where the index finger moves sixty";
        # measured on real pinches: the thumb anchor moved 440 px, this one 249, with the same noise
        ks = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
        return sum(lm[k][0] for k in ks) / 5, sum(lm[k][1] for k in ks) / 5
    if which == "thumb":
        ks = (THUMB_MCP, THUMB_IP, INDEX_MCP)
        return sum(lm[k][0] for k in ks) / 3, sum(lm[k][1] for k in ks) / 3
    if which == "thumbip":
        return lm[THUMB_IP][0], lm[THUMB_IP][1]
    if which == "thumbtip":
        return lm[THUMB_TIP][0], lm[THUMB_TIP][1]
    if which == "web":
        return (lm[THUMB_MCP][0] + lm[INDEX_MCP][0]) / 2, (lm[THUMB_MCP][1] + lm[INDEX_MCP][1]) / 2
    ks = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    return sum(lm[k][0] for k in ks) / 4, sum(lm[k][1] for k in ks) / 4


def rect_distance(p, frame):
    """Distance in px from point p to the rectangle (x, y, w, h); 0 inside."""
    x, y, w, h = frame
    dx = max(x - p[0], 0.0, p[0] - (x + w))
    dy = max(y - p[1], 0.0, p[1] - (y + h))
    return math.hypot(dx, dy)


CONTROL_ROLES = {"AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXSlider",
                 "AXIncrementor", "AXValueIndicator", "AXMenuItem", "AXMenuBarItem", "AXDisclosureTriangle",
                 "AXComboBox", "AXColorWell", "AXLevelIndicator", "AXDockItem", "AXCell", "AXLink", "AXTab"}
SNAP_MAX_PX = 90          # a control no bigger than this (both ways) snaps the cursor to its centre
STICKY_PX = 25            # a highlighted control stays the target this far beyond its zone
READ_FAILS_REOPEN = 30    # consecutive failed camera reads before the camera is closed and opened again


class Lock:
    """Hover lock onto controls (user, live: "it's not locking to buttons when they're hovered over, it's
    just locking wherever it stops"). When the pointer rests (filtered speed under `still` px/s) for
    `dwell` s over a control that Accessibility reports (a button, slider, knob, checkbox, menu item...),
    the cursor locks onto it: small controls snap to their centre, big ones lock where the cursor is.
    Hand motion is ignored until it leaves the control's frame by `breakout` px. Resting anywhere else
    does not lock (`anywhere` restores the old rest-anywhere lock). While a grab is held (pinch closed)
    it never releases; a grab that starts unlocked locks where the pointer was just before the pinch."""

    def __init__(self, dwell=0.2, still=120.0, breakout=50.0, anywhere=False, zone=0.0):
        self.dwell, self.still, self.breakout, self.anywhere = dwell, still, breakout, anywhere
        self.zone = zone           # px around a control that counts as on it (the probe's zone): the lock keeps it too
        self.at = None             # (x, y) while locked
        self.frame = None          # (x, y, w, h) of the control locked onto
        self.still_since = None

    def _inside(self, pos):
        if self.frame is None:
            return math.hypot(pos[0] - self.at[0], pos[1] - self.at[1]) <= self.breakout
        x, y, w, h = self.frame
        m = self.breakout + self.zone
        return x - m <= pos[0] <= x + w + m and y - m <= pos[1] <= y + h + m

    @staticmethod
    def land(pos, target):
        """Where a lock on `target` puts the cursor: a small control's centre, else pos pulled inside the frame
        (with zones the cursor may be up to `zone` px off the control, and a grab must press on the control)."""
        x, y, w, h = target
        if w <= SNAP_MAX_PX and h <= SNAP_MAX_PX:
            return (x + w / 2, y + h / 2)
        return (min(max(pos[0], x + 1), x + w - 1), min(max(pos[1], y + 1), y + h - 1))

    def update(self, pos, t, speed, held=False, grab_at=None, target=None):
        """The position to show for this frame. `target` is the control under the pointer, (x, y, w, h)
        in screen points, or None."""
        if self.at is not None:
            if held or self._inside(pos):
                return self.at
            self.at = self.frame = None
            self.still_since = None
            return pos
        if held:
            if target is None and not self.anywhere:
                return pos        # a pinch over nothing does not lock (user, live: "it still locks wherever the pinch is")
            self.at = self.land(grab_at or pos, target) if target is not None else (grab_at or pos)
            self.frame = target
            return self.at
        if self.dwell > 0 and speed < self.still and (target is not None or self.anywhere):
            if self.still_since is None:
                self.still_since = t
            elif t - self.still_since >= self.dwell:
                if target is not None:
                    self.at = self.land(pos, target)
                    self.frame = target
                else:
                    self.at = pos
                return self.at
        else:
            self.still_since = None
        return pos

    def reset(self):
        self.at = self.frame = None
        self.still_since = None


class AXProbe:
    """What is under the cursor, from Accessibility, a few times a second on a background thread (a
    lookup takes ~15 ms, too slow for the frame loop). `control_at(pos)` gives the frame of a control
    containing pos if one was seen there recently, else None. Without the bindings or the permission it
    stays empty and hover locking simply never engages."""

    def __init__(self, period=0.08, zone=0.0):
        self.period = period
        self.zone = zone           # px: a control this close counts as under the cursor (user: "widen the targets")
        self.pos = None
        self.found = None          # (t, role, (x, y, w, h), title, is_control)
        self.cache = {}            # container key -> (t, [(role, frame, name)]) for the zone search
        self.ok = False
        try:
            import ApplicationServices as AS
            self.AS = AS
            self.sys = AS.AXUIElementCreateSystemWide()
            self.ok = True
        except Exception:
            self.AS = None
        if self.ok:
            import threading
            threading.Thread(target=self._run, name="ax-probe", daemon=True).start()

    def _attr(self, el, name):
        err, v = self.AS.AXUIElementCopyAttributeValue(el, name, None)
        return v if err == 0 else None

    def _frame(self, el):
        p, s = self._attr(el, "AXPosition"), self._attr(el, "AXSize")
        if p is None or s is None:
            return None
        ok, pt = self.AS.AXValueGetValue(p, self.AS.kAXValueCGPointType, None)
        ok2, sz = self.AS.AXValueGetValue(s, self.AS.kAXValueCGSizeType, None)
        if not (ok and ok2):
            return None
        return (pt.x, pt.y, sz.width, sz.height)

    def classify(self, el):
        """(role, frame, name) of the control at this element, or (role, frame, name, False) style None-role if it
        is not one. Logic reports its pan knobs under a same-size 'knob readout' text, its plug-in slots as groups
        named 'audio plug-in', and much else as elements with press/increment actions (surveyed live)."""
        role = str(self._attr(el, "AXRole") or "")
        frame = self._frame(el)
        name = str(self._attr(el, "AXTitle") or self._attr(el, "AXDescription") or "")
        low = name.lower()
        if frame is None:
            return role, None, name, False
        small = frame[2] <= 120 and frame[3] <= 200
        if role in CONTROL_ROLES:
            return role, frame, name, True
        if role == "AXStaticText" and ("knob" in low or "readout" in low) and small:
            return "AXSlider", frame, name, True             # the pan knob under its readout: grab it like a knob
        if role == "AXGroup" and "plug" in low and small:
            return "AXButton", frame, name, True              # a plug-in slot
        err, acts = self.AS.AXUIElementCopyActionNames(el, None)
        acts = set(acts or []) if err == 0 else set()
        if small and acts & {"AXPress", "AXIncrement", "AXDecrement", "AXPick"}:
            return ("AXSlider" if "AXIncrement" in acts else "AXButton"), frame, name, True
        parent = self._attr(el, "AXParent")
        if parent is not None:
            prole = str(self._attr(parent, "AXRole") or "")
            pframe = self._frame(parent)
            if prole in CONTROL_ROLES and pframe and pframe[2] <= 120 and pframe[3] <= 200:
                return prole, pframe, str(self._attr(parent, "AXTitle") or self._attr(parent, "AXDescription") or name), True
        return role, frame, name, False

    def _knob(self, el, frame):
        """A long slider's own thumb when it has one: Logic's volume fader is a 20x180 track with a 22x46 'fader knob'
        inside (surveyed live), and grabbing or snapping to the track's middle misses the knob. A square slider (a
        knob) keeps its own frame: the track header's pan knob reports a 33 px value box offset from the 25 px knob,
        and a press aimed there landed on the track header and dragged the whole track (user, live)."""
        if frame is None or max(frame[2], frame[3]) < 2 * min(frame[2], frame[3]):
            return frame
        for k in self._attr(el, "AXChildren") or []:
            if str(self._attr(k, "AXRole") or "") == "AXValueIndicator":
                f = self._frame(k)
                if f and f[2] * f[3] > 0:
                    return f
        return frame

    def _controls_in(self, node, depth=3, limit=160):
        """Controls among a container's descendants: [(role, frame, name)], cached for 2 s per container."""
        key = (str(self._attr(node, "AXRole") or ""), self._frame(node))
        hit = self.cache.get(key)
        if hit and time.monotonic() - hit[0] < 2.0:
            return hit[1]
        out, stack, seen = [], [(node, 0)], 0
        while stack and seen < limit:
            el, d = stack.pop()
            seen += 1
            if d > 0:
                role, frame, name, is_control = self.classify(el)
                if is_control and frame and frame[2] * frame[3] <= 300 * 300:
                    if role == "AXSlider":
                        frame = self._knob(el, frame)
                    out.append((role, frame, name))
                    continue
            if d < depth:
                stack.extend((k, d + 1) for k in (self._attr(el, "AXChildren") or []))
        if len(self.cache) > 64:
            self.cache.clear()
        self.cache[key] = (time.monotonic(), out)
        return out

    def _nearby(self, el, pos):
        """The nearest control within `zone` px of pos, searching the element under the cursor and up to two of its
        containers (a mixer channel strip holds its fader, pan and sends), never a whole window."""
        node = el
        for _ in range(3):
            if node is None:
                break
            frame = self._frame(node)
            if frame is None or frame[2] * frame[3] > 500 * 1100:
                break
            best = None
            for role, f, name in self._controls_in(node):
                d = rect_distance(pos, f)
                if d <= self.zone and (best is None or d < best[0]):
                    best = (d, role, f, name)
            if best:
                return best[1], best[2], best[3]
            node = self._attr(node, "AXParent")
        return None

    def _run(self):
        last = None
        while True:
            time.sleep(self.period)
            pos = self.pos
            if pos is None or (last is not None and math.hypot(pos[0] - last[0], pos[1] - last[1]) < 2
                               and self.found and time.monotonic() - self.found[0] < 0.5):
                continue
            last = pos
            try:
                err, el = self.AS.AXUIElementCopyElementAtPosition(self.sys, pos[0], pos[1], None)
                if err != 0 or el is None:
                    self.found = (time.monotonic(), None, None, "", False)
                    continue
                role, frame, name, is_control = self.classify(el)
                if is_control and role == "AXSlider" and frame:
                    frame = self._knob(el, frame)
                elif not is_control and self.zone > 0:
                    near = self._nearby(el, pos)
                    if near:
                        (role, frame, name), is_control = near, True
                self.found = (time.monotonic(), role, frame, name, is_control)
            except Exception:
                self.found = (time.monotonic(), None, None, "", False)

    def verify(self, pos):
        """Right before a grab presses: is the element at exactly this point a slider or knob (or part of one)? One
        Accessibility lookup (~5-30 ms), so a press can never land on a track header or a region by mistake."""
        if not self.ok:
            return True, "no Accessibility"
        try:
            err, el = self.AS.AXUIElementCopyElementAtPosition(self.sys, pos[0], pos[1], None)
            if err != 0 or el is None:
                return False, "nothing"
            role, frame, name, is_control = self.classify(el)
            return (is_control and role in DRAG_ROLES), f"{role} {name[:24]}".strip()
        except Exception as e:
            return False, f"lookup failed ({e})"

    def control_at(self, pos):
        """The frame of the control under or within `zone` px of pos, if Accessibility saw one there recently."""
        f = self.found
        if not f or not f[4] or f[2] is None or time.monotonic() - f[0] > 0.4:
            return None
        x, y, w, h = f[2]
        if w * h > 300 * 300 or rect_distance(pos, f[2]) > self.zone + 2:
            return None
        return f[2]

    def label(self):
        f = self.found
        return f"{f[1] or '-'} {f[3][:24]}".strip() if f else ""


class AbsoluteMap:
    """A box of the (mirrored) camera frame maps onto the whole screen.

    rect = (x0, y0, x1, y1) in normalised raw-frame coordinates. Without one, a centred box
    spanning `box` of the frame is used."""

    def __init__(self, screen_w, screen_h, box=0.6, rect=None):
        self.w, self.h, self.box = screen_w, screen_h, box
        lo = (1 - box) / 2
        self.rect = tuple(rect) if rect else (lo, lo, 1 - lo, 1 - lo)

    def __call__(self, nx, ny):
        x0, y0, x1, y1 = self.rect
        u = min(max((x1 - nx) / (x1 - x0), 0.0), 1.0)  # mirrored x
        v = min(max((ny - y0) / (y1 - y0), 0.0), 1.0)
        return u * (self.w - 1), v * (self.h - 1)


def rect_from_sweep(pts, shrink=0.05, lo_p=0.03, hi_p=0.97):
    """Box covering the hand positions seen while the user swept their comfortable range:
    outliers trimmed, then shrunk by `shrink` of its size per side so the screen edges are
    reachable without stretching. None if the sweep was too small to use."""
    if len(pts) < 10:
        return None
    xs = sorted(p[0] for p in pts)
    ys = sorted(p[1] for p in pts)

    def q(s, p):
        return s[min(len(s) - 1, int(p * len(s)))]

    x0, x1, y0, y1 = q(xs, lo_p), q(xs, hi_p), q(ys, lo_p), q(ys, hi_p)
    sx, sy = (x1 - x0) * shrink, (y1 - y0) * shrink
    x0, x1, y0, y1 = x0 + sx, x1 - sx, y0 + sy, y1 - sy
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        return None
    return (x0, y0, x1, y1)


def px_per_metre(img, world, w, h):
    """How many image pixels one metre of hand spans right now (weak perspective), fitted over all 21
    landmarks: centred image points ≈ s × centred world points (x, y). The world landmarks are the hand's
    shape in metres in the camera's orientation, so a turned hand is foreshortened in both and the fit
    stays right; s = f / distance, so it measures how far the hand is from the camera."""
    n = len(img)
    ix = [p[0] * w for p in img]
    iy = [p[1] * h for p in img]
    wx = [p[0] for p in world]
    wy = [p[1] for p in world]
    mix, miy, mwx, mwy = sum(ix) / n, sum(iy) / n, sum(wx) / n, sum(wy) / n
    num = sum((ix[k] - mix) * (wx[k] - mwx) + (iy[k] - miy) * (wy[k] - mwy) for k in range(n))
    den = sum((wx[k] - mwx) ** 2 + (wy[k] - mwy) ** 2 for k in range(n))
    return num / den if den > 1e-12 else None


class MetricMap:
    """3-D hand position to a steady 2-D cursor (the parallax fix). The anchor's image position is
    turned into where the hand really is, sideways and up/down, in metres, using its distance from the
    camera (px_per_metre): X = (x_px - centre) / s. Moving toward or away from the camera no longer
    slides the cursor, and 1 cm of movement is the same number of pixels near or far. A pad of
    pad_w × pad_h metres around `centre` spans the screen. Across, the centre is the camera's line.
    Down, the pad sits as low as it can while the whole hand stays in the camera's view at the pad's
    bottom edge: the hand's lowest point (learned live, about 6 cm below the base anchor) may reach
    `keep` of the frame height, no further. `recenter` moves the centre to the hand, but never so low
    that the bottom edge leaves the view. (The first-seen position put half the pad below the frame,
    because hands come into view from the bottom: 18 of 22 pinches lost the hand, found live.)
    The distance estimate is smoothed hard: a hand's depth changes slowly."""

    def __init__(self, screen_w, screen_h, pad=(0.24, 0.15), keep=0.95):
        self.w, self.h = screen_w, screen_h
        self.pad = pad
        self.keep = keep
        self.centre = None           # the centre in use, (X, Y) metres
        self.wanted = None           # where "recenter" put it (the vertical part is still kept in view)
        self.below = 0.06            # metres from the anchor down to the hand's lowest point, learned live
        self.bottom = None           # the lowest the pad's bottom edge may sit, metres below the camera line
        self.s = None
        self.recenter_next = False
        self.last = None             # (X, Y) metres, for recenter and logs

    def scale(self, s):
        """Smooths px-per-metre: fast enough to follow a reach, slow enough not to shake the cursor."""
        if s is None or s <= 0:
            return self.s
        self.s = s if self.s is None else self.s + 0.12 * (s - self.s)
        return self.s

    def __call__(self, nx, ny, img, world, cam_w, cam_h):
        s = self.scale(px_per_metre(img, world, cam_w, cam_h) if (img and world) else None)
        if not s:
            return None
        X = -(nx * cam_w - cam_w / 2) / s        # mirrored: the user's right is +X
        Y = (ny * cam_h - cam_h / 2) / s
        self.last = (X, Y)
        if img:
            reach = max(p[1] for p in img) - ny  # how far the hand hangs below the anchor, in frame heights
            self.below += 0.05 * (max(0.0, reach * cam_h / s) - self.below)
        bottom = (self.keep * cam_h - cam_h / 2) / s - self.below
        self.bottom = bottom if self.bottom is None else self.bottom + 0.05 * (bottom - self.bottom)
        lowest = self.bottom - self.pad[1] / 2   # the lowest centre that keeps the bottom edge in view
        if self.recenter_next:
            self.wanted, self.recenter_next = (X, Y), False
        # across: the camera's line (it sits at the top-middle of the screen, so a hand straight in front of
        # it is the screen's middle); the first-seen position put the centre 19 cm off when the hand came in
        # from the side (found live). Down: as low as the view allows, or where "recenter" put it if higher.
        if self.wanted is None:
            self.centre = (0.0, lowest)
        else:
            self.centre = (self.wanted[0], min(self.wanted[1], lowest))
        u = min(max((X - self.centre[0]) / self.pad[0] + 0.5, 0.0), 1.0)
        v = min(max((Y - self.centre[1]) / self.pad[1] + 0.5, 0.0), 1.0)
        return u * (self.w - 1), v * (self.h - 1)

    def distance(self, f_px):
        """Estimated camera-to-hand distance in metres for a focal length in px (for the preview)."""
        return f_px / self.s if self.s else None


FACE_MODEL = os.path.join(HERE, "models", "face_landmarker.task")
EYE_CORNERS = ((33, 133), (362, 263))    # MediaPipe face mesh: the outer and inner corner of each eye
IPD_M = 0.063                            # adult eye spacing, average; the calibration absorbs a user's own
CAMERA_HFOV = 70.0                       # degrees across, MacBook camera (a guess: its error mostly cancels, see AimMap)
AIM_DOTS = ((0.5, 0.5), (0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85))
AIM_CAL_PATH = os.path.expanduser("~/Library/Application Support/DAW Control/aim-calibration.json")
# the hand point the ray passes through. Replayed on the first live session (2026-09-30, 200 half-second windows),
# frame-to-frame jitter of the hit, median / p90 pt: tip 31 / 110, index middle joint (pip) 16 / 78, base (wrist and
# the four knuckles) 9 / 43. A finger pointed at the camera is foreshortened and its tip is hard to place; the base
# moves only with the whole hand, and the calibration absorbs the fixed offset from it to the fingertip. Over the first
# three good calibration rounds (tools/aim_replay.py, each round tested against a fit of the others) pip missed least:
# 196 / 370 pt median / p90, base 279 / 429, tip 372 / 1120. Over all 13 rounds of the day (each fitted to itself)
# the tip with depth smoothing 0.12 fitted best, 123 px (pip 150, base 231), and the user wants the line through the
# tip ("the tip is where it's actually going to be pointing"): the tip is the default.
AIM_THROUGH = {"tip": (INDEX_TIP,), "pip": (6,), "base": (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)}


def eye_centres(face):
    """The two eye centres of a face as normalised (x, y, z), the image's left one first (on the unmirrored frame
    that is the user's right eye). Each is the middle of the eye's two corners: they sit on the skull, so unlike the
    iris they stay put when the eye turns towards a target."""
    pts = [tuple((face[a][k] + face[b][k]) / 2 for k in range(3)) for a, b in EYE_CORNERS]
    return tuple(sorted(pts))


def back_project(u, v, z, cam_w, cam_h, f_px):
    """Camera pixel (u, v) at depth z (m) -> a point in the screen's frame, metres: X to the user's right, Y down,
    Z out of the screen towards the user, origin at the camera. The camera sits in the screen's plane looking
    straight out of it (a laptop lid), so the screen is the plane Z = 0. X is mirrored: the camera faces the user."""
    return (-(u - cam_w / 2) * z / f_px, (v - cam_h / 2) * z / f_px, z)


def aim_hit(eye, tip, min_gap=0.15):
    """Where the line from the eye through the fingertip meets the screen (X, Y metres), or None unless the fingertip
    is at least `min_gap` in front of the eye. A short eye-to-fingertip line magnifies every wobble: with the hand near
    the face (10 % of frames live, gap under 13 cm) the aim flew to -3402, 4389 px; then the cursor holds still."""
    gap = eye[2] - tip[2]
    if gap < min_gap or tip[2] <= 0:
        return None
    k = eye[2] / gap
    return (eye[0] + (tip[0] - eye[0]) * k, eye[1] + (tip[1] - eye[1]) * k)


def load_aim_cal(sw, sh, path=AIM_CAL_PATH):
    """The saved calibration's map (a, b, c, d) for this screen size, or None. The camera and the screen move
    together, so it stays good until the display resolution or the person changes."""
    try:
        import json
        with open(path) as f:
            d = json.load(f)
        return tuple(d["map"]) if d.get("screen") == [sw, sh] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_aim_cal(sw, sh, fit, eye, path=AIM_CAL_PATH, rounds=None, through=None):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"screen": [sw, sh], "map": list(fit["map"]), "rms_px": round(fit["rms"], 1), "eye": eye,
                   "dots": fit["dots"], "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "through": through,
                   "rounds": rounds or []}, f, indent=1)


def load_aim_rounds(sw, sh, through, path=AIM_CAL_PATH):
    """The saved calibration rounds, each a list of [dot x, dot y, hit X, hit Y], taken with this screen and ray."""
    try:
        import json
        with open(path) as f:
            d = json.load(f)
        if d.get("screen") != [sw, sh] or d.get("through") != through:
            return []
        return [r for r in d.get("rounds") or [] if isinstance(r, list)]
    except (OSError, ValueError, TypeError):
        return []


POOL_ROUNDS = 6            # the calibration fits over this many recent rounds
POOL_ROUND_MAX_PX = 300    # a round whose own fit misses by more than this (the user moved mid-round) is left out


def pool_rounds(rounds, physical, sw, sh):
    """One fit over every usable recent round (user, live: "six more of these ... to account for people's lack of
    hand-eye coordination"): each round's own aiming error averages out. Rounds that disagree with themselves are left
    out. Returns (fit or None, rounds used, reason)."""
    keep = []
    for r in rounds[-POOL_ROUNDS:]:
        c = AimCalibration(sw, sh)
        c.got = [((q[0], q[1]), (q[2], q[3])) for q in r]
        f, _ = c.fit(physical)
        if f and f["rms"] <= POOL_ROUND_MAX_PX:
            keep.append(r)
    if not keep:
        return None, 0, "no round fits itself within %d px" % POOL_ROUND_MAX_PX
    c = AimCalibration(sw, sh)
    c.got = [((q[0], q[1]), (q[2], q[3])) for r in keep for q in r]
    f, why = c.fit(physical)
    return f, len(keep), why


class AimMap:
    """Aim from the eye through the fingertip (plan, 2026-09-30). People don't point along the finger: they line the
    fingertip up with the target from the eye, and studies of mid-air pointing find that ray more accurate than the
    finger's own direction. The camera sees the face as well as the hand, so both ends of the ray are positions, the
    steady measurement. The eye comes from MediaPipe's face landmarks, its distance from the eyes' spacing in px
    (IPD_M apart); the fingertip from the hand landmarks, its distance from the hand's size (px_per_metre) plus how
    far the tip sits in front of the hand's centre (world z). The ray is followed to the screen's plane, and the
    display's physical size turns metres into points. The camera's focal length is a guess, but it scales both
    depths alike and the hit barely moves. Depths are smoothed hard: a head and a reaching hand change depth slowly,
    and a depth error swings the hit sideways. The five-dot calibration then fits scale and offset each way, which
    absorbs the user's own eye spacing and hand size, the dominant eye and the camera's exact place. When the face
    is out of view (the hand between face and camera, pointing near the top-middle) the last eye is kept."""

    def __init__(self, screen_w, screen_h, mm=None, eye="mid", cal=None, hfov=CAMERA_HFOV, through="tip", depth_k=0.12):
        self.w, self.h = screen_w, screen_h
        # per-frame smoothing of the hand's distance. With the eye above the camera, 1 cm of hand depth moves the hit
        # ~0.5 cm, and the measured depth wandered 23-33 cm while the first calibration held each dot: at 0.12 its dots
        # missed by 207 pt rms after the fit, at 0.02 (~1.7 s) by 119 (replayed)
        self.depth_k = depth_k
        self.through = AIM_THROUGH[through]     # the hand landmarks whose mean the ray passes through
        mm = mm or (345.0, 223.0)                         # MacBook Pro 16" when the display does not say
        self.pitch = (mm[0] / 1000 / screen_w, mm[1] / 1000 / screen_h)    # metres per point
        self.physical = (1 / self.pitch[0], screen_w / 2, 1 / self.pitch[1], 0.0)   # x = a X + b, y = c Y + d
        self.map = tuple(cal) if cal else self.physical
        self.calibrated = bool(cal)
        self.eye, self.hfov = eye, hfov
        self.eye_px = None       # the chosen eye in camera px, smoothed
        self.ipd = None          # the eyes' spacing in camera px (3-D, so a turned head is not foreshortened), smoothed
        self.face_t = None       # when the face was last seen
        self.eyes = None         # this frame's eye centres (normalised), for the preview and the frame record
        self.s = None            # hand px per metre, smoothed
        self.dz = 0.0            # the fingertip's depth relative to the hand's centre (m), smoothed
        self.E = self.F = self.hit = None
        self.recenter_next = False

    def f_px(self, cam_w):
        return (cam_w / 2) / math.tan(math.radians(self.hfov / 2))

    def see_face(self, face, cam_w, cam_h, t):
        """One frame's face landmarks (normalised (x, y, z) list); None when the face was not found."""
        if not face:
            self.eyes = None
            return
        l, r = self.eyes = eye_centres(face)
        ipd = math.sqrt(((l[0] - r[0]) * cam_w) ** 2 + ((l[1] - r[1]) * cam_h) ** 2 + ((l[2] - r[2]) * cam_w) ** 2)
        p = l if self.eye == "right" else r if self.eye == "left" else ((l[0] + r[0]) / 2, (l[1] + r[1]) / 2)
        u, v = p[0] * cam_w, p[1] * cam_h
        stale = self.face_t is None or t - self.face_t > 1.0
        if stale or self.eye_px is None:
            self.eye_px, self.ipd = (u, v), ipd
        else:
            self.eye_px = (self.eye_px[0] + 0.3 * (u - self.eye_px[0]), self.eye_px[1] + 0.3 * (v - self.eye_px[1]))
            self.ipd += 0.08 * (ipd - self.ipd)
        self.face_t = t

    def eye_point(self, cam_w, cam_h):
        if self.eye_px is None or not self.ipd:
            return None
        f = self.f_px(cam_w)
        return back_project(self.eye_px[0], self.eye_px[1], f * IPD_M / self.ipd, cam_w, cam_h, f)

    def __call__(self, img, world, cam_w, cam_h):
        """The screen point the user aims at, clamped to the screen, or None (no face seen yet, no world landmarks,
        or the fingertip not in front of the eye). self.hit keeps the hit in metres for the calibration."""
        self.hit = None
        if not (img and world):
            return None
        s = px_per_metre(img, world, cam_w, cam_h)
        if s and s > 0:
            self.s = s if self.s is None else self.s + self.depth_k * (s - self.s)
        ks = self.through
        dz = sum(world[k][2] for k in ks) / len(ks) - sum(p[2] for p in world) / len(world)
        self.dz += 0.3 * (dz - self.dz)
        E = self.eye_point(cam_w, cam_h)
        if E is None or not self.s:
            return None
        f = self.f_px(cam_w)
        self.E = E
        u, v = sum(img[k][0] for k in ks) / len(ks) * cam_w, sum(img[k][1] for k in ks) / len(ks) * cam_h
        self.F = back_project(u, v, f / self.s + self.dz, cam_w, cam_h, f)
        self.hit = aim_hit(E, self.F)
        if self.hit is None:
            return None
        a, b, c, d = self.map
        if self.recenter_next:                            # where the user aims now becomes the screen's centre
            b, d = self.w / 2 - a * self.hit[0], self.h / 2 - c * self.hit[1]
            self.map, self.recenter_next = (a, b, c, d), False
        x, y = a * self.hit[0] + b, c * self.hit[1] + d
        # not clamped to the screen: a grab drags by the aim's movement, and a fader low on the screen could not be
        # dragged further down once the aim reached the bottom edge (live). The cursor is clamped where it is posted.
        return x, y


class AimCalibration:
    """Five dots, one at a time (the centre, then the corners 15 % in): point at each until it is captured. A dot is
    captured once the uncalibrated hit has held within `spread` m (80 % of the samples) for `hold` s (2 s to get there and a
    1.5 s hold: at 0.8 s and 1 s it caught a hand still on its way, user, live: "I'm not pointing at it ... chill"; 4 cm: at 2 cm the
    first live try captured 1 of 5, the hit of a held hand wobbling 2-10 cm at 70-120 cm from the screen), counted from
    `settle` s after the dot appeared (time to get there); a dot not captured in `timeout` s is skipped. fit() gives
    the scale and offset each way (least squares) and the rms miss in points."""

    def __init__(self, sw, sh, dots=AIM_DOTS, settle=2.0, hold=1.5, spread=0.04, timeout=15.0, min_move=0.02):
        self.sw, self.sh = sw, sh
        self.dots = [(fx * sw, fy * sh) for fx, fy in dots]
        self.settle, self.hold, self.spread, self.timeout = settle, hold, spread, timeout
        # a dot counts only once the aim has left the previous dot's spot by this much (m): live, a hand still resting
        # on the centre dot 0.8 s after the next dot appeared was captured as the top-left corner
        self.min_move = min_move
        self.i, self.t_dot, self.t_last = 0, None, None
        self.win = collections.deque()
        self.got = []            # (dot px, hit m)
        self.skipped = 0
        self.progress = 0.0

    @property
    def current(self):
        return self.dots[self.i] if self.i < len(self.dots) else None

    @property
    def done(self):
        return self.i >= len(self.dots)

    def _next(self, t):
        self.i, self.t_dot, self.progress = self.i + 1, t, 0.0
        self.win.clear()

    def feed(self, t, hit):
        """One frame: hit = the uncalibrated (X, Y) in metres, or None. Returns 'captured', 'skipped' or None."""
        if self.done:
            return None
        if self.t_dot is None:
            self.t_dot = t
        if hit is None and self.t_last is not None:
            # no aim this frame (hand or face out of view): the dot's clock waits, so a dot is never skipped for
            # something the text under it is asking the user to fix (live: all five skipped with no face in view)
            self.t_dot += t - self.t_last
        self.t_last = t
        if t - self.t_dot > self.timeout:
            self.skipped += 1
            self._next(t)
            return "skipped"
        if t - self.t_dot < self.settle or hit is None:
            return None
        self.win.append((t, hit))
        while t - self.win[0][0] > self.hold:
            self.win.popleft()
        xs = sorted(h[0] for _, h in self.win)
        ys = sorted(h[1] for _, h in self.win)
        med = (xs[len(xs) // 2], ys[len(ys) // 2])
        far = sum(1 for _, h in self.win if math.hypot(h[0] - med[0], h[1] - med[1]) > self.spread)
        steady = far <= 0.2 * len(self.win)
        span = t - self.win[0][0]
        self.progress = min(span / self.hold, 1.0) if steady else 0.0
        moved = not self.got or math.hypot(med[0] - self.got[-1][1][0], med[1] - self.got[-1][1][1]) >= self.min_move
        if not moved:
            self.progress = 0.0
        if steady and moved and span >= 0.9 * self.hold and len(self.win) >= 12:
            self.got.append((self.dots[self.i], med))
            self._next(t)
            return "captured"
        if not steady:                                   # restart the hold from the newest sample
            while len(self.win) > 1:
                self.win.popleft()
        return None

    def fit(self, physical):
        """{'map': (a, b, c, d), 'rms': px, 'dots': [...]} from the captured dots, or (None, reason). Needs three dots
        spanning both ways; a scale more than 3x off the display's physical size is refused (bad depths)."""
        if len(self.got) < 3:
            return None, f"only {len(self.got)} of {len(self.dots)} dots captured"

        def line(ps, ss):
            n = len(ps)
            mp, ms = sum(ps) / n, sum(ss) / n
            var = sum((p - mp) ** 2 for p in ps)
            if var < 1e-8:
                return None
            k = sum((p - mp) * (s - ms) for p, s in zip(ps, ss)) / var
            return k, ms - k * mp

        fx = line([h[0] for _, h in self.got], [d[0] for d, _ in self.got])
        fy = line([h[1] for _, h in self.got], [d[1] for d, _ in self.got])
        if fx is None or fy is None:
            return None, "the captured dots do not span the screen both ways"
        (a, b), (c, d) = fx, fy
        for k, k0, name in ((a, physical[0], "across"), (c, physical[2], "down")):
            if not (k0 / 3 <= k <= k0 * 3):
                return None, f"scale {name} is {k / k0:.2f}x the display's, depths look wrong"
        miss = [(a * h[0] + b - dt[0]) ** 2 + (c * h[1] + d - dt[1]) ** 2 for dt, h in self.got]
        rms = math.sqrt(sum(miss) / len(miss))
        return {"map": (a, b, c, d), "rms": rms,
                "dots": [[round(dt[0]), round(dt[1]), round(math.sqrt(m))] for (dt, _), m in zip(self.got, miss)]}, ""


class AimDots:
    """The calibration's dot: a red centre, a white ring, and a green disc that grows while the aim holds steady.
    Its own click-through full-screen window, made when the calibration starts and closed when it ends."""

    def __init__(self):
        import AppKit
        import Quartz
        self.AK, self.Q = AppKit, Quartz
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        screen = AppKit.NSScreen.screens()[0].frame()
        self.sh = int(screen.size.height)
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            screen, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        win.setLevel_(AppKit.NSScreenSaverWindowLevel)
        win.setOpaque_(False)
        win.setBackgroundColor_(AppKit.NSColor.clearColor())
        win.setIgnoresMouseEvents_(True)
        win.setHasShadow_(False)
        win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                   | AppKit.NSWindowCollectionBehaviorStationary)
        root = win.contentView()
        root.setWantsLayer_(True)
        root = root.layer()

        def disc(size, rgba, border=None):
            layer = Quartz.CALayer.layer()
            layer.setBounds_(((0, 0), (size, size)))
            layer.setCornerRadius_(size / 2)
            layer.setBackgroundColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*rgba).CGColor())
            if border:
                layer.setBorderWidth_(4)
                layer.setBorderColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*border).CGColor())
            root.addSublayer_(layer)
            return layer

        self.ring = disc(96, (0, 0, 0, 0.25), border=(1, 1, 1, 0.95))
        self.fill = disc(96, (0.2, 0.95, 0.55, 0.6))
        self.dot = disc(16, (1.0, 0.2, 0.2, 1.0))
        self.text = Quartz.CATextLayer.layer()
        self.text.setFontSize_(22)
        self.text.setAlignmentMode_("center")
        self.text.setContentsScale_(2.0)
        self.text.setBounds_(((0, 0), (760, 30)))
        self.text.setForegroundColor_(AppKit.NSColor.whiteColor().CGColor())
        self.text.setBackgroundColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 0.55).CGColor())
        self.text.setCornerRadius_(6)
        root.addSublayer_(self.text)
        win.orderFrontRegardless()
        self.win = win

    def show(self, at, progress, label):
        Q = self.Q
        Q.CATransaction.begin()
        Q.CATransaction.setDisableActions_(True)
        p = (at[0], self.sh - at[1])
        for layer in (self.ring, self.fill, self.dot):
            layer.setPosition_(p)
        k = max(progress, 0.02)
        self.fill.setTransform_(Q.CATransform3DMakeScale(k, k, 1))
        self.text.setString_(label)
        self.text.setPosition_((p[0], p[1] - 80 if at[1] < self.sh / 2 else p[1] + 80))
        Q.CATransaction.commit()
        Q.CATransaction.flush()

    def pump(self):
        app = self.AK.NSApp
        while True:
            ev = app.nextEventMatchingMask_untilDate_inMode_dequeue_(self.AK.NSEventMaskAny, None, self.AK.NSDefaultRunLoopMode, True)
            if ev is None:
                break
            app.sendEvent_(ev)

    def close(self):
        self.win.orderOut_(None)
        self.win.close()


class RelativeMap:
    """Air-mouse style: hand motion moves the cursor by gain × normalised motion."""

    def __init__(self, screen_w, screen_h, gain=2.5, deadzone=0.0008):
        self.w, self.h, self.gain, self.deadzone = screen_w, screen_h, gain, deadzone
        self.prev = None
        self.x, self.y = screen_w / 2, screen_h / 2

    def __call__(self, nx, ny):
        nx = 1 - nx  # mirrored
        if self.prev is not None:
            dx, dy = nx - self.prev[0], ny - self.prev[1]
            if math.hypot(dx, dy) > self.deadzone:
                self.x = min(max(self.x + dx * self.gain * self.w, 0), self.w - 1)
                self.y = min(max(self.y + dy * self.gain * self.w, 0), self.h - 1)
        self.prev = (nx, ny)
        return self.x, self.y

    def lost(self):
        self.prev = None


class Presence:
    """Decides when a missing hand counts as gone. MediaPipe drops the hand for a frame or two
    hundreds of times an hour, most often mid-pinch, and treating each of those as a departure
    reset the filter and cancelled the pinch. For `hold_s` after the last sighting the tracker
    coasts instead (filters, pinch, button and cursor keep their state); only when the hold
    expires is the hand lost, once."""

    def __init__(self, hold_s=0.18):
        self.hold_s = hold_s
        self.last_seen = None
        self.gone = True  # nothing to lose before the first sighting, or after "lost" was reported

    def update(self, hand, t):
        """'tracked' | 'held' (short gap, coast) | 'lost' (once, when the hold expires) | 'gone'."""
        if hand:
            self.last_seen, self.gone = t, False
            return "tracked"
        if self.gone:
            return "gone"
        if t - self.last_seen <= self.hold_s:
            return "held"
        self.gone = True
        return "lost"


class Cursor:
    """Posts cursor moves and clicks with Quartz. Nothing else touches the cursor."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.down = False
        self.Q = __import__("Quartz") if enabled else None  # fails early if the binding is missing

    def screen(self):
        if not self.Q:
            return 1728, 1117
        b = self.Q.CGDisplayBounds(self.Q.CGMainDisplayID())
        return int(b.size.width), int(b.size.height)

    def screen_mm(self):
        """The main display's physical size in mm, or None if it does not say (aim mode's metres to points)."""
        try:
            Q = self.Q or __import__("Quartz")
            s = Q.CGDisplayScreenSize(Q.CGMainDisplayID())
            return (s.width, s.height) if s.width > 50 and s.height > 50 else None
        except Exception:
            return None

    def _post(self, kind, x, y, button=0):
        ev = self.Q.CGEventCreateMouseEvent(None, kind, (x, y), button)
        self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)

    def move(self, x, y):
        if not self.enabled:
            return
        kind = self.Q.kCGEventLeftMouseDragged if self.down else self.Q.kCGEventMouseMoved
        self._post(kind, x, y)

    def press(self, x, y):
        if not self.enabled:
            return
        self._post(self.Q.kCGEventLeftMouseDown, x, y)
        self.down = True

    def release(self, x, y):
        if not self.enabled:
            return
        self._post(self.Q.kCGEventLeftMouseUp, x, y)
        self.down = False

    def drag(self, x, y, dx, dy):
        """A drag with the delta fields set too: knobs that hide and warp the cursor read the deltas."""
        if not self.enabled:
            return
        ev = self.Q.CGEventCreateMouseEvent(None, self.Q.kCGEventLeftMouseDragged, (x, y), 0)
        self.Q.CGEventSetIntegerValueField(ev, self.Q.kCGMouseEventDeltaX, int(round(dx)))
        self.Q.CGEventSetIntegerValueField(ev, self.Q.kCGMouseEventDeltaY, int(round(dy)))
        self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)

    def warp(self, x, y):
        """Puts the cursor back without a move event (the end of a grab)."""
        if not self.enabled:
            return
        self.Q.CGWarpMouseCursorPosition((x, y))
        self.Q.CGAssociateMouseAndMouseCursorPosition(True)


DRAG_ROLES = {"AXSlider", "AXIncrementor", "AXValueIndicator", "AXLevelIndicator"}


class Grab:
    """The grab session (user's design, 2026-09-29): pinch and hold (or close the hand) on a knob or slider, the
    tracker presses the mouse there after `hold_s` (a quick pinch never presses: pinches stay gestures), hand
    movement drags it both ways like a real mouse (`gain` px of drag per px of free cursor movement), and letting go
    releases and puts the cursor back where the grab began. Both ways because each control reads its own direction:
    Logic's knobs and its tempo display turn with vertical drags, the track header's volume slider moves sideways
    (user, live: "we don't have a horizontal drag yet, just vertical drag"). mode: off; drag = only on sliders and
    knobs Accessibility reports; any = anywhere (plug-in GUIs that hide their knobs from Accessibility).
    `hand` is the free cursor position (x, y), or a bare y for a vertical-only caller."""

    def __init__(self, mode="off", gain=0.6, hold_s=0.25):
        self.mode, self.gain, self.hold_s = mode, gain, hold_s
        self.reset()

    def reset(self):
        self.at = self.t0 = self.h0 = None
        self.active = self.allowed = False
        self.dx = self.dy = 0.0

    @staticmethod
    def _xy(hand):
        return (0.0, float(hand)) if not isinstance(hand, (tuple, list)) else (float(hand[0]), float(hand[1]))

    def begin(self, at, t, hand, role):
        self.reset()
        self.at, self.t0, self.h0 = at, t, self._xy(hand)
        self.allowed = self.mode == "any" or (self.mode == "drag" and role in DRAG_ROLES)

    def step(self, t, hand):
        """('press', x, y) once the hold has lasted, then ('drag', x, y, ddx, ddy) as the hand moves; else None."""
        if self.at is None or not self.allowed:
            return None
        hx, hy = self._xy(hand)
        if not self.active:
            if t - self.t0 >= self.hold_s:
                self.active, self.h0, self.dx, self.dy = True, (hx, hy), 0.0, 0.0
                return ("press", self.at[0], self.at[1])
            return None
        dx, dy = (hx - self.h0[0]) * self.gain, (hy - self.h0[1]) * self.gain
        ddx, ddy = dx - self.dx, dy - self.dy
        if abs(ddx) < 0.5 and abs(ddy) < 0.5:
            return None
        self.dx, self.dy = dx, dy
        return ("drag", self.at[0] + dx, self.at[1] + dy, ddx, ddy)

    def end(self):
        """('release', x, y, back_x, back_y) if a press happened, else None; the session is over."""
        r = ("release", self.at[0] + self.dx, self.at[1] + self.dy, self.at[0], self.at[1]) if self.active else None
        self.reset()
        return r


class Sound:
    """System sounds as feedback the user can hear without watching the log."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def play(self, name):
        if not self.enabled:
            return
        path = f"/System/Library/Sounds/{name}.aiff"
        if os.path.exists(path):
            subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Overlay:
    """What the user sees instead of a small arrow (user, live: the cursor "should be like a camera button on an
    iPhone, a round fairly big smooth circle", and every button, knob and fader "should noticeably light up like
    an overlay when hovered"). One transparent, click-through window covers the main display; the ring and the
    highlight are layers inside it, moved with Core Animation in the same step as the cursor and flushed at once.
    (The first version moved and resized two windows every frame: the ring trailed the arrow, user: "a ton of
    latency between the cursor and your simulated cursor", and the frame loop stalled 250-500 ms at times.)
    The highlight is at least `min_tile` px and `pad` px bigger than the control all round (user: "the tiles ...
    could be wider"). Colours: white idle, blue hovered, green locked, amber grabbing."""

    IDLE, HOVER, LOCK, GRAB = (1.0, 1.0, 1.0), (0.25, 0.85, 1.0), (0.2, 0.95, 0.55), (1.0, 0.7, 0.15)
    NUDGE = (0.75, 0.45, 1.0)     # purple: fine-tuning (aim mode, after a pause)

    def __init__(self, size=64, enabled=True, pad=14, min_tile=48):
        self.size, self.enabled, self.pad, self.min_tile = size, enabled, pad, min_tile
        self.win = self.ring = self.glow = None
        self.shown = None
        self.last = None
        if not enabled:
            return
        import AppKit
        import Quartz
        self.AK, self.Q = AppKit, Quartz
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon
        screen = AppKit.NSScreen.screens()[0].frame()
        self.sh = int(screen.size.height)
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            screen, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        win.setLevel_(AppKit.NSScreenSaverWindowLevel)
        win.setOpaque_(False)
        win.setBackgroundColor_(AppKit.NSColor.clearColor())
        win.setIgnoresMouseEvents_(True)
        win.setHasShadow_(False)
        win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                   | AppKit.NSWindowCollectionBehaviorStationary
                                   | AppKit.NSWindowCollectionBehaviorIgnoresCycle)
        view = win.contentView()
        view.setWantsLayer_(True)
        root = view.layer()
        self.glow = Quartz.CALayer.layer()
        self.glow.setCornerRadius_(10)
        self.glow.setBorderWidth_(3)
        self.ring = Quartz.CALayer.layer()
        self.ring.setBounds_(((0, 0), (size, size)))
        self.ring.setCornerRadius_(size / 2)
        self.ring.setBorderWidth_(max(3.0, size / 16))
        for layer in (self.glow, self.ring):
            layer.setHidden_(True)
            root.addSublayer_(layer)
        win.orderFrontRegardless()
        self.win = win
        self.style(self.IDLE, None)

    def _colour(self, rgb, a):
        return self.AK.NSColor.colorWithSRGBRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], a).CGColor()

    def style(self, rgb, fill_alpha):
        key = (rgb, fill_alpha)
        if self.ring is None or key == self.shown:
            return
        self.shown = key
        self.ring.setBorderColor_(self._colour(rgb, 0.95))
        self.ring.setBackgroundColor_(self._colour(rgb, 0.22 if fill_alpha is None else fill_alpha))
        self.glow.setBorderColor_(self._colour(rgb, 0.95))
        self.glow.setBackgroundColor_(self._colour(rgb, 0.20))

    def tile(self, target):
        """The highlight's frame for a control: `pad` bigger all round, at least `min_tile` each way."""
        x, y, w, h = target
        W, H = max(w + 2 * self.pad, self.min_tile), max(h + 2 * self.pad, self.min_tile)
        return (x + w / 2 - W / 2, y + h / 2 - H / 2, W, H)

    def update(self, pos, target=None, locked=False, grabbing=False, visible=True, nudging=False):
        """pos: Quartz (top-left) point; target: the hovered control's (x, y, w, h) or None."""
        if self.ring is None:
            return
        state = None if (not visible or pos is None) else (round(pos[0]), round(pos[1]), target, locked, grabbing, nudging)
        if state == self.last:
            return
        self.last = state
        Q = self.Q
        Q.CATransaction.begin()
        Q.CATransaction.setDisableActions_(True)          # no implicit 0.25 s animations: move now
        if state is None:
            self.ring.setHidden_(True)
            self.glow.setHidden_(True)
        else:
            self.style(self.GRAB if grabbing else self.LOCK if locked else self.HOVER if target else
                       self.NUDGE if nudging else self.IDLE, 0.45 if (grabbing or nudging) else None)
            self.ring.setPosition_((pos[0], self.sh - pos[1]))
            self.ring.setHidden_(False)
            if target:
                x, y, w, h = self.tile(target)
                self.glow.setFrame_(((x, self.sh - y - h), (w, h)))
                self.glow.setHidden_(False)
            else:
                self.glow.setHidden_(True)
        Q.CATransaction.commit()
        Q.CATransaction.flush()                            # hand it to the render server now, not at the next run loop

    def pump(self):
        if self.win is None:
            return
        app = self.AK.NSApp
        while True:
            ev = app.nextEventMatchingMask_untilDate_inMode_dequeue_(self.AK.NSEventMaskAny, None, self.AK.NSDefaultRunLoopMode, True)
            if ev is None:
                break
            app.sendEvent_(ev)

    def close(self):
        if self.win is not None:
            self.win.orderOut_(None)
            self.win.close()
            self.win = self.ring = self.glow = None


class Marker:
    """A large black square with a thick white border that follows the pointer, in a borderless,
    click-through window above everything else. For filming latency in slow motion: the frame in
    which the square first moves is unambiguous, where the small cursor is not. It follows the
    computed position even with --no-cursor, so the real cursor can be left alone."""

    def __init__(self, size=160, enabled=True):
        self.size, self.enabled = size, enabled
        self.win = None
        if not enabled:
            return
        import AppKit
        import Quartz
        from Foundation import NSMakeRect
        self.AK, self.Q = AppKit, Quartz
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon
        # AppKit's origin is the bottom-left of the primary display; Quartz's is the top-left.
        self.sh = int(AppKit.NSScreen.screens()[0].frame().size.height)
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, size, size), AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered, False)
        win.setLevel_(AppKit.NSScreenSaverWindowLevel)
        win.setOpaque_(False)
        win.setBackgroundColor_(AppKit.NSColor.clearColor())
        win.setIgnoresMouseEvents_(True)  # clicks go through to whatever is underneath
        win.setHasShadow_(False)
        win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                   | AppKit.NSWindowCollectionBehaviorStationary)
        view = win.contentView()
        view.setWantsLayer_(True)
        layer = view.layer()
        layer.setBackgroundColor_(AppKit.NSColor.blackColor().CGColor())
        layer.setBorderColor_(AppKit.NSColor.whiteColor().CGColor())
        layer.setBorderWidth_(size / 8)
        win.orderFrontRegardless()
        self.win = win
        self.move(self.sh / 2, self.sh / 2)
        Quartz.CATransaction.flush()
        self.pump()

    @staticmethod
    def origin(x, y, size, screen_h):
        """Window origin (AppKit, bottom-left) that centres a `size` square on Quartz point (x, y)."""
        return x - size / 2, screen_h - y - size / 2

    def pump(self):
        """Drain pending AppKit events so the window keeps up without a run loop of its own."""
        app = self.AK.NSApp
        while True:
            ev = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                self.AK.NSEventMaskAny, None, self.AK.NSDefaultRunLoopMode, True)
            if ev is None:
                break
            app.sendEvent_(ev)

    def move(self, x, y):
        if not self.win:
            return
        self.win.setFrameOrigin_(self.origin(x, y, self.size, self.sh))
        self.pump()


def percentile(a, p):
    if not a:
        return 0.0
    s = sorted(a)
    return s[min(len(s) - 1, int(p * len(s)))]


def median_smooth(pts, k=5):
    """Per-axis running median of (x, y) points over k frames: the hand's path without the
    landmark jitter, for deciding whether it is at rest."""
    h = k // 2
    out = []
    for i in range(len(pts)):
        xs = sorted(p[0] for p in pts[max(0, i - h):i + h + 1])
        ys = sorted(p[1] for p in pts[max(0, i - h):i + h + 1])
        out.append((xs[len(xs) // 2], ys[len(ys) // 2]))
    return out


def still_windows(raw, win=15, still_px=15.0):
    """Start indices of the non-overlapping `win`-frame windows in which the median-smoothed raw
    point stayed within `still_px` on both axes: the hand was at rest. Judging stillness on the
    filtered point would depend on the filter being measured, and a speed threshold on the raw
    point trips on its own jitter (5 px rms looks like 100-300 px/s at 30 fps)."""
    sm = median_smooth(raw)
    starts = []
    for s in range(0, len(raw) - win + 1, win):
        xs = [p[0] for p in sm[s:s + win]]
        ys = [p[1] for p in sm[s:s + win]]
        if max(xs) - min(xs) <= still_px and max(ys) - min(ys) <= still_px:
            starts.append(s)
    return starts


def jitter_rms(pts):
    """RMS distance of points from their mean."""
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    return math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in pts) / n)


def parse_pair(s):
    try:
        lo, hi = (float(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected lo,hi")
    if not 0 <= lo < hi:
        raise argparse.ArgumentTypeError("expected 0 <= lo < hi")
    return (lo, hi)


def parse_rect(s):
    try:
        x0, y0, x1, y1 = (float(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("rect is x0,y0,x1,y1 in 0..1 of the raw camera frame")
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise argparse.ArgumentTypeError("rect needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(MAPS), default="abs",
                    help="abs: a box of the camera image spans the screen; rel: air mouse; metric: the hand's real "
                         "position in cm, corrected for its distance from the camera (no parallax), --pad spans the screen; "
                         "aim: the line from your eye through your index fingertip, followed to the screen (adds the face "
                         "tracker; calibrate once with --aim-calibrate)")
    ap.add_argument("--aim-calibrate", action="store_true",
                    help="aim: start with the five-dot calibration (point at each dot until its disc fills); saved for "
                         "next time. The control line `calibrate` starts it later")
    ap.add_argument("--aim-depth", type=float, default=0.12,
                    help="aim: per-frame smoothing of the hand's distance (0.12; 0.02 is slower and steadier)")
    ap.add_argument("--aim-through", choices=["base", "pip", "tip"], default="tip",
                    help="aim: the hand point the ray passes through: pip (the index finger's middle joint: 2x steadier than "
                         "the tip and the best calibrations, measured live), base (wrist and knuckles, steadiest, follows "
                         "finger aiming less) or tip (the fingertip itself)")
    ap.add_argument("--trigger", choices=["pinch", "thumb"], default="pinch",
                    help="aim: what grabs and holds a lock: pinch (thumb to index tip; moves the aim) or thumb (press the "
                         "thumb against the side of the hand, a finger gun: the index, and the aim, stay put); with thumb, "
                         "a pinch or a closed hand no longer grabs")
    ap.add_argument("--tap", choices=["on", "off"], default="on",
                    help="aim with --trigger thumb: a quick trigger tap (under 0.4 s) that did not grab clicks at the cursor")
    ap.add_argument("--thumb-on", type=float, default=0.015, help="thumb trigger presses below this thumb-to-hand distance (m)")
    ap.add_argument("--thumb-off", type=float, default=0.03,
                    help="thumb trigger: the thumb counts as raised (arming the trigger, or letting go) above this (m)")
    ap.add_argument("--face-prep", choices=["colour", "hand"], default="colour",
                    help="aim: the image the face tracker sees: colour (the camera's own, full size) or hand (the hand's "
                         "prepared image, e.g. half-size grey)")
    ap.add_argument("--eye", choices=["mid", "left", "right"], default="mid",
                    help="aim: which eye the ray starts from (mid = between them; the calibration absorbs the dominant eye)")
    ap.add_argument("--pad", type=parse_pair, default=(24.0, 15.0), metavar="W,H",
                    help="metric: centimetres of hand movement that span the screen, across and down")
    ap.add_argument("--keep", type=float, default=0.95,
                    help="metric: how far down the camera frame (0..1) the hand's lowest point may reach at the pad's "
                         "bottom edge; the pad sits as low as this allows, so the whole screen is reachable in view")
    ap.add_argument("--point", choices=list(POINTS), default="tip",
                    help="what drives the cursor: tip (index fingertip), thumb (thumb middle joint: the index can grab "
                         "without moving it), thumbtip, web (thumb base/index knuckle), palm")
    ap.add_argument("--lock-ms", type=float, default=0,
                    help="hover lock: after resting this long the cursor locks in place until the hand moves more "
                         "than --breakout px (0 = off; a pinch always locks for the length of the grab)")
    ap.add_argument("--breakout", type=float, default=50, help="px the hand must move to leave a hover lock")
    ap.add_argument("--lock-speed", type=float, default=120, help="px/s under which the pointer counts as resting")
    ap.add_argument("--lock-anywhere", action="store_true",
                    help="hover lock wherever the pointer rests (default: only on controls Accessibility reports)")
    ap.add_argument("--grab", choices=["off", "drag", "any"], default="off",
                    help="pinch-and-hold drags the knob or slider under the cursor (drag) or anything (any); a quick "
                         "pinch never presses")
    ap.add_argument("--grab-gain", type=float, default=0.6, help="px of drag per px of hand movement while grabbing")
    ap.add_argument("--grip", choices=["on", "off"], default="off",
                    help="a closed hand (every finger curled) grabs like a held pinch (user, live: \"just grab it\")")
    ap.add_argument("--zone", type=float, default=0.0,
                    help="px: a control this close to the cursor counts as under it (bigger targets); 0 = only on it")
    ap.add_argument("--ring", type=int, default=0, metavar="PX",
                    help="draw a round cursor of this diameter and light up the control under it (0 = off)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30,
                    help="camera rate to pin. The built-in camera does 15-30; asking for more makes it report 15 "
                         "and lets a dim room drop it to 15 fps with 66 ms exposures")
    ap.add_argument("--box", type=float, default=0.6, help="abs: fraction of the frame (centred) that spans the screen")
    ap.add_argument("--rect", type=parse_rect, help="abs: x0,y0,x1,y1 of the raw frame that spans the screen (overrides --box)")
    ap.add_argument("--calibrate", type=float, default=0,
                    help="abs: for this many seconds after a hand first appears, sweep your comfortable range; "
                         "that range becomes the box (printed as I,rect,... for --rect next time)")
    ap.add_argument("--gain", type=float, default=2.5, help="rel: screen widths per frame width of motion")
    ap.add_argument("--mincutoff", type=float, default=3.0,
                    help="1€ filter: cutoff when still (Hz); its time constant 1/(2*pi*f) is the lag of a move from rest "
                         "(53 ms at 3 Hz, 159 ms at the old 1 Hz). See tune_filter.py")
    ap.add_argument("--beta", type=float, default=0.05,
                    help="1€ filter: how fast the cutoff opens with speed (Hz per px/s); 0.05 halves the lag at "
                         "speed against 0.02 for 0.7 px more jitter")
    ap.add_argument("--pinch-on", type=float, default=0.035, help="pinch closes below this thumb-index distance (m, world landmarks)")
    ap.add_argument("--pinch-off", type=float, default=0.055, help="pinch opens above this distance (m)")
    ap.add_argument("--heisenberg-ms", type=float, default=80, help="click where the cursor was this long before the pinch")
    ap.add_argument("--hold-ms", type=float, default=180,
                    help="coast through a hand loss shorter than this (5 frames, measured from the last sighting): "
                         "no filter reset, pinch and button keep their state. 0 = every missing frame is a loss")
    ap.add_argument("--lead-ms", type=float, default=0,
                    help="predict this far ahead along the filtered velocity while the hand moves (gated by speed, "
                         "so a still hand is not made jittery); buys back latency, costs overshoot at stops")
    ap.add_argument("--lead-gate", type=parse_pair, default=(150.0, 500.0), metavar="LO,HI",
                    help="filtered speeds (px/s) between which the lead ramps from nothing to all of it")
    ap.add_argument("--still-px", type=float, default=15,
                    help="the hand counts as still for a half-second window when its median-smoothed raw point "
                         "stays within this many px (the S line's still/jitter fields)")
    ap.add_argument("--presence", type=float, default=0.4, help="MediaPipe min hand presence confidence (lower = fewer drop-outs)")
    ap.add_argument("--tracking", type=float, default=0.4, help="MediaPipe min tracking confidence")
    ap.add_argument("--no-cursor", action="store_true", help="measure only; do not move the cursor")
    ap.add_argument("--click", action="store_true",
                    help="make a pinch press and release the mouse button (default: pinch is a gesture only)")
    ap.add_argument("--quiet", action="store_true", help="no sounds on pinch / calibration")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--marker", type=int, nargs="?", const=160, metavar="PX",
                    help="show a big black/white square (default 160 px) that follows the pointer, "
                         "for filming latency in slow motion; works with --no-cursor too")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--log", help="CSV of every frame")
    ap.add_argument("--dump", help="directory: at each drop-out save the last frame with a hand and the first without")
    ap.add_argument("--frames", metavar="PATH", help="write every frame's landmarks as one JSON line to PATH ('-' = stdout), for the gesture engine")
    ap.add_argument("--diag", type=int, default=0, metavar="PORT",
                    help="send every frame's pointer pipeline (image anchor, hand in cm, distance, each filter stage, "
                         "lock, grab) as UDP JSON to 127.0.0.1:PORT, for the live diagnostic page (trace/diag.html)")
    ap.add_argument("--hands", type=int, default=1, help="hands to track (2 for two-hand gestures; costs detect time)")
    ap.add_argument("--enhance", action="store_true", help="same as --prep enhance")
    ap.add_argument("--prep", choices=list(PREPS), default="none",
                    help="before detection: none; enhance (grey-world colour + CLAHE); gray (greyscale + CLAHE, "
                         "best in a dim coloured room). The preview shows the prepared image")
    ap.add_argument("--precision", type=float, default=1.0,
                    help="slow hand movement moves the cursor this fraction (0.35 = a third; 1 = off): steadier aiming, "
                         "the cursor catches back up with the hand on quick moves")
    ap.add_argument("--nudge", type=float, default=0.0, metavar="GAIN",
                    help="aim: after the hand rests --nudge-dwell ms the cursor moves GAIN of the aim's movement (fine-tune "
                         "the last few cm); a quick move snaps it back onto the aim. 0 = off")
    ap.add_argument("--nudge-dwell", type=float, default=300, metavar="MS", help="aim: rest this long before nudging starts")
    ap.add_argument("--median", type=int, default=1, help="despike: median of the last N anchor points (3 kills single-frame jumps; 1 = off)")
    ap.add_argument("--detection", type=float, default=0.5, help="MediaPipe min hand detection confidence (lower finds a hand sooner in poor light)")
    ap.add_argument("--control", action="store_true",
                    help="read live tuning lines on stdin: 'filter <mincutoff> <beta>', 'lead <ms>', 'cursor on|off'")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    cursor = Cursor(enabled=not args.no_cursor)
    sound = Sound(enabled=not args.quiet)
    marker = Marker(size=args.marker or 160, enabled=bool(args.marker))
    overlay = Overlay(size=args.ring or 64, enabled=bool(args.ring))
    sw, sh = cursor.screen()
    if args.aim_calibrate:
        args.mode = "aim"
    if args.mode == "aim" and not os.path.exists(FACE_MODEL):
        sys.exit(f"aim mode needs the face model at {FACE_MODEL} (run setup.sh)")

    def make_mapper(mode):
        if mode == "metric":
            return MetricMap(sw, sh, (args.pad[0] / 100, args.pad[1] / 100), keep=args.keep)
        if mode == "aim":
            cal = load_aim_cal(sw, sh)
            note = "calibrated" if cal else "not calibrated yet: the control line `calibrate` starts it"
            print(f"I,aim,{note}", file=sys.stderr, flush=True)
            return AimMap(sw, sh, cursor.screen_mm(), eye=args.eye, cal=cal, through=args.aim_through,
                          depth_k=args.aim_depth)
        if mode == "abs":
            return AbsoluteMap(sw, sh, args.box, args.rect)
        return RelativeMap(sw, sh, args.gain)
    mapper = make_mapper(args.mode)
    nonlocal_state = {}           # requests from the control thread, applied between frames by the main loop
    fx, fy = OneEuro(args.mincutoff, args.beta), OneEuro(args.mincutoff, args.beta)
    lock = Lock(args.lock_ms / 1000, args.lock_speed, args.breakout, anywhere=args.lock_anywhere, zone=args.zone)
    grab = Grab(args.grab, args.grab_gain)
    diag = None
    if args.diag:
        import json as _json
        import socket as _socket
        diag = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        diag.setblocking(False)
    f_px = (1280 / 2) / math.tan(math.radians(35))       # MacBook camera: ~70 degrees across (for the distance readout)
    probe = AXProbe(zone=args.zone) if (args.lock_ms or args.grab != "off" or args.ring) else None
    if args.lock_ms and not (probe and probe.ok):
        print("I,warning,no Accessibility bindings (pip install pyobjc-framework-ApplicationServices): hover lock off",
              file=sys.stderr, flush=True)
    despike = Despike(args.median)
    precision = Precision(args.precision)
    nudge = Nudge(args.nudge, args.nudge_dwell / 1000)
    rate = Rate()
    if args.enhance and args.prep == "none":
        args.prep = "enhance"
    if args.control:
        # live tuning from the parent (the gesture engine forwards its file's `pointer` line):
        #   filter <mincutoff> <beta> | lead <ms> | cursor on|off
        import threading

        def control_loop():
            for raw in sys.stdin:
                parts = raw.split()
                try:
                    if parts[:1] == ["filter"] and len(parts) == 3:
                        for f in (fx, fy):
                            f.mincutoff, f.beta = float(parts[1]), float(parts[2])
                    elif parts[:1] == ["lead"] and len(parts) == 2:
                        args.lead_ms = float(parts[1])
                    elif parts[:1] == ["cursor"] and len(parts) == 2:
                        if parts[1] == "on" and cursor.Q is None:
                            cursor.Q = __import__("Quartz")
                        cursor.enabled = parts[1] == "on"
                    elif parts[:1] == ["point"] and len(parts) == 2 and parts[1] in POINTS:
                        args.point = parts[1]
                        fx.reset(); fy.reset(); lock.reset(); despike.reset(); precision.reset()  # a new anchor starts a new track
                    elif parts[:1] == ["prep"] and len(parts) == 2 and parts[1] in PREPS:
                        args.prep = parts[1]
                    elif parts[:1] == ["precision"] and len(parts) == 2:
                        precision.gain = min(max(float(parts[1]), 0.05), 1.0)
                        precision.reset()
                    elif parts[:1] == ["tap"] and len(parts) == 2 and parts[1] in ("on", "off"):
                        args.tap = parts[1]
                    elif parts[:1] == ["trigger"] and len(parts) in (2, 4) and parts[1] in ("pinch", "thumb"):
                        args.trigger = parts[1]
                        if len(parts) == 4:
                            thumb.on_thr, thumb.off_thr = float(parts[2]), float(parts[3])
                        thumb.reset()
                    elif parts[:1] == ["nudge"] and len(parts) in (2, 3):
                        g = 0.0 if parts[1] == "off" else float(parts[1])
                        nudge.gain = min(max(g, 0.0), 1.0)
                        if len(parts) == 3:
                            nudge.dwell = float(parts[2]) / 1000
                        nudge.reset()
                    elif parts[:1] == ["median"] and len(parts) == 2:
                        despike.n = max(1, int(parts[1]))
                    elif parts[:1] == ["map"] and len(parts) == 2 and parts[1] in MAPS:
                        if parts[1] == "aim" and not os.path.exists(FACE_MODEL):
                            raise ValueError(f"no face model at {FACE_MODEL} (run setup.sh)")
                        nonlocal_state["mapper"] = parts[1]
                    elif parts[:1] == ["calibrate"] and len(parts) == 1:
                        if not os.path.exists(FACE_MODEL):
                            raise ValueError(f"no face model at {FACE_MODEL} (run setup.sh)")
                        nonlocal_state["calibrate"] = True
                    elif parts[:1] == ["aimdepth"] and len(parts) == 2 and 0.0 < float(parts[1]) <= 1.0:
                        if isinstance(mapper, AimMap):
                            mapper.depth_k = float(parts[1])
                    elif parts[:1] == ["through"] and len(parts) == 2 and parts[1] in AIM_THROUGH:
                        args.aim_through = parts[1]
                        if isinstance(mapper, AimMap):
                            mapper.through = AIM_THROUGH[parts[1]]
                    elif parts[:1] == ["eye"] and len(parts) == 2 and parts[1] in ("mid", "left", "right"):
                        args.eye = parts[1]
                        if isinstance(mapper, AimMap):
                            mapper.eye, mapper.face_t = parts[1], None
                    elif parts[:1] == ["pad"] and len(parts) == 3:
                        args.pad = (float(parts[1]), float(parts[2]))
                        nonlocal_state["pad"] = True
                    elif parts[:1] == ["keep"] and len(parts) == 2:
                        args.keep = min(max(float(parts[1]), 0.5), 1.0)
                        if isinstance(mapper, MetricMap):
                            mapper.keep = args.keep
                    elif parts[:1] == ["recenter"]:
                        nonlocal_state["recenter"] = True
                    elif parts[:1] == ["lock"] and len(parts) in (3, 4):
                        lock.dwell, lock.breakout = float(parts[1]) / 1000, float(parts[2])
                        if len(parts) == 4:
                            lock.still = float(parts[3])
                        lock.reset()
                    elif parts[:1] == ["grab"] and len(parts) in (2, 3) and parts[1] in ("off", "drag", "any"):
                        grab.mode = parts[1]
                        if len(parts) == 3:
                            grab.gain = float(parts[2])
                    elif parts[:1] == ["zone"] and len(parts) == 2:
                        args.zone = min(max(float(parts[1]), 0.0), 150.0)
                        lock.zone = args.zone
                        if probe is not None:
                            probe.zone, probe.cache = args.zone, {}
                    elif parts[:1] == ["grip"] and len(parts) == 2 and parts[1] in ("on", "off"):
                        args.grip = parts[1]
                    elif parts[:1] == ["ring"] and len(parts) == 2 and (parts[1] in ("on", "off") or parts[1].isdigit()):
                        nonlocal_state["ring"] = 0 if parts[1] == "off" else (64 if parts[1] == "on" else int(parts[1]))
                    elif parts[:1] == ["lockon"] and len(parts) == 2 and parts[1] in ("controls", "anywhere"):
                        lock.anywhere = parts[1] == "anywhere"
                        lock.reset()
                    else:
                        raise ValueError(raw.strip())
                    print(f"I,control,{raw.strip()}", file=sys.stderr, flush=True)
                except (ValueError, IndexError) as e:
                    print(f"I,control,ignored,{e}", file=sys.stderr, flush=True)
        threading.Thread(target=control_loop, name="control", daemon=True).start()
    pinch = Pinch(on=args.pinch_on, off=args.pinch_off, min_frames=2, off_frames=3)
    grip = Pinch(on=0.72, off=0.85, min_frames=3, off_frames=3)    # on openness: every finger curled = a closed hand
    thumb = ThumbTrigger(on=args.thumb_on, off=args.thumb_off)          # the finger gun's trigger (aim mode)
    held_was = False              # a pinch or (with --grip on) a closed hand: what grabs and holds a lock
    grab_state = None             # "pending" | "active" | None, sent with each frame so the gesture engine stands down
    pending_frame = None
    hover = None                  # the control highlighted last frame (sticky, see STICKY_PX)
    tap = None                    # the finger gun's press in progress: {"t", "grabbed"} (a quick one clicks)
    presence = Presence(hold_s=args.hold_ms / 1000)
    history = collections.deque(maxlen=64)  # (t, x, y) for the Heisenberg offset
    hand_hist = collections.deque(maxlen=64)  # (t, pinch m, openness) alongside it, for rewind_point
    calibrating = args.mode == "abs" and args.calibrate > 0
    cal_pts, cal_until = [], None
    facer = None                  # aim mode: the face landmarker, made the first time it is needed
    aimcal = dots = None          # aim mode: the five-dot calibration and its dot while it runs
    face_ms, face_seen, face_frames, aim_report = [], 0, 0, 0.0
    if args.aim_calibrate:
        nonlocal_state["calibrate"] = True

    def open_camera():
        c = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        c.set(cv2.CAP_PROP_FPS, args.fps)  # (CAP_PROP_BUFFERSIZE is not supported by the AVFoundation backend)
        return c

    cap = open_camera()
    read_fails = 0                # consecutive failed reads; the camera is reopened after READ_FAILS_REOPEN
    if not cap.isOpened():
        sys.exit("camera %d did not open (Camera permission for this terminal?)" % args.camera)
    cw, ch, cfps = cap.get(3), cap.get(4), cap.get(5)
    # OpenCV's rate read-back right after opening is unreliable (15 on one run, 30 on the next, both
    # delivering 30), so it is only labelled; the measured rate is what the S line and the warning use.
    print(f"I,camera,{int(cw)}x{int(ch)},{cfps:.0f}fps-reported,screen,{sw}x{sh},mode,{args.mode},point,{args.point},"
          f"pinch,{args.pinch_on}/{args.pinch_off},presence,{args.presence},tracking,{args.tracking},"
          f"hold,{args.hold_ms:g}ms,lead,{args.lead_ms:g}ms,filter,{args.mincutoff:g}/{args.beta:g}",
          file=sys.stderr, flush=True)
    pinned = rate_warned = False  # the rate is pinned again once frames flow: set before the session runs, it did not always stick
    if args.mode == "abs":
        print("I,rect,%.3f,%.3f,%.3f,%.3f" % mapper.rect, file=sys.stderr, flush=True)

    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=args.model, delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
        min_hand_detection_confidence=args.detection,
        min_hand_presence_confidence=args.presence,
        min_tracking_confidence=args.tracking,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    def make_facer():
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=FACE_MODEL, delegate=mpt.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))
    frames_out = FrameWriter(args.frames, cw, ch) if args.frames else None
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    log_file = open(args.log, "w", newline="") if args.log else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["t_frame", "detect_ms", "post_ms", "hand", "raw_x", "raw_y", "x", "y",
                      "pinch_ratio", "pinch_m", "pinched", "held"])

    last_hand_frame, dumps = None, 0
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    t0 = time.perf_counter()
    last_report = t0
    det_ms, loop_ms, frames, hands = [], [], 0, 0
    sec_raw, sec_pos = [], []  # this second's tracked points (raw and shown) for the stillness report
    last_pos = None
    last_raw = None
    held_gaps = lost_n = 0
    prev_state = "gone"
    try:
        while True:
            t_loop = time.perf_counter()
            if nonlocal_state:
                want = nonlocal_state.pop("mapper", None)
                if nonlocal_state.pop("calibrate", None):
                    want = "aim"
                    if aimcal is None:
                        aimcal = AimCalibration(sw, sh)
                        dots = dots or AimDots()
                        lock.reset()
                        r = grab.end()
                        if r:
                            cursor.release(r[1], r[2])
                            cursor.warp(r[3], r[4])
                        overlay.update(None, visible=False)
                        print("I,aimcal,start,point at each dot until its disc fills", file=sys.stderr, flush=True)
                if want and want != args.mode:
                    args.mode = want
                    mapper = make_mapper(want)
                    fx.reset(); fy.reset(); despike.reset(); lock.reset(); precision.reset()
                    last_raw = None
                if aimcal is not None and not isinstance(mapper, AimMap):   # switched away mid-calibration
                    print(f"I,aimcal,cancelled,map {args.mode}", file=sys.stderr, flush=True)
                    aimcal = None
                    dots.close()
                    dots = None
                if isinstance(mapper, AimMap) and nonlocal_state.pop("recenter", None):
                    mapper.recenter_next = True
                    lock.reset()
                    sound.play("Glass")
                if nonlocal_state.pop("pad", None) and isinstance(mapper, MetricMap):
                    mapper.pad = (args.pad[0] / 100, args.pad[1] / 100)
                ring = nonlocal_state.pop("ring", None)
                if ring is not None:                 # AppKit windows are made and closed on this (the main) thread
                    overlay.close()
                    overlay = Overlay(size=ring or 64, enabled=bool(ring))
                if nonlocal_state.pop("recenter", None) and isinstance(mapper, MetricMap):
                    mapper.recenter_next = True
                    lock.reset()
                    sound.play("Glass")
            t_mark = time.perf_counter()
            ok, frame = cap.read()
            t_frame = time.perf_counter()
            slow = [("camera", t_frame - t_mark)]
            if not ok:
                read_fails += 1
                if read_fails == 1 or read_fails % 100 == 0:
                    print(f"I,camera read failed,{read_fails}", file=sys.stderr, flush=True)
                if read_fails % READ_FAILS_REOPEN == 0:
                    # the camera stopped delivering mid-session (found live: 341 failed reads until a restart); reopen it
                    print("I,camera,reopening", file=sys.stderr, flush=True)
                    cap.release()
                    time.sleep(0.5)
                    cap = open_camera()
                    pinned = False
                time.sleep(0.01)
                continue
            if read_fails:
                print(f"I,camera,frames again after {read_fails} failed reads", file=sys.stderr, flush=True)
                read_fails = 0
            if not pinned:
                cap.set(cv2.CAP_PROP_FPS, args.fps)
                pinned = True
            frames += 1
            fps_now = rate.tick(t_frame)
            camera_frame = frame              # the face tracker gets the camera's own image (see below)
            if args.prep == "enhance":
                frame = enhance(frame, clahe)
            elif args.prep == "gray":
                frame = gray_prep(frame, clahe)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(img, int((t_frame - t0) * 1000))
            t_det = time.perf_counter()
            slow.append(("detect", t_det - t_frame))
            det_ms.append((t_det - t_frame) * 1000)

            hand = bool(res.hand_landmarks)
            state = presence.update(hand, t_det)
            if frames_out:
                pending_frame = (res, state, t_frame, t_det)
            if hand and prev_state == "held":
                held_gaps += 1
            prev_state = state
            if args.dump and dumps < 30:
                if hand:
                    last_hand_frame = frame
                elif last_hand_frame is not None:
                    stamp = f"{t_frame - t0:06.2f}"
                    cv2.imwrite(os.path.join(args.dump, f"{stamp}_last.jpg"), last_hand_frame)
                    cv2.imwrite(os.path.join(args.dump, f"{stamp}_lost.jpg"), frame)
                    last_hand_frame, dumps = None, dumps + 1
            raw = pos = q = free = smooth = aim_hit_now = None
            nx = ny = None
            ratio = dist = opened = pressed_m = float("nan")
            event = None
            if hand:
                hands += 1
                lm = [(p.x, p.y, p.z) for p in res.hand_landmarks[0]]
                nx, ny = point_of(lm, "tip" if isinstance(mapper, AimMap) else args.point)   # aim: the ray runs through the tip
                ratio = pinch_ratio(lm)
                if res.hand_world_landmarks:
                    dist = pinch_dist([(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]])
                    opened = openness([(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]])
                    pressed_m = thumb_press([(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]])
                if calibrating:
                    if cal_until is None:
                        cal_until = t_det + args.calibrate
                        print(f"I,calibrating,{args.calibrate:g}s,sweep your comfortable range", file=sys.stderr, flush=True)
                    cal_pts.append((nx, ny))
                    if t_det >= cal_until:
                        rect = rect_from_sweep(cal_pts)
                        if rect:
                            mapper = AbsoluteMap(sw, sh, args.box, rect)
                            print("I,rect,%.3f,%.3f,%.3f,%.3f" % rect, file=sys.stderr, flush=True)
                        else:
                            print("I,calibration sweep too small; keeping the default box", file=sys.stderr, flush=True)
                        sound.play("Glass")
                        calibrating = False
                elif aimcal is not None:
                    # calibrating aim: the cursor stays put; only the uncalibrated hit is collected (below)
                    wlm = [(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]] if res.hand_world_landmarks else None
                    mapper(lm, wlm, cw, ch)
                    aim_hit_now = mapper.hit
                else:
                    if isinstance(mapper, MetricMap):
                        wlm = [(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]] if res.hand_world_landmarks else None
                        raw = mapper(nx, ny, lm, wlm, cw, ch) or last_raw or (sw / 2, sh / 2)
                    elif isinstance(mapper, AimMap):
                        wlm = [(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]] if res.hand_world_landmarks else None
                        raw = mapper(lm, wlm, cw, ch) or last_raw or (sw / 2, sh / 2)
                    else:
                        raw = mapper(nx, ny)
                    last_raw = raw
                    q = despike(raw)
                    pos = (fx(q[0], t_det), fy(q[1], t_det))
                    if args.lead_ms:
                        ahead = lead_gain(math.hypot(fx.dx, fy.dx), *args.lead_gate) * args.lead_ms / 1000
                        pos = (fx.lead(ahead), fy.lead(ahead))
                    smooth = pos
                    if isinstance(mapper, AimMap) and nudge.enabled:
                        pos = nudge(pos, t_det, freeze=(thumb.closed if args.trigger == "thumb" else
                                                        pinch.closed or (args.grip == "on" and grip.closed)))
                    else:
                        pos = precision(pos, t_det)
                    history.append((t_det, pos[0], pos[1]))
                    hand_hist.append((t_det, 1.0 if math.isnan(dist) else dist, 1.0 if math.isnan(opened) else opened,
                                      1.0 if math.isnan(pressed_m) else pressed_m))
                    if not math.isnan(dist):
                        event = pinch.update(dist)
                    gevent = grip.update(opened) if not math.isnan(opened) else None
                    if gevent:
                        print(f"I,grip,{gevent}", file=sys.stderr, flush=True)
                    tevent = thumb.update(pressed_m, t_det) if not math.isnan(pressed_m) else None
                    if tevent:
                        print(f"I,trigger,{tevent},{pressed_m:.3f}", file=sys.stderr, flush=True)
                        if tevent == "down":
                            sound.play("Tink")
                            tap = {"t": t_det, "grabbed": False}
                    thumb_mode = isinstance(mapper, AimMap) and args.trigger == "thumb"
                    held = thumb.closed if thumb_mode else (pinch.closed or (args.grip == "on" and grip.closed))
                    held_down = held and not held_was
                    held_was = held
                    if 0 < raw[0] < sw - 1 and 0 < raw[1] < sh - 1:  # clamped at an edge is not "still"
                        sec_raw.append(raw)
                        sec_pos.append(pos)
                    last_pos = pos
                    # Heisenberg: the pinch itself nudges the finger; use the position from before it.
                    tb = t_det - args.heisenberg_ms / 1000
                    click_at = next(((x, y) for (t, x, y) in history if t >= tb), pos)
                    if held_down and isinstance(mapper, AimMap):
                        # the pinch moved the fingertip and the aim with it: grab where the cursor was before it began
                        rw = rewind_point(history, hand_hist, t_det, what="thumb" if thumb_mode else (
                            "grip" if (not pinch.closed and args.grip == "on" and grip.closed) else "pinch"),
                            span=0.3 if thumb_mode else REWIND_S)
                        if rw is not None and thumb_mode and math.hypot(pos[0] - rw[0], pos[1] - rw[1]) > 250:
                            # the finger gun barely moves the aim: a long way back is where the user aimed before (live:
                            # a 1582 px jump), not where the thumb began to drop
                            rw = None
                        if rw is not None:
                            print(f"I,rewind,{math.hypot(pos[0] - rw[0], pos[1] - rw[1]):.0f}px", file=sys.stderr, flush=True)
                            click_at = pos = rw
                    if probe is None and (lock.dwell > 0 or grab.mode != "off" or overlay.ring is not None):
                        probe = AXProbe(zone=args.zone)         # turned on live by `lock <ms> ...` or `grab ...`
                    target = probe.control_at(pos) if probe else None
                    # the control already highlighted stays the target until the cursor is clearly past it (user: the
                    # tiles "could be wider when they're detected to be the thing" being hovered over)
                    if hover is not None and rect_distance(pos, hover) <= lock.zone + STICKY_PX and (
                            target is None or rect_distance(pos, target) > 0):
                        target = hover
                    hover = target
                    # the unlocked filtered position: what a grab drags by (in aim mode the aim itself: the nudge
                    # holds the cursor still while a pinch forms, and a drag must still follow the hand)
                    if isinstance(mapper, AimMap):
                        # a drag follows the hand's base (wrist and knuckles, the steadiest point), not the aim: the ray
                        # magnifies every wobble, and a fader grab jumped 841 px (user, live: "freaking all over the place")
                        bx = sum(lm[k][0] for k in AIM_THROUGH["base"]) / 5
                        by = sum(lm[k][1] for k in AIM_THROUGH["base"]) / 5
                        free = (-bx * cw * GRAB_HAND_PX, by * ch * GRAB_HAND_PX)
                    else:
                        free = pos
                    pos = lock.update(pos, t_det, math.hypot(fx.dx, fy.dx), held=held,
                                      grab_at=click_at if held_down else None, target=target)
                    if probe and not grab.active:
                        probe.pos = pos
                    marker.move(*pos)
                    # the grab session: pinch-and-hold (or a closed hand) presses, vertical movement drags, release returns the cursor
                    if grab.mode != "off" and (cursor.enabled or grab.at is not None):   # no grabs while hands are off
                        if held_down and cursor.enabled:
                            role = probe.found[1] if (probe and probe.found and target is not None) else None
                            grab.begin(pos, t_det, free, role)
                        if held and grab.at is not None:
                            a = grab.step(t_det, free)
                            if a and a[0] == "press":
                                ok_here, what = probe.verify((a[1], a[2])) if (probe and grab.mode == "drag") else (True, "")
                                if ok_here:
                                    cursor.press(a[1], a[2])
                                    print(f"I,grab,press,{a[1]:.0f},{a[2]:.0f},{what}", file=sys.stderr, flush=True)
                                else:
                                    grab.reset()            # never press on something that is not the control
                                    print(f"I,grab,refused,{a[1]:.0f},{a[2]:.0f},{what}", file=sys.stderr, flush=True)
                            elif a and a[0] == "drag":
                                cursor.drag(a[1], a[2], a[3], a[4])
                        elif grab.at is not None and not held:
                            r = grab.end()
                            if r:
                                cursor.release(r[1], r[2])
                                cursor.warp(r[3], r[4])
                                print(f"I,grab,release,{r[2] - r[4]:+.0f}px,{r[1] - r[3]:+.0f}px across", file=sys.stderr, flush=True)
                    if tap is not None and grab.active:
                        tap["grabbed"] = True
                    if tevent == "up" and tap is not None:
                        # the finger gun fires: a quick tap that did not become a grab clicks where the cursor is held
                        # (user, live, trying to open Logic from the Dock with the trigger)
                        quick = t_det - tap["t"] <= TAP_S
                        if quick and not tap["grabbed"] and args.tap == "on" and cursor.enabled and thumb_mode:
                            at = lock.at if lock.at is not None else pos
                            if 0 <= at[0] < sw and 0 <= at[1] < sh:
                                cursor.press(*at)
                                cursor.release(*at)
                                print(f"I,tap,click,{at[0]:.0f},{at[1]:.0f}", file=sys.stderr, flush=True)
                                sound.play("Pop")
                            else:                     # aimed off the screen: never click (the edge would be the Dock)
                                print(f"I,tap,off-screen,{at[0]:.0f},{at[1]:.0f}", file=sys.stderr, flush=True)
                        tap = None
                    if event == "down":
                        click_at = pos
                        print(f"P,{t_det - t0:.2f},down,{dist:.3f},{click_at[0]:.0f},{click_at[1]:.0f}", file=sys.stderr, flush=True)
                        if not thumb_mode:            # with the finger gun, only the trigger sounds
                            sound.play("Tink")
                        if args.click:
                            cursor.press(*click_at)
                    elif event == "up":
                        print(f"P,{t_det - t0:.2f},up,{dist:.3f},{pos[0]:.0f},{pos[1]:.0f}", file=sys.stderr, flush=True)
                        if args.click:
                            cursor.release(*pos)
                    elif not grab.active:
                        cursor.move(min(max(pos[0], 0.0), sw - 1), min(max(pos[1], 0.0), sh - 1))
                    t_ov = time.perf_counter()
                    slow.append(("track", t_ov - t_det))
                    overlay.update(pos, lock.frame if lock.at is not None else target, locked=lock.at is not None,
                                   grabbing=grab.active, visible=cursor.enabled, nudging=nudge.on)
                    slow.append(("overlay", time.perf_counter() - t_ov))
            elif state == "held":
                pos = last_pos  # nothing moves: cursor, filters, pinch and button keep their state
            elif state == "lost":
                lost_n += 1
                if isinstance(mapper, RelativeMap):
                    mapper.lost()
                fx.reset(); fy.reset()
                lock.reset()
                despike.reset()
                precision.reset()
                nudge.reset()
                if grab.active:                                 # the hand vanished mid-grab: let go where it was
                    r = grab.end()
                    cursor.release(r[1], r[2])
                    cursor.warp(r[3], r[4])
                    print(f"I,grab,release,{r[2] - r[4]:+.0f}px,{r[1] - r[3]:+.0f}px across,hand lost", file=sys.stderr, flush=True)
                grab.reset()
                last_pos = None
                if cursor.down:
                    cursor.release(*(history[-1][1:] if history else (sw / 2, sh / 2)))
                if pinch.closed:
                    print(f"P,{t_det - t0:.2f},up,lost,0,0", file=sys.stderr, flush=True)
                pinch.reset()
                grip.reset()
                thumb.reset()
                held_was = False
                hover = None
                overlay.update(None, visible=False)             # the ring vanishing says "hand lost"
            if aimcal is not None:
                ev = aimcal.feed(t_det, aim_hit_now)
                n = len(aimcal.got) + aimcal.skipped
                if ev == "captured":
                    h = aimcal.got[-1][1]
                    print(f"I,aimcal,dot {n},captured,{h[0] * 100:.1f}cm,{h[1] * 100:.1f}cm", file=sys.stderr, flush=True)
                    sound.play("Tink")
                elif ev == "skipped":
                    print(f"I,aimcal,dot {n},skipped,no steady aim in {aimcal.timeout:g} s", file=sys.stderr, flush=True)
                    sound.play("Basso")
                if aimcal.done:
                    own, why = aimcal.fit(mapper.physical)
                    rounds = load_aim_rounds(sw, sh, args.aim_through)
                    if own and own["rms"] <= POOL_ROUND_MAX_PX:   # a round that moved is never stored: it would push a
                        # good one out of the recent window (live: 3 of 6 rounds pooled fell to 2; user: "that's a bad
                        # test, don't use that test")
                        rounds.append([[round(dt[0]), round(dt[1]), round(h[0], 5), round(h[1], 5)] for dt, h in aimcal.got])
                    rounds = rounds[-POOL_ROUNDS:]
                    fit, used, why2 = pool_rounds(rounds, mapper.physical, sw, sh)
                    this = f"this round {own['rms']:.0f} px" if own else f"this round unusable ({why})"
                    if fit:
                        mapper.map, mapper.calibrated = fit["map"], True
                        save_aim_cal(sw, sh, fit, args.eye, rounds=rounds, through=args.aim_through)
                        print(f"I,aimcal,done,{this},pooled over {used} of {len(rounds)} rounds rms {fit['rms']:.0f} px,"
                              f"saved {AIM_CAL_PATH}", file=sys.stderr, flush=True)
                        sound.play("Glass")
                    else:
                        if rounds:
                            old = load_aim_cal(sw, sh)
                            if old:
                                save_aim_cal(sw, sh, {"map": old, "rms": 0.0, "dots": []}, args.eye, rounds=rounds,
                                             through=args.aim_through)
                        print(f"I,aimcal,failed,{this},{why2},keeping the previous map", file=sys.stderr, flush=True)
                        sound.play("Basso")
                    aimcal = None
                    dots.close()
                    dots = None
                    fx.reset(); fy.reset(); despike.reset(); precision.reset(); lock.reset()
                    last_raw = None
                else:
                    # say what is in the way, so the user is never left guessing (live: "I don't understand where the
                    # cursor is"; there is none while calibrating)
                    why = ("I can't see your hand" if not hand else
                           "I can't see your face: keep your hand below it" if not mapper.eyes else
                           "hold still" if aimcal.progress < 0.05 else "holding...")
                    dots.show(aimcal.current, aimcal.progress,
                              f"Dot {aimcal.i + 1} of {len(aimcal.dots)}: line your fingertip up with the red dot.  {why}")
            if frames_out and pending_frame:
                # written after this frame's grab handling, so the engine sees a released grab on the release frame
                grab_state = "active" if grab.active else ("pending" if (grab.at is not None and grab.allowed) else None)
                extra = {"grab": grab_state} if grab_state else {}
                if isinstance(mapper, AimMap) and mapper.eyes:
                    extra["eyes"] = [[round(v, 4) for v in e] for e in mapper.eyes]   # aim replays need the eyes too
                frames_out.write(*pending_frame, extra=extra or None)
                pending_frame = None
            t_post = time.perf_counter()
            loop_ms.append((t_post - t_loop) * 1000)

            if diag is not None:
                # one packet per frame: every stage from the camera image to the cursor, for trace/diag.html
                r2 = lambda p: None if p is None else [round(p[0], 1), round(p[1], 1)]  # noqa: E731
                pkt = {"t": round(t_det - t0, 3), "fps": round(fps_now, 1), "det": round((t_det - t_frame) * 1000, 1),
                       "state": state, "hand": bool(hand), "screen": [sw, sh], "mode": args.mode, "point": args.point,
                       "img": None if nx is None else [round(nx, 4), round(ny, 4)],
                       "raw": r2(raw), "despiked": r2(q), "filtered": r2(smooth), "precise": r2(free), "cursor": r2(pos),
                       "precision": precision.gain,
                       "locked": lock.at is not None, "lock_frame": None if lock.frame is None else [round(v) for v in lock.frame],
                       "under": probe.label() if probe else "", "pinch": bool(pinch.closed),
                       "pinch_m": None if dist != dist else round(dist, 4),
                       "grab": grab.active, "grab_dy": round(grab.dy, 1), "grab_dx": round(grab.dx, 1)}
                if isinstance(mapper, MetricMap):
                    pkt["metric"] = None if mapper.last is None else [round(mapper.last[0] * 100, 2), round(mapper.last[1] * 100, 2)]
                    pkt["centre"] = None if mapper.centre is None else [round(mapper.centre[0] * 100, 2), round(mapper.centre[1] * 100, 2)]
                    pkt["pad"] = [round(mapper.pad[0] * 100, 1), round(mapper.pad[1] * 100, 1)]
                    pkt["dist_cm"] = None if not mapper.s else round(f_px * (cw / 1280) / mapper.s * 100, 1)
                    pkt["px_per_m"] = None if not mapper.s else round(mapper.s, 1)
                if isinstance(mapper, AimMap):
                    c3 = lambda p: None if p is None else [round(v * 100, 1) for v in p]  # noqa: E731
                    pkt["aim"] = {"eye_cm": c3(mapper.E), "tip_cm": c3(mapper.F), "hit_cm": c3(mapper.hit),
                                  "face": bool(mapper.eyes), "calibrated": mapper.calibrated, "nudge": nudge.on,
                                  "calibrating": aimcal is not None}
                try:
                    diag.sendto(_json.dumps(pkt, separators=(",", ":")).encode(), ("127.0.0.1", args.diag))
                except OSError:
                    pass

            if log:
                log.writerow([f"{t_frame - t0:.4f}", f"{(t_det - t_frame) * 1000:.2f}", f"{(t_post - t_det) * 1000:.2f}",
                              int(hand), *(f"{v:.1f}" for v in (raw or (0, 0))), *(f"{v:.1f}" for v in (pos or (0, 0))),
                              f"{ratio:.3f}", f"{dist:.4f}", int(pinch.closed), int(state == "held")])

            if args.preview:
                fh, fw = frame.shape[:2]          # the corrected frame is half size with --enhance
                for lm in (res.hand_landmarks or []):
                    for p in lm:
                        cv2.circle(frame, (int(p.x * fw), int(p.y * fh)), 3, (0, 255, 0), -1)
                if isinstance(mapper, AimMap):
                    for e in (mapper.eyes or ()):
                        cv2.circle(frame, (int(e[0] * fw), int(e[1] * fh)), 5, (255, 255, 0), -1)
                    if mapper.eye_px and res.hand_landmarks:   # the aiming line, eye to the hand point it passes through
                        hl = res.hand_landmarks[0]
                        ax = sum(hl[k].x for k in mapper.through) / len(mapper.through)
                        ay = sum(hl[k].y for k in mapper.through) / len(mapper.through)
                        cv2.line(frame, (int(mapper.eye_px[0] * fw / cw), int(mapper.eye_px[1] * fh / ch)),
                                 (int(ax * fw), int(ay * fh)), (0, 255, 255), 2)
                disp = cv2.flip(frame, 1)
                if isinstance(mapper, AbsoluteMap):
                    x0, y0, x1, y1 = mapper.rect
                    cv2.rectangle(disp, (int((1 - x1) * fw), int(y0 * fh)), (int((1 - x0) * fw), int(y1 * fh)),
                                  (0, 200, 255), 2)
                found = "HAND" if hand else ("HAND (held)" if state == "held" else "NO HAND")
                colour = (0, 220, 0) if hand else ((0, 200, 255) if state == "held" else (0, 0, 255))
                cv2.putText(disp, found, (16, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
                # live frame rate (user request): green at 28+, amber at 20+, red below
                fcol = (0, 220, 0) if fps_now >= 28 else ((0, 200, 255) if fps_now >= 20 else (0, 0, 255))
                stats = f"{fps_now:4.1f} fps   detect {(t_det - t_frame) * 1000:4.1f} ms   {args.point} / {args.prep} / {args.mode}"
                if isinstance(mapper, MetricMap) and mapper.s:
                    # the MacBook camera's field of view is ~70 degrees across: f = (w/2) / tan(35 deg)
                    d = mapper.distance((cw / 2) / math.tan(math.radians(35)))
                    stats += f"   hand ~{d * 100:.0f} cm away"
                if isinstance(mapper, AimMap):
                    if mapper.E and mapper.F:
                        stats += f"   eye ~{mapper.E[2] * 100:.0f} cm, hand ~{mapper.F[2] * 100:.0f} cm from the screen"
                    if face_ms:
                        stats += f"   face {face_ms[-1]:.1f} ms"
                cv2.putText(disp, stats, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(disp, stats, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fcol, 1)
                on = f" on {probe.label()}" if (probe and lock.frame is not None) else ""
                label = "CALIBRATING: sweep your range" if calibrating else (
                    f"AIM CALIBRATION: point at dot {aimcal.i + 1} of {len(aimcal.dots)}" if aimcal is not None else (
                        ("NO FACE: eye held" if mapper.eye_px else "NO FACE YET")
                        if isinstance(mapper, AimMap) and not mapper.eyes else "")) or (
                    "HANDS OFF: say \"hands on\"" if not cursor.enabled else (
                        f"GRAB: {grab.dx:+.0f}, {grab.dy:+.0f} px" if grab.active else (
                            "PINCH (locked)" if pinch.closed else (f"LOCKED{on}" if lock.at is not None else (
                                "NUDGE: small moves fine-tune" if nudge.on else "")))))
                if label:
                    cv2.putText(disp, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                t_pv = time.perf_counter()
                cv2.imshow("camera bench (q quits)", disp)
                key = cv2.waitKey(1) & 0xFF
                slow.append(("preview", time.perf_counter() - t_pv))
                if key == ord("q") and not args.control:
                    break                       # driven by the gesture program (--control), keys typed into the preview never quit it (found live)
            elif overlay.win is not None:
                overlay.pump()                  # no preview window runs AppKit's events: do it here
            elif dots is not None:
                dots.pump()
            if isinstance(mapper, AimMap):
                # the face runs after the cursor has moved, in the time read() would spend waiting for the next frame:
                # the eye used is one frame old, which a head (moving slowly) never notices, and the cursor gains nothing
                # in latency from the face tracker
                if facer is None:
                    facer = make_facer()
                t_fc = time.perf_counter()
                # the camera's own full-size colour image, not the hand's prepared one: the grey prep is half size, and at
                # 6 ft the eyes were ~20 px apart in it (user, live: "use color ... too dark of an area to register eyes
                # from more than like six feet away")
                fimg = img if args.face_prep == "hand" else mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
                fres = facer.detect_for_video(fimg, int((t_frame - t0) * 1000))
                face_ms.append((time.perf_counter() - t_fc) * 1000)
                slow.append(("face", face_ms[-1] / 1000))
                face = [(p.x, p.y, p.z) for p in fres.face_landmarks[0]] if fres.face_landmarks else None
                mapper.see_face(face, cw, ch, t_det)
                face_frames += 1
                face_seen += face is not None
                if t_det - aim_report >= 5.0:
                    if aim_report:
                        E, F = mapper.E, mapper.F
                        where = (f",eye {E[2] * 100:.0f} cm,hand {F[2] * 100:.0f} cm from the screen"
                                 if (E and F) else "")
                        print(f"I,aim,face {100 * face_seen / max(face_frames, 1):.0f}%,face ms p50 "
                              f"{percentile(face_ms, .5):.1f}{where}", file=sys.stderr, flush=True)
                    face_ms, face_seen, face_frames, aim_report = face_ms[-1:], 0, 0, t_det
            worst = max(slow, key=lambda x: x[1])
            if worst[1] > 0.08:                 # name any stall, so a hitch is never a mystery (found live: 250-500 ms)
                print(f"I,slow,{worst[0]},{worst[1] * 1000:.0f}ms", file=sys.stderr, flush=True)

            now = time.perf_counter()
            if now - last_report >= 1.0:
                dt = now - last_report
                if not rate_warned and now - t0 > 2 and frames / dt < args.fps - 3:
                    print(f"I,warning,camera delivering {frames / dt:.0f} fps against the {args.fps} pinned: a dim room "
                          f"has stretched the exposure; add light", file=sys.stderr, flush=True)
                    rate_warned = True
                wins = still_windows(sec_raw, still_px=args.still_px)
                if wins:
                    jitter = percentile([jitter_rms(sec_pos[s:s + 15]) for s in wins], .5)
                    still = 1
                else:
                    jitter, still = 0.0, 0
                print(f"S,{frames / dt:.1f},{percentile(det_ms, .5):.1f},{percentile(det_ms, .95):.1f},"
                      f"{percentile(loop_ms, .5):.1f},{100 * hands / max(frames, 1):.0f},{still},{jitter:.2f},"
                      f"{int(pinch.closed)},{held_gaps},{lost_n}", file=sys.stderr, flush=True)
                det_ms, loop_ms, frames, hands, sec_raw, sec_pos = [], [], 0, 0, [], []
                held_gaps = lost_n = 0
                last_report = now
            if args.seconds and now - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cursor.down:
            cursor.release(*(history[-1][1:] if history else (sw / 2, sh / 2)))
        cap.release()
        if frames_out:
            n, dropped = frames_out.close()
            print(f"I,frames,{n},dropped,{dropped}", file=sys.stderr, flush=True)
        if log_file:
            log_file.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
