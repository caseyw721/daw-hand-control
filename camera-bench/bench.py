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
POINTS = ("tip", "thumb", "thumbip", "thumbtip", "web", "palm")
PREPS = ("none", "enhance", "gray")


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


CONTROL_ROLES = {"AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXSlider",
                 "AXIncrementor", "AXValueIndicator", "AXMenuItem", "AXMenuBarItem", "AXDisclosureTriangle",
                 "AXComboBox", "AXColorWell", "AXLevelIndicator", "AXDockItem", "AXCell", "AXLink", "AXTab"}
SNAP_MAX_PX = 90          # a control no bigger than this (both ways) snaps the cursor to its centre


class Lock:
    """Hover lock onto controls (user, live: "it's not locking to buttons when they're hovered over, it's
    just locking wherever it stops"). When the pointer rests (filtered speed under `still` px/s) for
    `dwell` s over a control that Accessibility reports (a button, slider, knob, checkbox, menu item...),
    the cursor locks onto it: small controls snap to their centre, big ones lock where the cursor is.
    Hand motion is ignored until it leaves the control's frame by `breakout` px. Resting anywhere else
    does not lock (`anywhere` restores the old rest-anywhere lock). While a grab is held (pinch closed)
    it never releases; a grab that starts unlocked locks where the pointer was just before the pinch."""

    def __init__(self, dwell=0.2, still=120.0, breakout=50.0, anywhere=False):
        self.dwell, self.still, self.breakout, self.anywhere = dwell, still, breakout, anywhere
        self.at = None             # (x, y) while locked
        self.frame = None          # (x, y, w, h) of the control locked onto
        self.still_since = None

    def _inside(self, pos):
        if self.frame is None:
            return math.hypot(pos[0] - self.at[0], pos[1] - self.at[1]) <= self.breakout
        x, y, w, h = self.frame
        m = self.breakout
        return x - m <= pos[0] <= x + w + m and y - m <= pos[1] <= y + h + m

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
            self.at = grab_at or pos
            return self.at
        if self.dwell > 0 and speed < self.still and (target is not None or self.anywhere):
            if self.still_since is None:
                self.still_since = t
            elif t - self.still_since >= self.dwell:
                if target is not None:
                    x, y, w, h = target
                    small = w <= SNAP_MAX_PX and h <= SNAP_MAX_PX
                    self.at = (x + w / 2, y + h / 2) if small else pos
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

    def __init__(self, period=0.08):
        self.period = period
        self.pos = None
        self.found = None          # (t, role, (x, y, w, h), title)
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
                    self.found = (time.monotonic(), None, None, "")
                    continue
                role = self._attr(el, "AXRole") or ""
                frame = self._frame(el)
                title = self._attr(el, "AXTitle") or self._attr(el, "AXDescription") or ""
                self.found = (time.monotonic(), str(role), frame, str(title))
            except Exception:
                self.found = (time.monotonic(), None, None, "")

    def control_at(self, pos):
        f = self.found
        if not f or f[1] not in CONTROL_ROLES or f[2] is None or time.monotonic() - f[0] > 0.4:
            return None
        x, y, w, h = f[2]
        if w * h > 300 * 300 or not (x - 2 <= pos[0] <= x + w + 2 and y - 2 <= pos[1] <= y + h + 2):
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
    pad_w × pad_h metres around `centre` spans the screen; the centre is where the hand first appears
    (or `recenter`). The distance estimate is smoothed hard: a hand's depth changes slowly."""

    def __init__(self, screen_w, screen_h, pad=(0.24, 0.15)):
        self.w, self.h = screen_w, screen_h
        self.pad = pad
        self.centre = None
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
        if self.centre is None or self.recenter_next:
            self.centre, self.recenter_next = (X, Y), False
        u = min(max((X - self.centre[0]) / self.pad[0] + 0.5, 0.0), 1.0)
        v = min(max((Y - self.centre[1]) / self.pad[1] + 0.5, 0.0), 1.0)
        return u * (self.w - 1), v * (self.h - 1)

    def distance(self, f_px):
        """Estimated camera-to-hand distance in metres for a focal length in px (for the preview)."""
        return f_px / self.s if self.s else None


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
    """The grab session (user's design, 2026-09-29): pinch and hold on a knob or slider, the tracker presses
    the mouse there after `hold_s` (a quick pinch never presses: pinches stay gestures), vertical hand
    movement drags it (`gain` px of drag per px of free cursor movement), and releasing the pinch lets go
    and puts the cursor back where the grab began. mode: off; drag = only on sliders/knobs Accessibility
    reports; any = anywhere (for plug-in GUIs that hide their knobs from Accessibility)."""

    def __init__(self, mode="off", gain=0.6, hold_s=0.25):
        self.mode, self.gain, self.hold_s = mode, gain, hold_s
        self.reset()

    def reset(self):
        self.at = self.t0 = self.y0 = None
        self.active = self.allowed = False
        self.dy = 0.0

    def begin(self, at, t, y, role):
        self.reset()
        self.at, self.t0, self.y0 = at, t, y
        self.allowed = self.mode == "any" or (self.mode == "drag" and role in DRAG_ROLES)

    def step(self, t, y):
        """('press', x, y) once the hold has lasted, then ('drag', x, y, ddy) as the hand moves; else None."""
        if self.at is None or not self.allowed:
            return None
        if not self.active:
            if t - self.t0 >= self.hold_s:
                self.active, self.y0, self.dy = True, y, 0.0
                return ("press", self.at[0], self.at[1])
            return None
        dy = (y - self.y0) * self.gain
        ddy = dy - self.dy
        if abs(ddy) < 0.5:
            return None
        self.dy = dy
        return ("drag", self.at[0], self.at[1] + dy, ddy)

    def end(self):
        """('release', x, y, back_x, back_y) if a press happened, else None; the session is over."""
        r = ("release", self.at[0], self.at[1] + self.dy, self.at[0], self.at[1]) if self.active else None
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
    ap.add_argument("--mode", choices=["abs", "rel", "metric"], default="abs",
                    help="abs: a box of the camera image spans the screen; rel: air mouse; metric: the hand's real "
                         "position in cm, corrected for its distance from the camera (no parallax), --pad spans the screen")
    ap.add_argument("--pad", type=parse_pair, default=(24.0, 15.0), metavar="W,H",
                    help="metric: centimetres of hand movement that span the screen, across and down")
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
    sw, sh = cursor.screen()
    def make_mapper(mode):
        if mode == "metric":
            return MetricMap(sw, sh, (args.pad[0] / 100, args.pad[1] / 100))
        if mode == "abs":
            return AbsoluteMap(sw, sh, args.box, args.rect)
        return RelativeMap(sw, sh, args.gain)
    mapper = make_mapper(args.mode)
    nonlocal_state = {}           # requests from the control thread, applied between frames by the main loop
    fx, fy = OneEuro(args.mincutoff, args.beta), OneEuro(args.mincutoff, args.beta)
    lock = Lock(args.lock_ms / 1000, args.lock_speed, args.breakout, anywhere=args.lock_anywhere)
    grab = Grab(args.grab, args.grab_gain)
    diag = None
    if args.diag:
        import json as _json
        import socket as _socket
        diag = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        diag.setblocking(False)
    f_px = (1280 / 2) / math.tan(math.radians(35))       # MacBook camera: ~70 degrees across (for the distance readout)
    probe = AXProbe() if (args.lock_ms or args.grab != "off") else None
    if args.lock_ms and not (probe and probe.ok):
        print("I,warning,no Accessibility bindings (pip install pyobjc-framework-ApplicationServices): hover lock off",
              file=sys.stderr, flush=True)
    despike = Despike(args.median)
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
                        fx.reset(); fy.reset(); lock.reset(); despike.reset()      # a new anchor starts a new track
                    elif parts[:1] == ["prep"] and len(parts) == 2 and parts[1] in PREPS:
                        args.prep = parts[1]
                    elif parts[:1] == ["median"] and len(parts) == 2:
                        despike.n = max(1, int(parts[1]))
                    elif parts[:1] == ["map"] and len(parts) == 2 and parts[1] in ("abs", "metric", "rel"):
                        nonlocal_state["mapper"] = parts[1]
                    elif parts[:1] == ["pad"] and len(parts) == 3:
                        args.pad = (float(parts[1]), float(parts[2]))
                        nonlocal_state["pad"] = True
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
    presence = Presence(hold_s=args.hold_ms / 1000)
    history = collections.deque(maxlen=64)  # (t, x, y) for the Heisenberg offset
    calibrating = args.mode == "abs" and args.calibrate > 0
    cal_pts, cal_until = [], None

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)  # (CAP_PROP_BUFFERSIZE is not supported by the AVFoundation backend)
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
                if want and want != args.mode:
                    args.mode = want
                    mapper = make_mapper(want)
                    fx.reset(); fy.reset(); despike.reset(); lock.reset()
                    last_raw = None
                if nonlocal_state.pop("pad", None) and isinstance(mapper, MetricMap):
                    mapper.pad = (args.pad[0] / 100, args.pad[1] / 100)
                if nonlocal_state.pop("recenter", None) and isinstance(mapper, MetricMap):
                    mapper.recenter_next = True
                    lock.reset()
                    sound.play("Glass")
            ok, frame = cap.read()
            t_frame = time.perf_counter()
            if not ok:
                print("I,camera read failed", file=sys.stderr, flush=True)
                time.sleep(0.01)
                continue
            if not pinned:
                cap.set(cv2.CAP_PROP_FPS, args.fps)
                pinned = True
            frames += 1
            fps_now = rate.tick(t_frame)
            if args.prep == "enhance":
                frame = enhance(frame, clahe)
            elif args.prep == "gray":
                frame = gray_prep(frame, clahe)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(img, int((t_frame - t0) * 1000))
            t_det = time.perf_counter()
            det_ms.append((t_det - t_frame) * 1000)

            hand = bool(res.hand_landmarks)
            state = presence.update(hand, t_det)
            if frames_out:
                frames_out.write(res, state, t_frame, t_det)
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
            raw = pos = q = free = None
            nx = ny = None
            ratio = dist = float("nan")
            event = None
            if hand:
                hands += 1
                lm = [(p.x, p.y, p.z) for p in res.hand_landmarks[0]]
                nx, ny = point_of(lm, args.point)
                ratio = pinch_ratio(lm)
                if res.hand_world_landmarks:
                    dist = pinch_dist([(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]])
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
                else:
                    if isinstance(mapper, MetricMap):
                        wlm = [(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]] if res.hand_world_landmarks else None
                        raw = mapper(nx, ny, lm, wlm, cw, ch) or last_raw or (sw / 2, sh / 2)
                    else:
                        raw = mapper(nx, ny)
                    last_raw = raw
                    q = despike(raw)
                    pos = (fx(q[0], t_det), fy(q[1], t_det))
                    if args.lead_ms:
                        ahead = lead_gain(math.hypot(fx.dx, fy.dx), *args.lead_gate) * args.lead_ms / 1000
                        pos = (fx.lead(ahead), fy.lead(ahead))
                    history.append((t_det, pos[0], pos[1]))
                    if not math.isnan(dist):
                        event = pinch.update(dist)
                    if 0 < raw[0] < sw - 1 and 0 < raw[1] < sh - 1:  # clamped at an edge is not "still"
                        sec_raw.append(raw)
                        sec_pos.append(pos)
                    last_pos = pos
                    # Heisenberg: the pinch itself nudges the finger; use the position from before it.
                    tb = t_det - args.heisenberg_ms / 1000
                    click_at = next(((x, y) for (t, x, y) in history if t >= tb), pos)
                    if probe is None and (lock.dwell > 0 or grab.mode != "off"):
                        probe = AXProbe()                       # turned on live by `lock <ms> ...` or `grab ...`
                    target = probe.control_at(pos) if probe else None
                    free = pos                                  # the unlocked filtered position: what a grab drags by
                    pos = lock.update(pos, t_det, math.hypot(fx.dx, fy.dx), held=pinch.closed,
                                      grab_at=click_at if event == "down" else None, target=target)
                    if probe and not grab.active:
                        probe.pos = pos
                    marker.move(*pos)
                    # the grab session: pinch-and-hold presses, vertical movement drags, release returns the cursor
                    if grab.mode != "off":
                        if event == "down":
                            role = probe.found[1] if (probe and probe.found) else None
                            grab.begin(pos, t_det, free[1], role)
                        if pinch.closed and grab.at is not None:
                            a = grab.step(t_det, free[1])
                            if a and a[0] == "press":
                                cursor.press(a[1], a[2])
                                print(f"I,grab,press,{a[1]:.0f},{a[2]:.0f}", file=sys.stderr, flush=True)
                            elif a and a[0] == "drag":
                                cursor.drag(a[1], a[2], 0, a[3])
                        elif grab.at is not None and not pinch.closed:
                            r = grab.end()
                            if r:
                                cursor.release(r[1], r[2])
                                cursor.warp(r[3], r[4])
                                print(f"I,grab,release,{r[2] - r[4]:+.0f}px", file=sys.stderr, flush=True)
                    if event == "down":
                        click_at = pos
                        print(f"P,{t_det - t0:.2f},down,{dist:.3f},{click_at[0]:.0f},{click_at[1]:.0f}", file=sys.stderr, flush=True)
                        sound.play("Tink")
                        if args.click:
                            cursor.press(*click_at)
                    elif event == "up":
                        print(f"P,{t_det - t0:.2f},up,{dist:.3f},{pos[0]:.0f},{pos[1]:.0f}", file=sys.stderr, flush=True)
                        if args.click:
                            cursor.release(*pos)
                    elif not grab.active:
                        cursor.move(*pos)
            elif state == "held":
                pos = last_pos  # nothing moves: cursor, filters, pinch and button keep their state
            elif state == "lost":
                lost_n += 1
                if isinstance(mapper, RelativeMap):
                    mapper.lost()
                fx.reset(); fy.reset()
                lock.reset()
                despike.reset()
                if grab.active:                                 # the hand vanished mid-grab: let go where it was
                    r = grab.end()
                    cursor.release(r[1], r[2])
                    cursor.warp(r[3], r[4])
                grab.reset()
                last_pos = None
                if cursor.down:
                    cursor.release(*(history[-1][1:] if history else (sw / 2, sh / 2)))
                if pinch.closed:
                    print(f"P,{t_det - t0:.2f},up,lost,0,0", file=sys.stderr, flush=True)
                pinch.reset()
            t_post = time.perf_counter()
            loop_ms.append((t_post - t_loop) * 1000)

            if diag is not None:
                # one packet per frame: every stage from the camera image to the cursor, for trace/diag.html
                r2 = lambda p: None if p is None else [round(p[0], 1), round(p[1], 1)]  # noqa: E731
                pkt = {"t": round(t_det - t0, 3), "fps": round(fps_now, 1), "det": round((t_det - t_frame) * 1000, 1),
                       "state": state, "hand": bool(hand), "screen": [sw, sh], "mode": args.mode, "point": args.point,
                       "img": None if nx is None else [round(nx, 4), round(ny, 4)],
                       "raw": r2(raw), "despiked": r2(q), "filtered": r2(free), "cursor": r2(pos),
                       "locked": lock.at is not None, "lock_frame": None if lock.frame is None else [round(v) for v in lock.frame],
                       "under": probe.label() if probe else "", "pinch": bool(pinch.closed),
                       "pinch_m": None if dist != dist else round(dist, 4),
                       "grab": grab.active, "grab_dy": round(grab.dy, 1)}
                if isinstance(mapper, MetricMap):
                    pkt["metric"] = None if mapper.last is None else [round(mapper.last[0] * 100, 2), round(mapper.last[1] * 100, 2)]
                    pkt["centre"] = None if mapper.centre is None else [round(mapper.centre[0] * 100, 2), round(mapper.centre[1] * 100, 2)]
                    pkt["pad"] = [round(mapper.pad[0] * 100, 1), round(mapper.pad[1] * 100, 1)]
                    pkt["dist_cm"] = None if not mapper.s else round(f_px * (cw / 1280) / mapper.s * 100, 1)
                    pkt["px_per_m"] = None if not mapper.s else round(mapper.s, 1)
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
                cv2.putText(disp, stats, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(disp, stats, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fcol, 1)
                on = f" on {probe.label()}" if (probe and lock.frame is not None) else ""
                label = "CALIBRATING: sweep your range" if calibrating else (
                    f"GRAB: turning {grab.dy:+.0f} px" if grab.active else (
                        "PINCH (locked)" if pinch.closed else (f"LOCKED{on}" if lock.at is not None else "")))
                if label:
                    cv2.putText(disp, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                cv2.imshow("camera bench (q quits)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

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
