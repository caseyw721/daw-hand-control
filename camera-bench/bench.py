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

Needs macOS Camera permission for the terminal (first run asks) and Accessibility
permission to move the cursor. Ctrl-C stops it.

Lines on stderr:
    I,...                                  information (camera, screen, calibrated box)
    P,<t s>,<down|up>,<pinch m>,<x>,<y>    pinch events (a Tink sound plays on "down")
    S,<fps>,<detect ms p50>,<detect ms p95>,<loop ms p50>,<hand %>,<still>,<jitter px rms>,<pinch>   once a second
"""
import argparse
import collections
import csv
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "hand_landmarker.task")

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 13, 17


class OneEuro:
    """1€ filter (Casiez et al. 2012). Units: whatever you feed it, per second."""

    def __init__(self, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-4)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mincutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x

    def reset(self):
        self.x = self.dx = self.t = None


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


def point_of(lm, which):
    if which == "tip":
        return lm[INDEX_TIP][0], lm[INDEX_TIP][1]
    # palm: mean of the four finger knuckles (much steadier than a fingertip)
    ks = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    return sum(lm[k][0] for k in ks) / 4, sum(lm[k][1] for k in ks) / 4


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


class Cursor:
    """Posts cursor moves and clicks with Quartz. Nothing else touches the cursor."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.down = False
        if enabled:
            import Quartz  # noqa: F401  (fail early if the binding is missing)
        self.Q = __import__("Quartz") if enabled else None

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
    ap.add_argument("--mode", choices=["abs", "rel"], default="abs")
    ap.add_argument("--point", choices=["tip", "palm"], default="tip")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60, help="requested camera rate (the camera decides)")
    ap.add_argument("--box", type=float, default=0.6, help="abs: fraction of the frame (centred) that spans the screen")
    ap.add_argument("--rect", type=parse_rect, help="abs: x0,y0,x1,y1 of the raw frame that spans the screen (overrides --box)")
    ap.add_argument("--calibrate", type=float, default=0,
                    help="abs: for this many seconds after a hand first appears, sweep your comfortable range; "
                         "that range becomes the box (printed as I,rect,... for --rect next time)")
    ap.add_argument("--gain", type=float, default=2.5, help="rel: screen widths per frame width of motion")
    ap.add_argument("--mincutoff", type=float, default=1.0, help="1€ filter: smoothing when still (Hz)")
    ap.add_argument("--beta", type=float, default=0.02, help="1€ filter: how fast smoothing turns off (per px/s)")
    ap.add_argument("--pinch-on", type=float, default=0.035, help="pinch closes below this thumb-index distance (m, world landmarks)")
    ap.add_argument("--pinch-off", type=float, default=0.055, help="pinch opens above this distance (m)")
    ap.add_argument("--heisenberg-ms", type=float, default=80, help="click where the cursor was this long before the pinch")
    ap.add_argument("--still", type=float, default=15, help="speed (px/s) below which the hand counts as still")
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
    mapper = AbsoluteMap(sw, sh, args.box, args.rect) if args.mode == "abs" else RelativeMap(sw, sh, args.gain)
    fx, fy = OneEuro(args.mincutoff, args.beta), OneEuro(args.mincutoff, args.beta)
    pinch = Pinch(on=args.pinch_on, off=args.pinch_off, min_frames=2, off_frames=3)
    history = collections.deque(maxlen=64)  # (t, x, y) for the Heisenberg offset
    calibrating = args.mode == "abs" and args.calibrate > 0
    cal_pts, cal_until = [], None

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit("camera %d did not open (Camera permission for this terminal?)" % args.camera)
    cw, ch, cfps = cap.get(3), cap.get(4), cap.get(5)
    print(f"I,camera,{int(cw)}x{int(ch)},{cfps:.0f}fps,screen,{sw}x{sh},mode,{args.mode},point,{args.point},"
          f"pinch,{args.pinch_on}/{args.pinch_off},presence,{args.presence},tracking,{args.tracking}",
          file=sys.stderr, flush=True)
    if args.mode == "abs":
        print("I,rect,%.3f,%.3f,%.3f,%.3f" % mapper.rect, file=sys.stderr, flush=True)

    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=args.model, delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=args.presence,
        min_tracking_confidence=args.tracking,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    log = None
    if args.log:
        log = csv.writer(open(args.log, "w", newline=""))
        log.writerow(["t_frame", "detect_ms", "post_ms", "hand", "raw_x", "raw_y", "x", "y",
                      "pinch_ratio", "pinch_m", "pinched"])

    last_hand_frame, dumps = None, 0
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    t0 = time.perf_counter()
    last_report = t0
    det_ms, loop_ms, frames, hands = [], [], 0, 0
    still_pts = []
    last_pos = None
    last_t = None
    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            t_frame = time.perf_counter()
            if not ok:
                print("I,camera read failed", file=sys.stderr, flush=True)
                time.sleep(0.01)
                continue
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(img, int((t_frame - t0) * 1000))
            t_det = time.perf_counter()
            det_ms.append((t_det - t_frame) * 1000)

            hand = bool(res.hand_landmarks)
            if args.dump and dumps < 30:
                if hand:
                    last_hand_frame = frame
                elif last_hand_frame is not None:
                    stamp = f"{t_frame - t0:06.2f}"
                    cv2.imwrite(os.path.join(args.dump, f"{stamp}_last.jpg"), last_hand_frame)
                    cv2.imwrite(os.path.join(args.dump, f"{stamp}_lost.jpg"), frame)
                    last_hand_frame, dumps = None, dumps + 1
            raw = pos = None
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
                    raw = mapper(nx, ny)
                    pos = (fx(raw[0], t_det), fy(raw[1], t_det))
                    history.append((t_det, pos[0], pos[1]))
                    if not math.isnan(dist):
                        event = pinch.update(dist)
                    if last_pos is not None and last_t is not None:
                        speed = math.dist(pos, last_pos) / max(t_det - last_t, 1e-4)
                        if speed < args.still:
                            still_pts.append(pos)
                    last_pos, last_t = pos, t_det
                    marker.move(*pos)
                    if event == "down":
                        # Heisenberg: the pinch itself nudges the finger; use the position from before it.
                        tb = t_det - args.heisenberg_ms / 1000
                        click_at = next(((x, y) for (t, x, y) in history if t >= tb), pos)
                        print(f"P,{t_det - t0:.2f},down,{dist:.3f},{click_at[0]:.0f},{click_at[1]:.0f}", file=sys.stderr, flush=True)
                        sound.play("Tink")
                        if args.click:
                            cursor.press(*click_at)
                    elif event == "up":
                        print(f"P,{t_det - t0:.2f},up,{dist:.3f},{pos[0]:.0f},{pos[1]:.0f}", file=sys.stderr, flush=True)
                        if args.click:
                            cursor.release(*pos)
                    else:
                        cursor.move(*pos)
            else:
                if isinstance(mapper, RelativeMap):
                    mapper.lost()
                fx.reset(); fy.reset()
                last_pos = None
                if cursor.down:
                    cursor.release(*(history[-1][1:] if history else (sw / 2, sh / 2)))
                if pinch.closed:
                    print(f"P,{t_det - t0:.2f},up,lost,0,0", file=sys.stderr, flush=True)
                    pinch.closed = False
            t_post = time.perf_counter()
            loop_ms.append((t_post - t_loop) * 1000)

            if log:
                log.writerow([f"{t_frame - t0:.4f}", f"{(t_det - t_frame) * 1000:.2f}", f"{(t_post - t_det) * 1000:.2f}",
                              int(hand), *(f"{v:.1f}" for v in (raw or (0, 0))), *(f"{v:.1f}" for v in (pos or (0, 0))),
                              f"{ratio:.3f}", f"{dist:.4f}", int(pinch.closed)])

            if args.preview:
                if hand:
                    for p in res.hand_landmarks[0]:
                        cv2.circle(frame, (int(p.x * cw), int(p.y * ch)), 3, (0, 255, 0), -1)
                disp = cv2.flip(frame, 1)
                if isinstance(mapper, AbsoluteMap):
                    x0, y0, x1, y1 = mapper.rect
                    cv2.rectangle(disp, (int((1 - x1) * cw), int(y0 * ch)), (int((1 - x0) * cw), int(y1 * ch)),
                                  (0, 200, 255), 2)
                label = "CALIBRATING: sweep your range" if calibrating else ("PINCH" if pinch.closed else "")
                if label:
                    cv2.putText(disp, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                cv2.imshow("camera bench (q quits)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.perf_counter()
            if now - last_report >= 1.0:
                dt = now - last_report
                if len(still_pts) >= 5:
                    mx = sum(p[0] for p in still_pts) / len(still_pts)
                    my = sum(p[1] for p in still_pts) / len(still_pts)
                    jitter = math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in still_pts) / len(still_pts))
                    still = 1
                else:
                    jitter, still = 0.0, 0
                print(f"S,{frames / dt:.1f},{percentile(det_ms, .5):.1f},{percentile(det_ms, .95):.1f},"
                      f"{percentile(loop_ms, .5):.1f},{100 * hands / max(frames, 1):.0f},{still},{jitter:.2f},"
                      f"{int(pinch.closed)}", file=sys.stderr, flush=True)
                det_ms, loop_ms, frames, hands, still_pts = [], [], 0, 0, []
                last_report = now
            if args.seconds and now - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cursor.down:
            cursor.release(*(history[-1][1:] if history else (sw / 2, sh / 2)))
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
