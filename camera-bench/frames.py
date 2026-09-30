"""Frame stream for the gesture engine: one JSON object per camera frame, on stdout or in a file.

    ./run.sh --frames -            # to stdout (the I/P/S lines stay on stderr), e.g. piped into the engine
    ./run.sh --frames hand.jsonl   # to a file

One line per frame, including frames with no hand, so a replay sees the gaps:

    {"v":1,"seq":812,"t":51234.567890,"t_read":51234.532000,"t_capture":null,"det_ms":10.3,
     "w":1280,"h":720,"mirrored":false,"state":"tracked",
     "hands":[{"label":"Right","score":0.97,"img":[[x,y,z]*21],"world":[[x,y,z]*21]}]}

    t          right after inference, in CLOCK_UPTIME_RAW seconds (mach_absolute_time: seconds awake
               since boot, the same number in every process). The DAW Control trace uses that clock,
               so engine, ring and camera events share one timeline. bench.py measures with
               time.perf_counter, which under some interpreters counts from process start, so the
               writer converts with an offset taken once at start
    t_read     when cap.read() returned the frame, same clock. t_capture stays null until a native
               AVFoundation reader supplies the presentation timestamp (OpenCV does not expose it)
    det_ms     t - t_read: colour conversion plus inference
    img        21 landmarks in normalised coordinates of the *unmirrored* camera frame (x right,
               y down, z relative depth). world: the same 21 points in metres, centred on the hand
               (shape only, no translation). label/score: MediaPipe's handedness, which assumes a
               mirrored selfie image, so on this unmirrored frame it is probably swapped; the engine
               normalises it
    state      the Presence state the pointer used for this frame: tracked | held | lost | gone

The writer never blocks the pointer: lines go through a bounded queue to a writer thread, and a
frame that finds the queue full is dropped and counted (reported at exit as I,frames,<n>,dropped,<m>).
"""
import json
import queue
import sys
import threading
import time

SCHEMA_VERSION = 1


def shared_clock():
    """CLOCK_UPTIME_RAW: the cross-process clock the DAW Control trace uses (falls back to monotonic off macOS)."""
    try:
        return time.clock_gettime(time.CLOCK_UPTIME_RAW)
    except AttributeError:
        return time.monotonic()


def _pts(points):
    return [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in points]


def frame_record(seq, res, state, t_read, t_det, w, h, mirrored=False, t_capture=None):
    """The dict for one frame. `res` is a MediaPipe HandLandmarkerResult, or None for no detection."""
    hands = []
    if res is not None and res.hand_landmarks:
        worlds = res.hand_world_landmarks or []
        handed = res.handedness or []
        for i, lm in enumerate(res.hand_landmarks):
            label = score = None
            if i < len(handed) and handed[i]:
                label, score = handed[i][0].category_name, round(handed[i][0].score, 3)
            hands.append({"label": label, "score": score, "img": _pts(lm),
                          "world": _pts(worlds[i]) if i < len(worlds) else None})
    return {"v": SCHEMA_VERSION, "seq": seq, "t": round(t_det, 6), "t_read": round(t_read, 6),
            "t_capture": t_capture, "det_ms": round((t_det - t_read) * 1000, 2), "w": int(w), "h": int(h),
            "mirrored": mirrored, "state": state, "hands": hands}


class FrameWriter:
    """Serialises frames to `path` ('-' = stdout) from a writer thread; write() never blocks."""

    def __init__(self, path, w, h, maxsize=16):
        self.path = path
        self.w, self.h = int(w), int(h)
        self.offset = shared_clock() - time.perf_counter()   # perf_counter stamps -> the shared clock
        self.seq = 0
        self.written = 0
        self.dropped = 0
        self.q = queue.Queue(maxsize=maxsize)
        self.f = sys.stdout if path == "-" else open(path, "w")
        self.thread = threading.Thread(target=self._run, name="frames", daemon=True)
        self.thread.start()

    def write(self, res, state, t_read, t_det, t_capture=None, extra=None):
        """`extra`: more top-level fields, e.g. {"grab": "active"} while the tracker holds a grab (the gesture engine
        stands down then)."""
        self.seq += 1
        rec = frame_record(self.seq, res, state, t_read + self.offset, t_det + self.offset, self.w, self.h,
                           t_capture=t_capture)
        if extra:
            rec.update(extra)
        try:
            self.q.put_nowait(json.dumps(rec, separators=(",", ":")))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            line = self.q.get()
            if line is None:
                break
            try:
                self.f.write(line + "\n")
                self.f.flush()
                self.written += 1
            except (BrokenPipeError, OSError, ValueError):  # the reader went away or the file closed
                self.dropped += 1
        if self.f is not sys.stdout:
            self.f.close()

    def close(self, timeout=2.0):
        """Flushes what is queued and stops the thread. Returns (written, dropped)."""
        try:
            self.q.put(None, timeout=timeout)
        except queue.Full:
            pass
        self.thread.join(timeout)
        return self.written, self.dropped
