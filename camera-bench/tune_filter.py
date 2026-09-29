#!/usr/bin/env python3
"""Replay the raw positions of camera-bench --log CSVs through the 1€ filter at several settings and
report what each one costs, so the defaults come from recordings rather than guesses:

    jitter     px rms of the filtered point while the hand is still (bench.still_windows: the
               median-smoothed raw point stays within 15 px for half a second)
    lag        ms the filtered point trails the raw one while the hand moves (> 800 px/s), measured
               along the direction of travel: the filter's own share of motion-to-photon at speed
    onset      extra ms before the filtered point leaves rest after the raw point has; this is what a
               slow-motion flick test sees, because the filter is at its heaviest at rest
    overshoot  px the point continues past where the hand stopped (only non-zero with a lead)

    python3 tune_filter.py recordings/*.csv [--mincutoff 0.5,1,2,3,5,8] [--beta 0.01,0.02,0.05,0.1]
                           [--lead 0,30,50] [--lead-gate 150,500] [--screen 1728x1117]

Raw positions are the mapped screen positions before filtering, so a recording made with any filter
settings can be replayed with any others. Frames clamped at a screen edge are skipped. Rest, motion,
starts and stops are detected on a 5-frame median of the raw point, because at 30 fps the raw
fingertip's own jitter (about 5 px rms) already looks like 100-300 px/s of speed. The onset and
overshoot columns need recordings with real rests (hold still, flick, hold still); the first row
("raw", no filter) shows each column's noise floor.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import OneEuro, jitter_rms, lead_gain, median_smooth, parse_pair, still_windows  # noqa: E402

REST_V, MOVE_V, START_V = 120.0, 800.0, 1000.0  # px/s on the median-smoothed raw point
LEAVE_PX, STILL_WIN, REST_FRAMES = 30.0, 15, 10


def pct(a, p):
    if not a:
        return float("nan")
    s = sorted(a)
    return s[min(len(s) - 1, int(p * len(s)))]


def segments(path, sw, sh, max_gap=0.2):
    """Contiguous tracked stretches of (t, x, y) raw screen positions; a gap up to `max_gap`
    (the bench's hold) is bridged, as the filter now bridges it."""
    segs, cur = [], []
    for r in csv.DictReader(open(path)):
        t, rx, ry = float(r["t_frame"]), float(r["raw_x"]), float(r["raw_y"])
        ok = r["hand"] == "1" and not (rx == 0 and ry == 0 and r["x"] == "0.0")  # calibration frames
        ok = ok and 0 < rx < sw - 1 and 0 < ry < sh - 1  # clamped at an edge: the hand moved, the point did not
        if ok and cur and t - cur[-1][0] <= max_gap:
            cur.append((t, rx, ry))
        else:
            if len(cur) >= 3:
                segs.append(cur)
            cur = [(t, rx, ry)] if ok else []
    if len(cur) >= 3:
        segs.append(cur)
    return segs


def run_filter(seg, mincutoff, beta, lead_s, gate):
    fx, fy = OneEuro(mincutoff, beta), OneEuro(mincutoff, beta)
    out = []
    for t, x, y in seg:
        fx(x, t)
        fy(y, t)
        if lead_s:
            ahead = lead_gain(math.hypot(fx.dx, fy.dx), *gate) * lead_s
            out.append((fx.lead(ahead), fy.lead(ahead)))
        else:
            out.append((fx.x, fy.x))
    return out


def velocities(t, pts):
    """Central-difference velocity (px/s) of a point series."""
    v = [(0.0, 0.0)]
    for i in range(1, len(pts) - 1):
        dt = t[i + 1] - t[i - 1]
        v.append(((pts[i + 1][0] - pts[i - 1][0]) / dt, (pts[i + 1][1] - pts[i - 1][1]) / dt))
    v.append((0.0, 0.0))
    return v


def events(speed):
    """Starts and stops in a (median-smoothed) speed series. The smoothing spreads every transition
    over a few frames, so a start is the first frame above REST_V after at least REST_FRAMES below
    it, provided START_V is reached within the next 4 frames; a stop is (last fast frame, first
    frame below 2*REST_V) provided it stays below for 10 frames."""
    n = len(speed)
    starts, stops = [], []
    below = 0
    for i in range(n):
        if speed[i] < REST_V:
            below += 1
            continue
        if below >= REST_FRAMES and any(speed[k] > START_V for k in range(i, min(n, i + 4))):
            starts.append(i)
        below = 0
    last_fast, i = None, 0
    while i < n:
        if speed[i] > START_V:
            last_fast = i
        elif last_fast is not None and speed[i] < 2 * REST_V and i + 10 <= n \
                and all(speed[k] < 2 * REST_V for k in range(i, i + 10)):
            stops.append((last_fast, i))
            last_fast = None
            i += 10
            continue
        i += 1
    return starts, stops


def metrics(segs, mincutoff, beta, lead_s, gate=(150.0, 500.0)):
    jit, lag, onset, over = [], [], [], []
    for seg in segs:
        out = run_filter(seg, mincutoff, beta, lead_s, gate)
        t = [p[0] for p in seg]
        raw = [(p[1], p[2]) for p in seg]
        vel = velocities(t, median_smooth(raw))
        speed = [math.hypot(*v) for v in vel]
        n = len(seg)
        for s in still_windows(raw, STILL_WIN):
            jit.append(jitter_rms(out[s:s + STILL_WIN]))
        # lag along the direction of travel
        for i in range(1, n - 1):
            if speed[i] > MOVE_V:
                ux, uy = vel[i][0] / speed[i], vel[i][1] / speed[i]
                behind = (raw[i][0] - out[i][0]) * ux + (raw[i][1] - out[i][1]) * uy
                lag.append(1000 * behind / speed[i])
        starts, stops = events(speed)
        # onset: how much later than the raw point the filtered point leaves the rest position
        for i in starts:
            rest = raw[i - REST_FRAMES:i]
            rx, ry = sum(p[0] for p in rest) / len(rest), sum(p[1] for p in rest) / len(rest)
            t_raw = next((t[k] for k in range(i - REST_FRAMES, n) if math.dist(raw[k], (rx, ry)) > LEAVE_PX), None)
            t_out = next((t[k] for k in range(i - REST_FRAMES, n) if math.dist(out[k], (rx, ry)) > LEAVE_PX), None)
            if t_raw is not None and t_out is not None:
                onset.append(1000 * (t_out - t_raw))
        # overshoot: how far past the stopping point the filtered point travels, along the old heading
        for fast, i in stops:
            if i + 12 > n:
                continue
            sx = sum(p[0] for p in raw[i + 3:i + 12]) / 9
            sy = sum(p[1] for p in raw[i + 3:i + 12]) / 9
            ux, uy = vel[fast][0] / speed[fast], vel[fast][1] / speed[fast]
            over.append(max(0.0, max((out[k][0] - sx) * ux + (out[k][1] - sy) * uy for k in range(fast, i + 10))))
    return jit, lag, onset, over


def row(label, jit, lag, onset, over):
    return (f"{label:>21} | {pct(jit, .5):>9.2f} {pct(jit, .9):>7.2f} | {pct(lag, .5):>6.1f} {pct(lag, .9):>7.1f} | "
            f"{pct(onset, .5):>8.1f} {pct(onset, .9):>7.1f} | {pct(over, .5):>7.1f} {pct(over, .9):>8.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--mincutoff", default="0.5,1,2,3,5,8")
    ap.add_argument("--beta", default="0.01,0.02,0.05,0.1")
    ap.add_argument("--lead", default="0", help="prediction lead in ms, comma-separated")
    ap.add_argument("--lead-gate", type=parse_pair, default=(150.0, 500.0), metavar="LO,HI")
    ap.add_argument("--screen", default="1728x1117")
    a = ap.parse_args()
    sw, sh = (int(v) for v in a.screen.split("x"))
    segs = [s for p in a.csv for s in segments(p, sw, sh)]
    frames = sum(len(s) for s in segs)
    jit, lag, onset, over = metrics(segs, 1e9, 0, 0)  # unfiltered reference (huge cutoff = passthrough)
    print(f"{len(segs)} tracked stretches, {frames} frames from {len(a.csv)} file(s): {len(jit)} still windows, "
          f"{len(lag)} moving frames, {len(onset)} starts, {len(over)} stops"
          + ("" if len(onset) >= 20 else " (too few starts to trust the onset column: record hold-still / flick cycles)"))
    print(f"{'mincutoff beta lead':>21} | {'jitter px':>9} {'jit p90':>7} | {'lag ms':>6} {'lag p90':>7} | "
          f"{'onset ms':>8} {'ons p90':>7} | {'over px':>7} {'over p90':>8}")
    print(row("raw (no filter)", jit, lag, onset, over))
    for lead in (float(v) for v in a.lead.split(",")):
        for mc in (float(v) for v in a.mincutoff.split(",")):
            for beta in (float(v) for v in a.beta.split(",")):
                print(row(f"{mc:g} {beta:g} {lead:g}", *metrics(segs, mc, beta, lead / 1000, a.lead_gate)))


if __name__ == "__main__":
    main()
