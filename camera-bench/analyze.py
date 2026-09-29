#!/usr/bin/env python3
"""Summarise a camera-bench --log CSV: coverage, pinches, drop-outs, jitter, filter lag.

    .venv/bin/python analyze.py run.csv [--screen 1728x1117] [--still 15]
"""
import argparse
import csv
import math
import statistics as st


def pct(a, p):
    if not a:
        return float("nan")
    s = sorted(a)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--screen", default="1728x1117", help="screen size in points (from the I,camera line)")
    ap.add_argument("--still", type=float, default=15, help="px/s below which the hand counts as still")
    a = ap.parse_args()
    sw, sh = (int(v) for v in a.screen.split("x"))

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        print("empty log")
        return
    dur = float(rows[-1]["t_frame"]) - float(rows[0]["t_frame"])
    hand = [r for r in rows if r["hand"] == "1"]
    print(f"frames {len(rows)} in {dur:.1f} s ({len(rows) / max(dur, 1e-6):.1f} fps), hand seen {100 * len(hand) / len(rows):.0f} %")
    det_h = [float(r["detect_ms"]) for r in hand]
    det_n = [float(r["detect_ms"]) for r in rows if r["hand"] == "0"]
    print(f"detect ms: tracked p50 {pct(det_h, .5):.1f} p95 {pct(det_h, .95):.1f} | no hand p50 {pct(det_n, .5):.1f}")

    # cursor coverage (frames where the cursor was driven: raw != 0,0 or hand and not calibrating)
    driven = [r for r in hand if not (r["raw_x"] == "0.0" and r["raw_y"] == "0.0" and r["x"] == "0.0")]
    if driven:
        xs = [float(r["x"]) for r in driven]
        ys = [float(r["y"]) for r in driven]
        cx = sum(1 for r in driven if float(r["raw_x"]) <= 0 or float(r["raw_x"]) >= sw - 1)
        cy = sum(1 for r in driven if float(r["raw_y"]) <= 0 or float(r["raw_y"]) >= sh - 1)
        print(f"cursor: x {min(xs):.0f}..{max(xs):.0f} of {sw - 1} ({100 * (max(xs) - min(xs)) / (sw - 1):.0f} % of width), "
              f"y {min(ys):.0f}..{max(ys):.0f} of {sh - 1} ({100 * (max(ys) - min(ys)) / (sh - 1):.0f} % of height); "
              f"clamped at an edge: x {100 * cx / len(driven):.0f} %, y {100 * cy / len(driven):.0f} % of frames")
        lag = [math.dist((float(r["raw_x"]), float(r["raw_y"])), (float(r["x"]), float(r["y"]))) for r in driven]
        print(f"1€ filter offset raw->filtered: p50 {pct(lag, .5):.1f} px, p95 {pct(lag, .95):.1f} px")

        # stillness jitter: rms about the mean over runs of consecutive still frames >= 0.5 s
        runs, cur = [], []
        prev = None
        for r in driven:
            t, x, y = float(r["t_frame"]), float(r["x"]), float(r["y"])
            if prev is not None:
                speed = math.dist((x, y), prev[1:]) / max(t - prev[0], 1e-4)
                if speed < a.still and t - prev[0] < 0.1:
                    cur.append((t, x, y))
                else:
                    if len(cur) >= 15:
                        runs.append(cur)
                    cur = []
            prev = (t, x, y)
        if len(cur) >= 15:
            runs.append(cur)
        if runs:
            js = []
            for run in runs:
                mx = sum(p[1] for p in run) / len(run)
                my = sum(p[2] for p in run) / len(run)
                js.append(math.sqrt(sum((p[1] - mx) ** 2 + (p[2] - my) ** 2 for p in run) / len(run)))
            longest = max(runs, key=len)
            print(f"still: {len(runs)} stretches, longest {longest[-1][0] - longest[0][0]:.1f} s, "
                  f"jitter rms p50 {st.median(js):.2f} px, worst {max(js):.2f} px")
        else:
            print("still: no stretch of 0.5 s or more with the hand still")

    # pinches (from the pinched column) and the pinch measure distribution
    key = "pinch_m" if "pinch_m" in rows[0] else "pinch_ratio"
    vals = [float(r[key]) for r in hand if r[key] not in ("nan", "")]
    if vals:
        print(f"{key}: p5 {pct(vals, .05):.3f} p50 {pct(vals, .5):.3f} p95 {pct(vals, .95):.3f}")
    events, prev = [], "0"
    for r in rows:
        if r["pinched"] != prev:
            events.append((float(r["t_frame"]), "down" if r["pinched"] == "1" else "up"))
            prev = r["pinched"]
    downs = [t for t, e in events if e == "down"]
    holds = []
    for i, (t, e) in enumerate(events):
        if e == "down" and i + 1 < len(events):
            holds.append(events[i + 1][0] - t)
    print(f"pinches: {len(downs)}" + (f", held {st.median(holds):.2f} s median" if holds else "") +
          (": " + ", ".join(f"{t:.1f}s" for t in downs) if downs else ""))

    # drop-outs: stretches with no hand after it was first seen
    first = float(hand[0]["t_frame"]) if hand else None
    gaps, start = [], None
    for r in rows:
        t = float(r["t_frame"])
        if first is None or t < first:
            continue
        if r["hand"] == "0" and start is None:
            start = t
        if r["hand"] == "1" and start is not None:
            gaps.append((start, t - start))
            start = None
    if start is not None:
        gaps.append((start, float(rows[-1]["t_frame"]) - start))
    long = [(s, d) for s, d in gaps if d >= 0.2]
    print(f"drop-outs >= 0.2 s after first sight: {len(long)}" +
          (" (" + ", ".join(f"{s:.1f}s for {d:.1f}s" for s, d in long[:8]) + (", ..." if len(long) > 8 else "") + ")" if long else ""))


if __name__ == "__main__":
    main()
