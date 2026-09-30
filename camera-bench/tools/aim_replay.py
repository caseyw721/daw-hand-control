#!/usr/bin/env python3
"""Replays aim-mode calibrations from recorded sessions under different settings, and pools them.

    .venv/bin/python tools/aim_replay.py [SESSION.frames.jsonl ...] [--write base,0.02]

Every session the gesture program records (~/Library/Logs/DAW Control/sessions/<ts>.frames.jsonl) holds each camera
frame's hand landmarks and eye centres, and the phrases heard (`V,` lines). Each "calibrate pointing" starts a
calibration round; this replays the round through AimMap and AimCalibration exactly as the tracker would, for each
setting, and reports:

    captured   dots captured over all rounds
    own fit    rms miss of each round's dots against its own fit (what the tracker prints), median over rounds
    pooled     rms miss of all dots against one fit over every round (hand wobble averages out)
    held out   miss of each round's dots against a fit of the *other* rounds: the accuracy to expect next time

--write THROUGH,DEPTH saves the pooled fit of that setting as the tracker's calibration (it is loaded the next time
aim mode starts: "pad mode" then "aim mode", or a restart).
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench import (AIM_CAL_PATH, POOL_ROUNDS, POOL_ROUND_MAX_PX, AimCalibration, AimMap, EYE_CORNERS, pool_rounds,  # noqa: E402
                   save_aim_cal)

W, H, SW, SH, MM = 1280, 720, 1728, 1117, (345.0, 223.0)
START = re.compile(r"calibrate (the )?(pointing|pointer|aim|aiming)|(start|run|redo|repeat) (the )?calibration|"
                   r"(redo|repeat) the test", re.I)
SESSIONS = os.path.expanduser("~/Library/Logs/DAW Control/sessions")


def load(path):
    frames, starts = [], []
    for line in open(path):
        t_arr, _, rest = line.partition("\t")
        if rest.startswith("V,"):
            if START.search(re.sub(r"[^\w ]", " ", rest[2:])):
                starts.append(float(t_arr))
        elif rest.startswith("{"):
            try:
                frames.append(json.loads(rest))
            except ValueError:
                pass
    return frames, starts


def face_from(eyes):
    face = [(0.5, 0.5, 0.0)] * 478
    for (a, b), e in zip(EYE_CORNERS, eyes):
        face[a] = face[b] = tuple(e)
    return face


TIMING = {}   # settle / hold / timeout the rounds were recorded with (set from the command line)


def replay_round(frames, t0, through, depth_k):
    """The (dot px, hit m) pairs one calibration round captures with these settings."""
    m = AimMap(SW, SH, MM, through=through, depth_k=depth_k)
    cal = AimCalibration(SW, SH, **TIMING)
    for d in frames:
        t = d["t"]
        if d.get("eyes"):
            m.see_face(face_from(d["eyes"]), W, H, t)
        hit = None
        if d.get("hands"):
            h = d["hands"][0]
            m(h["img"], h["world"], W, H)
            hit = m.hit
        if t >= t0:
            cal.feed(t, hit)
            if cal.done:
                break
    return cal.got, m.physical


def fit(pairs, physical):
    cal = AimCalibration(SW, SH)
    cal.got = list(pairs)
    return cal.fit(physical)


def miss(mapping, pairs):
    a, b, c, d = mapping
    return [math.hypot(a * h[0] + b - dot[0], c * h[1] + d - dot[1]) for dot, h in pairs]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--write", metavar="THROUGH,DEPTH")
    ap.add_argument("--timing", default="0.8,1.0,12", metavar="SETTLE,HOLD,TIMEOUT",
                    help="the dot timing the rounds were recorded with: they must be replayed with it, or a capture lands "
                         "while the user is already on the next dot (0.8,1.0,12 until 2026-09-30 14:45, then 2,1.5,15)")
    args = ap.parse_args()
    st, ho, to = (float(v) for v in args.timing.split(","))
    TIMING.update(settle=st, hold=ho, timeout=to)
    paths = args.sessions or sorted(glob.glob(os.path.join(SESSIONS, "*.frames.jsonl")))
    rounds = []                       # (session name, frames, start time)
    for p in paths:
        frames, starts = load(p)
        if not any(f.get("eyes") for f in frames):
            continue
        for t0 in starts:
            rounds.append((os.path.basename(p).split(".")[0], frames, t0))
    print(f"{len(rounds)} calibration rounds in {len({r[0] for r in rounds})} sessions")
    settings = [(th, dk) for th in ("base", "pip", "tip") for dk in (0.12, 0.02, 0.005)]
    results = {}
    print(f"{'through':>7} {'depth':>6} {'captured':>9} {'own fit':>8} {'pooled':>7} {'held out p50 / p90':>19}")
    for th, dk in settings:
        per = []
        physical = None
        for name, frames, t0 in rounds:
            got, physical = replay_round(frames, t0, th, dk)
            per.append(got)
        own = [f["rms"] for f, _ in (fit(g, physical) for g in per if len(g) >= 3) if f]
        allp = [p for g in per for p in g]
        pooled, _ = fit(allp, physical) if len(allp) >= 3 else (None, "")
        held = []
        for i, g in enumerate(per):
            rest = [p for j, gg in enumerate(per) if j != i for p in gg]
            f, _ = fit(rest, physical) if len(rest) >= 3 else (None, "")
            if f and g:
                held += miss(f["map"], g)
        held.sort()
        results[(th, dk)] = (pooled, allp, per)
        cap = f"{len(allp)}/{5 * len(per)}"
        own_s = f"{sorted(own)[len(own) // 2]:.0f}" if own else "-"
        pooled_s = f"{pooled['rms']:.0f}" if pooled else "-"
        held_s = f"{held[len(held) // 2]:.0f} / {held[int(len(held) * 0.9)]:.0f}" if held else "-"
        print(f"{th:>7} {dk:>6g} {cap:>9} {own_s:>8} {pooled_s:>7} {held_s:>19}")
    if args.write:
        th, dk = args.write.split(",")
        _, _, per = results[(th, float(dk))]
        m = AimMap(SW, SH, MM, through=th)
        usable = [g for g in per if len(g) >= 3 and (fit(g, m.physical)[0] or {"rms": 1e9})["rms"] <= POOL_ROUND_MAX_PX]
        rounds = [[[round(dt[0]), round(dt[1]), round(h[0], 5), round(h[1], 5)] for dt, h in g] for g in usable]
        pooled, used, why = pool_rounds(rounds, m.physical, SW, SH)
        if not pooled:
            sys.exit(f"no pooled fit for that setting: {why}")
        save_aim_cal(SW, SH, pooled, "mid", through=th, rounds=rounds)
        print(f"wrote the pooled fit over {used} of {len(rounds[-POOL_ROUNDS:])} recent rounds (rms {pooled['rms']:.0f} pt) "
              f"for through {th}, depth {dk} to {AIM_CAL_PATH}")

if __name__ == "__main__":
    main()
