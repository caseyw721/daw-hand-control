"""Compare pointing rays on the recorded calibration rounds (user's idea, live 2026-09-30: the line from the index
finger's big knuckle through the fingertip, wherever it points, against the eye-through-fingertip ray)."""
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/daw-hand-control/camera-bench"))
import numpy as np  # noqa: E402
from bench import AimMap, AimCalibration, EYE_CORNERS, back_project, aim_hit, px_per_metre, INDEX_TIP, INDEX_MCP  # noqa: E402

W, H, SW, SH, MM = 1280, 720, 1728, 1117, (345.0, 223.0)
START = re.compile(r"(calibrate|calibrated|calibrating|calorie|calvary|celebrate) (the )?(pointing|pointer|aim|aiming)|"
                   r"(start|run|redo|repeat) (the )?calibration|(redo|repeat) the test", re.I)
SESS = os.path.expanduser("~/Library/Logs/DAW Control/sessions")
NEW_TIMING_FROM = "20260930-144259"   # sessions from this one on ran the 2 s / 1.5 s / 15 s dots


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


def finger_hit(tip3, direction):
    """Where the line from the fingertip along `direction` meets the screen plane Z = 0, or None."""
    if direction[2] >= -1e-4:        # the finger must point towards the screen
        return None
    lam = -tip3[2] / direction[2]
    return (tip3[0] + lam * direction[0], tip3[1] + lam * direction[1])


RAYS = ("eye-tip", "knuckle-tip (world)", "knuckle-tip (image)", "blend eye+finger")


def rays_for_round(frames, t0, timing):
    m = AimMap(SW, SH, MM, through="tip", depth_k=0.12)
    cal = AimCalibration(SW, SH, **timing)
    f = m.f_px(W)
    per_frame = []                     # (t, {ray: hit})
    windows = []                       # (dot, t_from, t_to)
    for d in frames:
        t = d["t"]
        if d.get("eyes"):
            m.see_face(face_from(d["eyes"]), W, H, t)
        hits = {}
        if d.get("hands"):
            h = d["hands"][0]
            img, world = h["img"], h["world"]
            m(img, world, W, H)
            if m.hit is not None and m.s:
                hits["eye-tip"] = m.hit
                mz = sum(p[2] for p in world) / len(world)
                wt, wm = world[INDEX_TIP], world[INDEX_MCP]
                tip3 = m.F
                d_world = (-(wt[0] - wm[0]), wt[1] - wm[1], wt[2] - wm[2])
                fh = finger_hit(tip3, d_world)
                if fh:
                    hits["knuckle-tip (world)"] = fh
                mcp3 = back_project(img[INDEX_MCP][0] * W, img[INDEX_MCP][1] * H, f / m.s + (wm[2] - mz), W, H, f)
                d_img = (tip3[0] - mcp3[0], tip3[1] - mcp3[1], tip3[2] - mcp3[2])
                fi = finger_hit(tip3, d_img)
                if fi:
                    hits["knuckle-tip (image)"] = fi
                if fh:
                    hits["blend eye+finger"] = ((m.hit[0] + fh[0]) / 2, (m.hit[1] + fh[1]) / 2)
        per_frame.append((t, hits))
        if t >= t0 and not cal.done:
            i_before = cal.i
            ev = cal.feed(t, hits.get("eye-tip"))
            if ev == "captured":
                windows.append((cal.dots[i_before], t - cal.hold, t))
    out = {}
    for ray in RAYS:
        pairs = []
        for dot, a, b in windows:
            hs = [hh[ray] for tt, hh in per_frame if a <= tt <= b and ray in hh]
            if len(hs) >= 5:
                xs = sorted(p[0] for p in hs)
                ys = sorted(p[1] for p in hs)
                pairs.append((dot, (xs[len(xs) // 2], ys[len(ys) // 2])))
        out[ray] = pairs
    return out


def fit(pairs):
    if len(pairs) < 3:
        return None
    hx = np.array([p[1][0] for p in pairs]); hy = np.array([p[1][1] for p in pairs])
    sx = np.array([p[0][0] for p in pairs]); sy = np.array([p[0][1] for p in pairs])
    if hx.std() < 1e-4 or hy.std() < 1e-4:
        return None
    a, b = np.polyfit(hx, sx, 1)
    c, d = np.polyfit(hy, sy, 1)
    return a, b, c, d


def miss(mp, pairs):
    a, b, c, d = mp
    return [math.hypot(a * h[0] + b - s[0], c * h[1] + d - s[1]) for s, h in pairs]


rounds = []
for p in sorted(glob.glob(os.path.join(SESS, "20260930-*.frames.jsonl"))):
    name = os.path.basename(p).split(".")[0]
    frames, starts = load(p)
    if not any(fr.get("eyes") for fr in frames):
        continue
    timing = dict(settle=2.0, hold=1.5, timeout=15.0) if name >= NEW_TIMING_FROM else dict(settle=0.8, hold=1.0, timeout=12.0)
    for t0 in starts:
        rounds.append((name, rays_for_round(frames, t0, timing)))
# usable rounds: the eye-tip ray's own fit within 300 px (same set for every ray)
usable = []
for name, rr in rounds:
    mp = fit(rr["eye-tip"])
    if mp and len(rr["eye-tip"]) >= 4 and math.sqrt(np.mean(np.square(miss(mp, rr["eye-tip"])))) <= 300:
        usable.append((name, rr))
print(f"{len(rounds)} rounds recorded, {len(usable)} usable (eye-tip fits itself within 300 px)")
print(f"{'ray':>22} {'own fit rms (median)':>21} {'held out p50':>13} {'p90':>6}")
for ray in RAYS:
    own, held = [], []
    for i, (name, rr) in enumerate(usable):
        mp = fit(rr[ray])
        if mp:
            own.append(math.sqrt(np.mean(np.square(miss(mp, rr[ray])))))
        rest = [q for j, (_, r2) in enumerate(usable) if j != i for q in r2[ray]]
        mp2 = fit(rest)
        if mp2 and rr[ray]:
            held += miss(mp2, rr[ray])
    own.sort(); held.sort()
    q = lambda a, p: a[int(len(a) * p)] if a else float("nan")  # noqa: E731
    print(f"{ray:>22} {q(own, .5):21.0f} {q(held, .5):13.0f} {q(held, .9):6.0f}")
