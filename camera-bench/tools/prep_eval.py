#!/usr/bin/env python3
"""Compare preprocessing variants on the same recorded frames: how often the hand is found, how long
detection takes, and how steady the pointer anchor is.

    .venv/bin/python tools/prep_eval.py <dir of NNNN.jpg frames> [--fps 30]

Variants: raw, enhance (grey-world + CLAHE, half size, as --enhance), half (half size only), gray
(3-channel greyscale), gray_clahe, edges (Canny edges only, the "edges without colour" idea), and
edges_on_gray (edges drawn over greyscale). Each runs a fresh HandLandmarker in VIDEO mode with the
tracker's live settings (detection 0.3, presence 0.4, tracking 0.4).

Steadiness: for frames where the hand is found in consecutive frames, the median frame-to-frame move
of the anchor in screen pixels (the thumb anchor through the tracker's default box), and the count of
jumps over 40 px: the "jumpy" the user sees before any filtering.
"""
import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from bench import AbsoluteMap, MODEL, enhance, point_of  # noqa: E402


def variants():
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def half(f):
        return cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA)

    def gray(f):
        return cv2.cvtColor(cv2.cvtColor(half(f), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def gray_clahe(f):
        return cv2.cvtColor(clahe.apply(cv2.cvtColor(half(f), cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

    def edges(f):
        g = clahe.apply(cv2.cvtColor(half(f), cv2.COLOR_BGR2GRAY))
        e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 110)
        return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

    def edges_on_gray(f):
        g = clahe.apply(cv2.cvtColor(half(f), cv2.COLOR_BGR2GRAY))
        e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 110)
        return cv2.cvtColor(cv2.max(g, e), cv2.COLOR_GRAY2BGR)

    return {"raw": lambda f: f, "half": half, "enhance": lambda f: enhance(f, clahe), "gray": gray,
            "gray_clahe": gray_clahe, "edges": edges, "edges_on_gray": edges_on_gray}


def run(frames, fn, fps, point="thumb"):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL, delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_hands=1, min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.4, min_tracking_confidence=0.4)
    lmk = vision.HandLandmarker.create_from_options(opts)
    mapper = AbsoluteMap(1728, 1117, 0.6, None)
    found, prep_ms, det_ms, moves, prev = 0, [], [], [], None
    for k, f in enumerate(frames):
        t = time.perf_counter()
        img = fn(f)
        t1 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), int(k * 1000 / fps))
        t2 = time.perf_counter()
        prep_ms.append((t1 - t) * 1000)
        det_ms.append((t2 - t1) * 1000)
        if res.hand_landmarks:
            found += 1
            lm = [(p.x, p.y, p.z) for p in res.hand_landmarks[0]]
            x, y = mapper(*point_of(lm, point))
            if prev is not None:
                moves.append(math.hypot(x - prev[0], y - prev[1]))
            prev = (x, y)
        else:
            prev = None
    lmk.close()
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")  # noqa: E731
    return {"found": found / len(frames) * 100, "prep": med(prep_ms), "detect": med(det_ms),
            "move": med(moves), "jumps": sum(1 for m in moves if m > 40), "pairs": len(moves)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--only", help="comma-separated variant names")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
    frames = [cv2.imread(p) for p in paths]
    print(f"{len(frames)} frames from {args.dir}")
    print(f"{'variant':14} {'hand found':>10} {'prep ms':>8} {'detect ms':>9} {'anchor move px':>15} {'jumps >40px':>12}")
    names = args.only.split(",") if args.only else list(variants())
    for name in names:
        r = run(frames, variants()[name], args.fps)
        print(f"{name:14} {r['found']:9.0f}% {r['prep']:8.1f} {r['detect']:9.1f} {r['move']:15.1f} {r['jumps']:6d} of {r['pairs']}")


if __name__ == "__main__":
    main()
