"""Every control in an app's windows, from Accessibility, as JSON: for the intent model's test sets.

    .venv/bin/python tools/ax_snapshot.py ["Logic Pro"] > snapshot.json

Read-only: it only asks Accessibility what is on screen (nothing is pressed or changed). Each control gets
the same classification the live tracker uses (bench.AXProbe.classify) plus `where`: the names of its
named ancestors, innermost first (a mute button alone is "Mute"; its track header says which track).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import AppKit  # noqa: E402
from bench import AXProbe  # noqa: E402


def snapshot(app_name="Logic Pro", max_nodes=20000):
    apps = [a for a in AppKit.NSWorkspace.sharedWorkspace().runningApplications() if a.localizedName() == app_name]
    if not apps:
        raise SystemExit(f"{app_name} is not running")
    p = AXProbe.__new__(AXProbe)
    import ApplicationServices as AS
    p.AS, p.ok, p.cache = AS, True, {}
    app = AS.AXUIElementCreateApplication(apps[0].processIdentifier())
    out, seen = [], 0
    for win in p._attr(app, "AXWindows") or []:
        title = str(p._attr(win, "AXTitle") or "")
        stack = [(win, [])]
        while stack and seen < max_nodes:
            el, names = stack.pop()
            seen += 1
            role, frame, name, is_control = p.classify(el)
            if is_control and frame and frame[2] * frame[3] > 0:
                if role == "AXSlider":
                    frame = p._knob(el, frame)
                val = p._attr(el, "AXValue")
                out.append({"window": title, "role": role, "name": name, "frame": [round(v) for v in frame],
                            "where": names[:4], "value": None if val is None else str(val)[:40]})
                continue
            here = str(p._attr(el, "AXDescription") or p._attr(el, "AXTitle") or "")
            kids = p._attr(el, "AXChildren") or []
            stack.extend((k, ([here] if here else []) + names) for k in reversed(kids))
    return {"app": app_name, "taken": time.strftime("%Y-%m-%dT%H:%M:%S"), "nodes": seen, "controls": out}


if __name__ == "__main__":
    t = time.perf_counter()
    snap = snapshot(sys.argv[1] if len(sys.argv) > 1 else "Logic Pro")
    snap["seconds"] = round(time.perf_counter() - t, 1)
    json.dump(snap, sys.stdout, indent=1, ensure_ascii=False)
