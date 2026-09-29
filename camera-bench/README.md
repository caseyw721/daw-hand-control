# Camera bench: hand tracking with no wearable (Phase 0)

A second pointer for the same tests. MediaPipe Hand Landmarker watches one hand through the
Mac's built-in camera and drives the cursor. A thumb-to-index pinch is a **gesture**: it is
reported (a `P` line on stderr and a Tink sound) and never clicks unless `--click` is given, so
in the test app you press Space with the other hand, as with the ring. The ring repo's test app
(`~/ring/controller/test-app`) measures it with tests A–D exactly as it measures the ring, so the
two can be compared on the same CSV. This is the "camera-based tracking" alternative a
hardware engineer suggested: absolute pointing, no drift, nothing to wear, but sampled at
the camera's rate and delivered late (numbers below).

## Run

```bash
camera-bench/setup.sh   # once: Python 3.12 venv, MediaPipe 0.10.21, model (~8 MB)
camera-bench/run.sh     # absolute pointing from the index fingertip
```

Permissions the terminal app needs (macOS asks the first time, or grant them by hand in
System Settings → Privacy & Security): **Camera** (to see the hand) and **Accessibility**
(to move the cursor). Grant them to the terminal you launch from (Ghostty, Terminal, …).

Useful options (`run.sh --help` lists them all):

| Option | What it does |
|---|---|
| `--calibrate 5` | Absolute mode: for 5 s after the hand first appears, sweep your comfortable reach; that becomes the box that spans the screen. A chime marks the end and the box is printed as `I,rect,...` |
| `--rect x0,y0,x1,y1` | Absolute mode: use this box of the raw camera frame (paste the `I,rect` values from a calibrated run) |
| `--box 0.5` | Absolute mode without calibration: centred box, this fraction of the frame |
| `--mode rel` | Air-mouse style: hand motion moves the cursor, no fixed mapping (like the ring) |
| `--point palm` | Track the palm centre instead of the fingertip: steadier, less "pointing" |
| `--hold-ms 180` | Coast through a hand loss shorter than this (5 frames from the last sighting): filters, pinch, button and cursor keep their state. Most losses are one-frame flickers, see below. `0` restores the old behaviour |
| `--mincutoff 3 --beta 0.05` | 1€ filter, same meaning as the ring's settings. Defaults chosen from recordings with `tune_filter.py`, see below |
| `--lead-ms 40` | Predict this far ahead along the filtered velocity while the hand moves. Buys back latency, costs overshoot at stops (about speed × lead: 60 px for a 1500 px/s flick with 40 ms). Off by default until filmed |
| `--lead-gate 150,500` | Filtered speeds (px/s) between which the lead ramps in from nothing, so a resting hand (whose filtered speed reaches ~95 px/s from jitter alone) gets none of it |
| `--still-px 15` | For the `S` line: the hand counts as at rest for a half-second window when its median-smoothed raw point stays within this; `jitter` is then the rms of the shown point in those windows |
| `--fps 30` | Camera rate to pin. Leave it: asking for more lets a dim room drop the camera to 15 fps (see Notes) |
| `--pinch-on 0.035 --pinch-off 0.055` | Pinch thresholds on the thumb-to-index distance in MediaPipe's world landmarks (metres at an average hand size, independent of hand orientation) |
| `--click` | Make a pinch press and release the mouse button (off by default: pinch is a gesture) |
| `--heisenberg-ms 80` | With `--click`: click where the cursor was this long *before* the pinch began |
| `--presence 0.4 --tracking 0.4` | MediaPipe confidence floors; lower keeps the hand through harder poses at the cost of occasional false landmarks |
| `--quiet` | No sounds |
| `--no-cursor` | Measure only |
| `--preview` | Show the camera, landmarks, the box and pinch state (adds a few ms; off for measurements) |
| `--log run.csv` | Per-frame timings, positions, both pinch measures and whether the frame was coasted |
| `--marker [PX]` | A big black square with a white border (160 px by default) follows the pointer in a click-through window above everything, so a slow-motion video shows unambiguously which frame the pointer first moved in. Works with `--no-cursor` |

Lines on stderr: `I,...` information (camera, screen, box, warnings), `P,<t>,<down|up>,<pinch m>,<x>,<y>` on
each pinch, and once a second
`S,<fps>,<detect ms p50>,<detect ms p95>,<loop ms p50>,<hand %>,<still>,<jitter px rms>,<pinch>,<held gaps>,<lost>`
where `held gaps` counts short losses coasted through and `lost` the real ones.

## Testing against the ring

Use the test app's **Testing** menu to name the setup ("Camera abs tip", "Camera rel palm", …)
and run tests A–D like any other setup. Test A (lag, slow-motion video) is the one that
matters most: the camera's exposure and delivery add delay that no software timer can see.

## Where the time goes (measured 2026-09-29, MacBook Pro M4 Pro built-in camera, CPU delegate)

| Stage | Measured | How |
|---|---|---|
| Camera frame period | 33.3 ms (30 fps is the camera's ceiling on every format) | `tools/cam_probe.swift` |
| Camera delivery, frame timestamp → frame handed to the process | **p10 72, p50 87, p90 94 ms** (60–97 ms across runs) | `tools/cam_probe.swift` |
| Hand landmark detection | 9–13 ms p50 while tracking, ~19 ms when the palm detector has to run | `S` line |
| 1€ filter at cruising speed | 5 ms p50 / 13 ms p90 with the defaults (12 / 25 ms with the old 1 Hz / 0.02) | `tune_filter.py` |
| 1€ filter leaving rest | time constant 53 ms at 3 Hz (159 ms at the old 1 Hz); not yet measured on a recording with proper rests | theory |
| Cursor event dispatch | 0.4 ms | `--log` |
| Display (120 Hz panel) | ~4–12 ms | not measured |

So the process itself costs ~15 ms and the camera ~90 ms before the display. That matches the
slow-motion measurement of 100–150 ms below, and it puts the **floor for this camera near
100 ms** motion-to-photon. The ring targets under 5 ms. Only a faster camera changes the floor:
a USB (UVC) camera at 60–120 fps has an 8–17 ms frame period and usually a shorter pipeline;
`tools/cam_probe.swift <seconds> <name>` measures any camera the same way before buying into it.

## Tests

```bash
camera-bench/.venv/bin/python camera-bench/tests/test_bench.py
```

## Notes

- MediaPipe 1.0.x crashes on this Mac in its Metal helper (`DrishtiMetalHelper ... Service is
  unavailable`) even with the CPU delegate, so the bench pins 0.10.21 with OpenCV 4 and
  numpy 1.x. `pip check` prints "not supported on this platform" for that wheel; it runs fine.
- In absolute mode the cursor jumps to wherever the hand reappears. In relative mode it stays.
- **Keep `--fps 30`.** The camera's formats allow 15–30 fps. Asking for 60 leaves the maximum
  frame duration at 1/15 s, and in a dim room auto-exposure takes it: one probe run went to
  15 fps with 66 ms exposures. Pinned at 30 the camera raises gain instead. The bench prints an
  `I,warning` if the camera reports anything but the pinned rate.
- **Light the hand.** In this room the camera sits at ISO ~2100 with the exposure at the full
  33 ms frame period, so every frame is motion-blurred over 33 ms and MediaPipe's landmarks
  smear during fast moves. A desk lamp on the hand brings the exposure down to a few ms:
  steadier landmarks, fewer flickers, and the frame is captured earlier in its period.
  `tools/cam_probe.swift` prints the exposure and a verdict.
- The camera ignores both session presets and `activeFormat`: it delivers 1920x1080 whatever is
  asked (OpenCV then reports 1280x720). There is no resolution to trade for speed here.
- OpenCV's AVFoundation backend hands over the latest frame (`read()` blocks 26–40 ms, no queue),
  ignores `CAP_PROP_BUFFERSIZE` and has no frame timestamps (`POS_MSEC` is always 0). The
  camera latency above therefore comes from the Swift probe, not from `--log`.

## Drop-outs and `--hold-ms`

MediaPipe reports "no hand" for a frame or two hundreds of times an hour, most often during a
pinch (the closed fingers dip its presence score). In the 8-minute latency run below
(15 201 frames, hand in frame 60 %) there were **372 losses: 220 lasted one frame, 56 two or
three, 32 four to eight, and 62 nine or more** (the real departures). Across a one-frame loss
the fingertip had moved a median of 67 px. 111 of the losses had a pinch distance under
0.035 m just before, and 80 of them cancelled a pinch in progress.

Before `--hold-ms`, every missing frame reset the filter, cancelled the pinch (`P,...,up,lost`)
and released the button. Now a loss shorter than the hold (180 ms = 5 frames from the last
sighting, covering 79 % of the losses in that run) is coasted through: the cursor stays, the
filter and pinch keep their state, and the `S` line counts it under `held gaps`; the CSV marks
the frame `held=1` with the held position. Only a longer loss is reported and handled as before,
and it now clears the pinch detector completely (clearing only its state left a half-counted
release behind, so a single frame could have fired the next pinch). `analyze.py` prints how many
frames were coasted.

## Filter settings from recordings: `tune_filter.py`

```bash
python3 camera-bench/tune_filter.py camera-bench/recordings/*.csv [--mincutoff 1,3,5] [--beta 0.02,0.05] [--lead 0,40]
```

It replays the raw (unfiltered, mapped) positions of `--log` recordings through the 1€ filter and
reports, per setting: **jitter** (px rms while the hand is still), **lag** (ms the filtered point
trails the raw one while moving faster than 800 px/s), **onset** (extra ms before the filtered
point leaves rest, what a flick test sees) and **overshoot** (px past a stop, only with a lead).
Rest, motion, starts and stops are detected on a 5-frame median of the raw point because the raw
fingertip's own jitter (5.5 px rms) already looks like 100–300 px/s of speed at 30 fps.

Over the four recordings in `recordings/` (9 391 tracked frames, raw jitter 5.5 px rms):

| mincutoff | beta | jitter px rms | lag at speed, p50 / p90 ms | time constant at rest |
|---|---|---|---|---|
| 1 | 0.02 (old default) | 3.2 | 12.3 / 25.4 | 159 ms |
| 1 | 0.05 | 3.9 | 5.5 / 15.1 | 159 ms |
| 2 | 0.05 | 4.4 | 5.2 / 13.8 | 80 ms |
| **3** | **0.05 (default)** | **4.7** | **5.0 / 12.7** | **53 ms** |
| 5 | 0.05 | 4.8 | 4.7 / 10.8 | 32 ms |
| 3 | 0.1 | 4.8 | 2.7 / 8.0 | 53 ms |

The filter cannot buy much stillness at any setting (3–5 px against 5.5 raw), so the defaults
spend little on it: `beta 0.05` halves the lag at speed for 0.7 px, and `mincutoff 3` cuts the
lag of a move from rest three-fold. The onset and overshoot columns are nearly empty for these
recordings (4 starts, 40 stops in 9 391 frames) because the hand hardly ever rested in them; the
few starts agree with the theory (old defaults leave rest one frame late, new ones on time).
`--lead 40` on the default row leaves jitter at rest unchanged (4.7 → 4.8 px) and puts the point
12 ms ahead at cruising speed; on synthetic 1500 px/s flicks it overshoots stops by ~56 px, which
is the price to weigh against ~40 ms.

To fill those columns properly, record about two minutes with the hand inside the box and no
Accessibility needed:

```bash
camera-bench/run.sh --no-cursor --quiet --seconds 150 --log camera-bench/recordings/$(date +%F)-flicks.csv
```

20 times per axis: hold still 2 s, one sharp 15–25 cm flick, hold still 2 s; then 30 s of slow
tracing and 30 s of rest. That gives ≥ 60 starts and stops and ≥ 100 still windows. Light the
hand first (see Notes). A cheap experiment on the same run: `--presence 0.3`, since a third of the
losses had the pinch pose just before them and a lower presence floor may keep those frames.

Also fixed on the way: the filter's velocity estimate used the previous *filtered* value instead
of the previous raw one, which over-read speed by a third at 1000 px/s (the paper's form is now
used; the tests check a 40 ms lead is 40 px at 1000 px/s). And the `S` line's `still`/`jitter`
fields used to require the *filtered* speed to stay under 15 px/s, which the lighter defaults
almost never satisfy; they now use the same raw-point rest windows as `tune_filter.py` and
`analyze.py`, so the three agree and the number does not depend on the filter being judged.

## Recordings

`recordings/` holds the `--log` CSVs (and stderr) of the live runs so far, so the tuning above is
reproducible: three abs/tip runs from 2026-09-28 (default box, `--rect 0.2,0.2,0.8,0.8`, calibrated
box) and the 8-minute `--marker` latency run from 2026-09-29. They are the only CSVs not ignored
by git.

## Measuring latency (motion-to-photon)

`--log` only sees the time from a frame arriving to the cursor event being posted (detect + filter +
dispatch). The camera's exposure and delivery, macOS's capture pipeline and the display's own lag
are outside the process and dominate, so the real number needs an external clock:

1. `run.sh --marker`, and film hand and screen together at 240 fps (iPhone slow-mo).
2. Make one sharp, small flick of the finger, then **hold still for at least 0.3 s**. Repeat about ten times.
3. Scrub: count frames from the finger's first movement to the square's first movement.
   At 240 fps each frame is 4.17 ms. Report the median and the spread.
4. Same filming, same flick with the ring (a 1000 Hz USB mouse) gives the fair comparison.

Keep the 1€ settings fixed between runs: a heavier filter adds lag that is not the camera's.
The stills between flicks also make the recording usable for `tune_filter.py`'s onset column.

### Measured 2026-09-29 (MacBook Pro M4 Pro, built-in camera, then-defaults: abs, tip, 1€ mincutoff 1 / beta 0.02)

Phone clip at 120 fps (stored at 30 fps, so 8.4 ms of real time per video frame), hand and screen in
one shot, `--marker` on. The square steps every 4 video frames, which confirms the 30 fps camera
rate and the time stretch. Palm centre (MediaPipe on the clip) cross-correlated with the square's
position:

| Axis | Peak lag | Range where correlation is within 0.02 of peak |
|---|---|---|
| y | 101 ms | 70–140 ms |
| x | 152 ms | 110–190 ms |

Motion-to-photon was therefore **roughly 100–150 ms**. The budget table above now accounts for it:
~87 ms camera delivery, 33 ms frame period, ~12 ms process, 12–25 ms of the old filter at speed
(more at onset), plus the display. The clip had no clean stop-then-flick events, so per-flick frame
counting was not possible. Same run: hand lost 372 times in 8 min, 80 of them while pinched
(the drop-out section above; `--hold-ms` now covers most of these).
