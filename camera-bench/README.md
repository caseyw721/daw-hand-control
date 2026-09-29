# Camera bench: hand tracking with no wearable (Phase 0)

A second pointer for the same tests. MediaPipe Hand Landmarker watches one hand through the
Mac's built-in camera and drives the cursor. A thumb-to-index pinch is a **gesture**: it is
reported (a `P` line on stderr and a Tink sound) and never clicks unless `--click` is given, so
in the test app you press Space with the other hand, as with the ring. The test app
(`controller/test-app`) measures it with tests A–D exactly as it measures the ring, so the
two can be compared on the same CSV. This is the "camera-based tracking" alternative a
hardware engineer suggested: absolute pointing, no drift, nothing to wear, but sampled at
the camera's rate.

## Run

```bash
controller/camera-bench/setup.sh   # once: Python 3.12 venv, MediaPipe 0.10.21, model (~8 MB)
controller/camera-bench/run.sh     # absolute pointing from the index fingertip
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
| `--mincutoff 1 --beta 0.02` | 1€ filter, same meaning as the ring's `mincutoff`/`beta` settings |
| `--pinch-on 0.035 --pinch-off 0.055` | Pinch thresholds on the thumb-to-index distance in MediaPipe's world landmarks (metres at an average hand size, independent of hand orientation) |
| `--click` | Make a pinch press and release the mouse button (off by default: pinch is a gesture) |
| `--heisenberg-ms 80` | With `--click`: click where the cursor was this long *before* the pinch began |
| `--presence 0.4 --tracking 0.4` | MediaPipe confidence floors; lower keeps the hand through harder poses at the cost of occasional false landmarks |
| `--quiet` | No sounds |
| `--no-cursor` | Measure only |
| `--preview` | Show the camera, landmarks, the box and pinch state (adds a few ms; off for measurements) |
| `--log run.csv` | Per-frame timings, positions and both pinch measures |
| `--marker [PX]` | A big black square with a white border (160 px by default) follows the pointer in a click-through window above everything, so a slow-motion video shows unambiguously which frame the pointer first moved in. Works with `--no-cursor` |

Lines on stderr: `I,...` information (camera, screen, box), `P,<t>,<down|up>,<pinch m>,<x>,<y>` on
each pinch, and once a second
`S,<fps>,<detect ms p50>,<detect ms p95>,<loop ms p50>,<hand %>,<still>,<jitter px rms>,<pinch>`.

## Testing against the ring

Use the test app's **Testing** menu to name the setup ("Camera abs tip", "Camera rel palm", …)
and run tests A–D like any other setup. Test A (lag, slow-motion video) is the one that
matters most: the camera's exposure and readout add delay that no software timer can see.

## What to expect (measured here, MacBook Pro camera, Apple silicon, CPU delegate)

| Stage | Measured / expected |
|---|---|
| Hand landmark detection | 8.3 ms p50, 8.9 ms p95 (once a hand is being tracked) |
| Camera frame period | 33 ms at 30 fps (the built-in camera's real rate) |
| Camera exposure + readout | 10–30 ms, only visible in the slow-motion lag test |
| Filtering | 1€ adds ~1 frame at slow speeds |

So motion-to-cursor is roughly 45–75 ms before the screen's own delay, against the ring's
target of under 5 ms. The camera wins on drift (none) and on gesture vocabulary; the ring wins
on latency, update rate and working while you play with the hand out of frame.

## Tests

```bash
controller/camera-bench/.venv/bin/python controller/camera-bench/tests/test_bench.py
```

## Notes

- MediaPipe 1.0.x crashes on this Mac in its Metal helper (`DrishtiMetalHelper ... Service is
  unavailable`) even with the CPU delegate, so the bench pins 0.10.21 with OpenCV 4 and
  numpy 1.x. `pip check` prints "not supported on this platform" for that wheel; it runs fine.
- In absolute mode the cursor jumps to wherever the hand reappears. In relative mode it stays.

## Measuring latency (motion-to-photon)

`--log` only sees the time from a frame arriving to the cursor event being posted (detect + filter +
dispatch). The camera's exposure and delivery, macOS's capture pipeline and the display's own lag
are outside the process and usually dominate, so the real number needs an external clock:

1. `run.sh --marker`, and film hand and screen together at 240 fps (iPhone slow-mo).
2. Make one sharp, small flick of the finger, then hold still. Repeat about ten times.
3. Scrub: count frames from the finger's first movement to the square's first movement.
   At 240 fps each frame is 4.17 ms. Report the median and the spread.
4. Same filming, same flick with the ring (a 1000 Hz USB mouse) gives the fair comparison.

Keep the 1€ settings fixed between runs: a heavier filter adds lag that is not the camera's.

### Measured 2026-09-29 (MacBook Pro M4 Pro, built-in camera, defaults: abs, tip, 1€ mincutoff 1 / beta 0.02)

Phone clip at 120 fps (stored at 30 fps, so 8.4 ms of real time per video frame), hand and screen in
one shot, `--marker` on. The square steps every 4 video frames, which confirms the 30 fps camera
rate and the time stretch. Palm centre (MediaPipe on the clip) cross-correlated with the square's
position:

| Axis | Peak lag | Range where correlation is within 0.02 of peak |
|---|---|---|
| y | 101 ms | 70–140 ms |
| x | 152 ms | 110–190 ms |

Motion-to-photon is therefore **roughly 100–150 ms**, of which the process itself is only ~12 ms
(detect 11 ms p50, dispatch 0.4 ms) plus the 33 ms frame interval. The rest is camera exposure and
delivery, the 1€ filter (at mincutoff 1 Hz its slow-motion time constant is ~160 ms, so it is a
large share at the speeds in this clip) and the display. The clip had no clean stop-then-flick
events, so per-flick frame counting was not possible; a stiller hand between flicks would sharpen
this to a few ms. Same run: hand lost 372 times in 8 min, 80 of them while pinched.
