# daw-hand-control

Hand input for the DAW Control layer (Logic Pro + UAD Console). Two candidate pointers,
compared on the same tests:

- `camera-bench/` — MediaPipe Hand Landmarker on the Mac's built-in camera → cursor.
  Pinch is a gesture, never a click unless `--click`. See its README.
- Fingertip IMU ring — firmware and receiver live in the ring repo (caseyw721/new-project, ~/ring).

Both are measured with the ring repo's test app (`controller/test-app`, tests A–D), which
needs no serial connection, so they produce the same CSV.
