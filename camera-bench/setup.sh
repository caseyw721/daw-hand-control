#!/usr/bin/env bash
# One-time setup: Python 3.12 venv, MediaPipe, OpenCV, Quartz bindings, hand model (~8 MB).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$(command -v python3.12 || command -v python3)"
"$PY" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "MediaPipe needs Python 3.10-3.12"'
[ -d "$HERE/.venv" ] || "$PY" -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install -q --upgrade pip
"$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt"
mkdir -p "$HERE/models"
[ -s "$HERE/models/hand_landmarker.task" ] || curl -fsSL -o "$HERE/models/hand_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
echo "ready: $HERE/run.sh"
