#!/usr/bin/env bash
# Run the camera bench with its own Python environment (see setup.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/.venv/bin/python" ] || "$HERE/setup.sh"
exec "$HERE/.venv/bin/python" "$HERE/bench.py" "$@"
