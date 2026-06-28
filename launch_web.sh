#!/usr/bin/env bash
# Launch the TEP-Net web inference application.
set -e
cd "$(dirname "$0")"

# Prefer the project virtualenv if present.
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"

echo "Starting TEP-Net web app on http://${HOST}:${PORT}"
exec "$PYTHON" web_app.py --host "$HOST" --port "$PORT" "$@"
