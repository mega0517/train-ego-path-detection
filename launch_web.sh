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

# Auto-select an idle full (non-MIG) GPU unless the caller pinned one already.
# This host mixes a MIG-partitioned GPU (which breaks PyTorch's index-based CUDA
# init) with full A100s, so we must hand PyTorch a single non-MIG GPU by UUID.
# Respect an explicit CUDA_VISIBLE_DEVICES; only auto-fill when it is unset.
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_UUID="$(nvidia-smi \
        --query-gpu=uuid,memory.used,mig.mode.current --format=csv,noheader,nounits \
        2>/dev/null \
        | awk -F', *' '$3 != "Enabled" {gsub(/ /,"",$2); print $2, $1}' \
        | sort -n | head -1 | awk '{print $2}')"
    if [ -n "$GPU_UUID" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_UUID"
        echo "Auto-selected GPU: $CUDA_VISIBLE_DEVICES"
    else
        echo "No full (non-MIG) GPU found; running on CPU."
    fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"

echo "Starting TEP-Net web app on http://${HOST}:${PORT}"
exec "$PYTHON" web_app.py --host "$HOST" --port "$PORT" "$@"
