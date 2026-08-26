#!/usr/bin/env bash
# Fetch the official static `unrar` binary into ./bin so the File Browser tab can
# extract .rar archives. No root required — the binary is self-contained.
#
# Idempotent: exits early if a working ./bin/unrar already exists.
# Override the download URL with UNRAR_URL if rarlab.com is unreachable.
set -euo pipefail

cd "$(dirname "$0")"
BIN_DIR="bin"
TARGET="$BIN_DIR/unrar"
UNRAR_URL="${UNRAR_URL:-https://www.rarlab.com/rar/rarlinux-x64-700.tar.gz}"

if [ -x "$TARGET" ] && "$TARGET" >/dev/null 2>&1; then
    echo "unrar already present: $TARGET"
    "$TARGET" 2>&1 | head -1
    exit 0
fi

echo "Fetching unrar from: $UNRAR_URL"
mkdir -p "$BIN_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if ! curl -fsSL --max-time 60 -o "$tmp/rar.tar.gz" "$UNRAR_URL"; then
    echo "ERROR: download failed. Set UNRAR_URL to a reachable mirror and retry." >&2
    exit 1
fi

tar xzf "$tmp/rar.tar.gz" -C "$tmp"
src="$(find "$tmp" -type f -name unrar -perm -u+x 2>/dev/null | head -1)"
[ -z "$src" ] && src="$(find "$tmp" -type f -name unrar | head -1)"
if [ -z "$src" ]; then
    echo "ERROR: no 'unrar' binary inside the archive." >&2
    exit 1
fi

cp "$src" "$TARGET"
chmod +x "$TARGET"
echo "Installed: $TARGET"
"$TARGET" 2>&1 | head -1
