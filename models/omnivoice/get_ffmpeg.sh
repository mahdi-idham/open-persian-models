#!/usr/bin/env bash
# get_ffmpeg.sh - vendor a STATIC ffmpeg next to the Dockerfile for the build.
# Idempotent: reuses an existing copy. The binary is NOT in git (77 MB);
# every machine that builds the image runs this once.
#
# Order: 1) already vendored  2) host static ffmpeg (common paths + PATH)
#        3) download pinned static build (Iran-proof: --speed-limit stall kill)
set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/ffmpeg"
URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
SHA256_TARBALL="abda8d77ce8309141f83ab8edf0596834087c52467f6badf376a6a2a4c87cf67"

# NB: ldd exits NON-ZERO for static binaries — neutralize it before grep,
# else pipefail makes is_static fail on exactly the files we want.
is_static() { (ldd "$1" 2>&1 || true) | grep -q "not a dynamic executable"; }

if [ -x "$DEST" ] && is_static "$DEST"; then
  echo "ffmpeg already vendored: $DEST ($(du -h "$DEST" | cut -f1))"
  exit 0
fi

# 1) common host locations first
for CAND in "$HOME/.local/bin/ffmpeg" /usr/local/bin/ffmpeg /usr/bin/ffmpeg; do
  if [ -x "$CAND" ] && is_static "$CAND"; then
    mkdir -p "$(dirname "$DEST")"; cp "$CAND" "$DEST"; chmod +x "$DEST"
    echo "copied static ffmpeg from host: $CAND"
    "$DEST" -version | head -1
    exit 0
  fi
done

# 2) download (kills stalled transfers: abort if <10 KB/s for 30 s)
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "downloading static ffmpeg (stall-proof: aborts if <10 KB/s for 30 s)..."
curl -fL --retry 3 --retry-delay 2 \
  --speed-limit 10240 --speed-time 30 \
  -o "$TMP/ff.tar.xz" "$URL"
echo "$SHA256_TARBALL  $TMP/ff.tar.xz" | sha256sum -c -
tar -xf "$TMP/ff.tar.xz" -C "$TMP"
mkdir -p "$(dirname "$DEST")"
cp "$TMP"/ffmpeg-*/ffmpeg "$DEST"
chmod +x "$DEST"

is_static "$DEST" || { echo "ERROR: downloaded ffmpeg is not static" >&2; exit 1; }
"$DEST" -version | head -1
echo "OK: $DEST"
