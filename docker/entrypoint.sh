#!/usr/bin/env bash
# Ensure model weights exist (on the mounted volume) before starting the server.
# Kokoro files are fetched here; faster-whisper + Silero download themselves into
# HF_HOME on first use, which is also volume-backed so it persists across restarts.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/app/models}"
mkdir -p "$MODELS_DIR" "${HF_HOME:-/app/.cache/huggingface}"

if [ ! -s "$MODELS_DIR/kokoro-v1.0.onnx" ] || [ ! -s "$MODELS_DIR/voices-v1.0.bin" ]; then
  echo "[entrypoint] Kokoro models missing — downloading into $MODELS_DIR"
  python3.11 scripts/download_models.py
else
  echo "[entrypoint] Kokoro models present"
fi

exec "$@"
