"""Download the Kokoro-ONNX model + voices into ./models.

MLX-Whisper and the Ollama LLM download themselves on first use, so this only
handles Kokoro's two release artifacts.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = {
    "kokoro-v1.0.onnx": f"{REL}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{REL}/voices-v1.0.bin",
}


def _progress(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, block * block_size * 100 // total)
    sys.stdout.write(f"\r    {pct:3d}%")
    sys.stdout.flush()


def main() -> int:
    models = Path(__file__).resolve().parent.parent / "models"
    models.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = models / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[get ] {name}")
        urllib.request.urlretrieve(url, dest, _progress)
        print(f"\r[done] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
