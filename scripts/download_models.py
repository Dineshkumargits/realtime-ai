"""Download the Kokoro-ONNX model + voices into ./models (atomic + verified).

MLX-Whisper and the Ollama/vLLM LLM download themselves on first use, so this
only handles Kokoro's two release artifacts.

Robustness: each file is fetched to a temporary ``.part`` and only renamed into
place once it meets a minimum expected size. A truncated/partial file therefore
never passes as complete (which previously left a corrupt 135 MB onnx behind).
"""

from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path

REL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
# name -> (url, minimum acceptable size in bytes)
FILES = {
    "kokoro-v1.0.onnx": (f"{REL}/kokoro-v1.0.onnx", 250_000_000),  # real ~310 MB
    "voices-v1.0.bin": (f"{REL}/voices-v1.0.bin", 20_000_000),  # real ~27 MB
}

socket.setdefaulttimeout(60)  # fail a stalled connection instead of hanging forever


def _progress(block: int, block_size: int, total: int) -> None:
    if total > 0 and block % 256 == 0:  # throttle output
        pct = min(100, block * block_size * 100 // total)
        sys.stdout.write(f"\r    {pct:3d}%")
        sys.stdout.flush()


def _fetch(name: str, url: str, min_size: int, models: Path) -> None:
    dest = models / name
    if dest.exists() and dest.stat().st_size >= min_size:
        print(f"[skip] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    if dest.exists():
        print(f"[warn] {name} is {dest.stat().st_size / 1e6:.1f} MB (< min); re-downloading")
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {name}")
    urllib.request.urlretrieve(url, part, _progress)
    size = part.stat().st_size
    if size < min_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{name} download too small ({size} bytes < {min_size}); aborting")
    part.rename(dest)
    print(f"\r[done] {name} ({size / 1e6:.1f} MB)")


def main() -> int:
    models = Path(__file__).resolve().parent.parent / "models"
    models.mkdir(exist_ok=True)
    for name, (url, min_size) in FILES.items():
        _fetch(name, url, min_size, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
