"""Prove each backend runs on THIS machine, independent of Pipecat wiring.

Flow: Kokoro synthesizes a phrase -> MLX-Whisper transcribes it back ->
Ollama answers a prompt. If all three print sane output, the stack is viable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHRASE = "The quick brown fox jumps over the lazy dog."


def test_tts() -> tuple[np.ndarray, int]:
    from kokoro_onnx import Kokoro

    t0 = time.time()
    k = Kokoro(str(ROOT / "models/kokoro-v1.0.onnx"), str(ROOT / "models/voices-v1.0.bin"))
    samples, sr = k.create(PHRASE, voice="af_heart", speed=1.0, lang="en-us")
    dur = len(samples) / sr
    print(f"[TTS ] Kokoro ok: {dur:.2f}s audio @ {sr}Hz in {time.time()-t0:.2f}s")
    return np.asarray(samples, dtype=np.float32), sr


def test_stt(samples: np.ndarray, sr: int) -> None:
    import mlx_whisper

    t0 = time.time()
    # mlx-whisper accepts a float32 numpy array at 16kHz.
    if sr != 16000:
        import math

        n = int(len(samples) * 16000 / sr)
        idx = (np.arange(n) * (sr / 16000)).astype(int)
        idx = np.clip(idx, 0, len(samples) - 1)
        audio16 = samples[idx]
    else:
        audio16 = samples
    out = mlx_whisper.transcribe(
        audio16,
        path_or_hf_repo="mlx-community/distil-whisper-large-v3",
        language="en",
    )
    print(f"[STT ] MLX ok in {time.time()-t0:.2f}s -> {out['text'].strip()!r}")


def test_llm() -> None:
    from openai import OpenAI

    t0 = time.time()
    c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    r = c.chat.completions.create(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a terse assistant. Reply in one short sentence."},
            {"role": "user", "content": "Say hello and confirm you are running locally."},
        ],
        max_tokens=64,
    )
    print(f"[LLM ] Ollama ok in {time.time()-t0:.2f}s -> {r.choices[0].message.content.strip()!r}")


def main() -> int:
    ok = True
    try:
        samples, sr = test_tts()
        test_stt(samples, sr)
    except Exception as e:
        ok = False
        print(f"[TTS/STT] FAILED: {type(e).__name__}: {e}")
    try:
        test_llm()
    except Exception as e:
        ok = False
        print(f"[LLM ] FAILED: {type(e).__name__}: {e}")
    print("\nRESULT:", "ALL BACKENDS OK" if ok else "SOME BACKENDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
