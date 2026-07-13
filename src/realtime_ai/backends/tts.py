"""TTS backend factory (Kokoro-ONNX).

Same class on Mac and NVIDIA; the ONNX Runtime execution provider differs
(CoreML/CPU vs CUDA) and is selected by onnxruntime at load time.

Kokoro ships its own voice set, so OpenAI voice keys (alloy, echo, ...) coming
from the existing client are mapped to the closest Kokoro voice here. Unknown
keys fall back to the configured default.
"""

from __future__ import annotations

import threading

from loguru import logger

from pipecat.services.kokoro.tts import KokoroTTSService, KokoroTTSSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from realtime_ai.backends.text_filters import StageDirectionTextFilter
from realtime_ai.config import Settings

# Process-wide cache for the loaded Kokoro ONNX engine. KokoroTTSService.run_tts
# reads the voice per-call (self._settings.voice), so the underlying engine is
# voice-agnostic and safe to share across sessions -- without this, every
# session would reload the ONNX model onto the GPU from scratch and never free
# the previous copy promptly, compounding the same VRAM exhaustion as STT.
_engine_cache: dict[tuple[str, str], object] = {}
_engine_cache_lock = threading.Lock()

# OpenAI realtime voice key -> Kokoro voice. Tweak to taste.
OPENAI_TO_KOKORO_VOICE: dict[str, str] = {
    "alloy": "af_alloy",
    "ash": "am_adam",
    "ballad": "bm_george",
    "coral": "af_heart",
    "echo": "am_echo",
    "sage": "af_sarah",
    "shimmer": "af_bella",
    "verse": "am_michael",
    "marin": "bf_emma",
    "cedar": "bm_lewis",
}


def resolve_voice(requested: str | None, default: str) -> str:
    """Map an incoming voice key to a Kokoro voice."""
    if not requested:
        return default
    if requested.startswith(("af_", "am_", "bf_", "bm_", "hf_", "hm_")):
        return requested  # already a Kokoro voice
    return OPENAI_TO_KOKORO_VOICE.get(requested.lower(), default)


def _cached_kokoro_engine(model_path: str, voices_path: str):
    """Drop-in replacement for kokoro_onnx.Kokoro that reuses one process-wide
    engine per (model_path, voices_path) instead of reloading the ONNX model
    onto the GPU on every call.
    """
    key = (model_path, voices_path)
    with _engine_cache_lock:
        engine = _engine_cache.get(key)
        if engine is None:
            from kokoro_onnx import Kokoro as _RealKokoro

            logger.info(f"Loading Kokoro TTS engine from {model_path} (cached for reuse)")
            engine = _RealKokoro(model_path, voices_path)
            _engine_cache[key] = engine
        return engine


def _patch_kokoro_caching() -> None:
    import pipecat.services.kokoro.tts as kokoro_tts_module

    if getattr(kokoro_tts_module, "_realtime_ai_patched", False):
        return
    kokoro_tts_module.Kokoro = _cached_kokoro_engine  # type: ignore[assignment]
    kokoro_tts_module._realtime_ai_patched = True  # type: ignore[attr-defined]


def create_tts(settings: Settings, voice: str | None = None) -> KokoroTTSService:
    _patch_kokoro_caching()
    kokoro_voice = resolve_voice(voice, settings.tts_default_voice)
    logger.info(f"TTS backend: kokoro voice={kokoro_voice} (requested={voice})")
    return KokoroTTSService(
        model_path=settings.tts_model_path,
        voices_path=settings.tts_voices_path,
        sample_rate=settings.tts_sample_rate,
        settings=KokoroTTSSettings(voice=kokoro_voice, language=Language.EN),
        text_filters=[MarkdownTextFilter(), StageDirectionTextFilter()],
    )
