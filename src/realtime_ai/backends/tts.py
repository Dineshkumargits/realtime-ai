"""TTS backend factory (Kokoro-ONNX).

Same class on Mac and NVIDIA; the ONNX Runtime execution provider differs
(CoreML/CPU vs CUDA) and is selected by onnxruntime at load time.

Kokoro ships its own voice set, so OpenAI voice keys (alloy, echo, ...) coming
from the existing client are mapped to the closest Kokoro voice here. Unknown
keys fall back to the configured default.
"""

from __future__ import annotations

from loguru import logger

from pipecat.services.kokoro.tts import KokoroTTSService, KokoroTTSSettings
from pipecat.transcriptions.language import Language

from realtime_ai.config import Settings

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


def create_tts(settings: Settings, voice: str | None = None) -> KokoroTTSService:
    kokoro_voice = resolve_voice(voice, settings.tts_default_voice)
    logger.info(f"TTS backend: kokoro voice={kokoro_voice} (requested={voice})")
    return KokoroTTSService(
        model_path=settings.tts_model_path,
        voices_path=settings.tts_voices_path,
        sample_rate=settings.tts_sample_rate,
        settings=KokoroTTSSettings(voice=kokoro_voice, language=Language.EN),
    )
