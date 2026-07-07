"""STT backend factory.

Mac dev  -> WhisperSTTServiceMLX (Metal, distil-whisper-large-v3)
NVIDIA   -> WhisperSTTService    (faster-whisper CUDA int8)

Both are Pipecat SegmentedSTTService subclasses: they buffer audio between
VAD start/stop and emit a TranscriptionFrame per utterance.
"""

from __future__ import annotations

from loguru import logger

from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language

from realtime_ai.config import Settings


def _language(code: str) -> Language:
    try:
        return Language(code)
    except ValueError:
        return Language.EN


def create_stt(settings: Settings) -> STTService:
    backend = settings.resolved_stt_backend
    lang = _language(settings.stt_language)

    if backend == "mlx_whisper":
        from pipecat.services.whisper.stt import WhisperSTTServiceMLX

        logger.info(f"STT backend: mlx_whisper model={settings.resolved_stt_model}")
        return WhisperSTTServiceMLX(
            model=settings.resolved_stt_model,
            language=lang,
        )

    if backend == "faster_whisper":
        from pipecat.services.whisper.stt import WhisperSTTService

        device = "cuda" if settings.has_cuda else "cpu"
        logger.info(
            f"STT backend: faster_whisper model={settings.resolved_stt_model} "
            f"device={device} compute={settings.stt_compute_type}"
        )
        return WhisperSTTService(
            model=settings.resolved_stt_model,
            device=device,
            compute_type=settings.stt_compute_type,
            language=lang,
        )

    raise ValueError(f"Unknown STT backend: {backend}")
