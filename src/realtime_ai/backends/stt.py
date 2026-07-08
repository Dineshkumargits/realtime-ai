"""STT backend factory.

Mac dev  -> WhisperSTTServiceMLX (Metal, distil-whisper-large-v3)
NVIDIA   -> WhisperSTTService    (faster-whisper CUDA int8)

Both are Pipecat SegmentedSTTService subclasses: they buffer audio between
VAD start/stop and emit a TranscriptionFrame per utterance.
"""

from __future__ import annotations

import threading

from loguru import logger

from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language

from realtime_ai.config import Settings

# Process-wide cache for the GPU/Metal-resident model objects. Without this,
# every session would reconstruct WhisperModel (faster-whisper) from scratch
# -- reloading weights onto the GPU each time and never freeing the previous
# copy promptly, which exhausts VRAM after a handful of sessions. The config
# (model/device/compute_type) never changes at runtime, so one model per
# process is correct; only the lightweight WhisperSTTService wrapper (a
# Pipecat frame processor, unsafe to share across pipelines) is per-session.
_model_cache: dict[tuple, object] = {}
_model_cache_lock = threading.Lock()


def _language(code: str) -> Language:
    try:
        return Language(code)
    except ValueError:
        return Language.EN


def _cached_faster_whisper_model(model_name: str, device: str, compute_type: str):
    """Drop-in replacement for faster_whisper.WhisperModel that reuses one
    process-wide instance per (model, device, compute_type) instead of
    reloading GPU weights on every call.
    """
    key = ("faster_whisper", model_name, device, compute_type)
    with _model_cache_lock:
        model = _model_cache.get(key)
        if model is None:
            from faster_whisper import WhisperModel as _RealWhisperModel

            logger.info(f"Loading faster-whisper model {model_name} onto {device} (cached for reuse)")
            model = _RealWhisperModel(model_name, device=device, compute_type=compute_type)
            _model_cache[key] = model
        return model


def _patch_faster_whisper_caching() -> None:
    """Monkey-patch the WhisperModel symbol WhisperSTTService._load() calls, so
    it resolves through our cache instead of constructing a fresh model each
    session. mlx-whisper needs no equivalent patch: it already ships its own
    ModelHolder class-level cache keyed by model path.
    """
    import pipecat.services.whisper.stt as whisper_stt_module

    if getattr(whisper_stt_module, "_realtime_ai_patched", False):
        return
    whisper_stt_module.WhisperModel = _cached_faster_whisper_model  # type: ignore[assignment]
    whisper_stt_module._realtime_ai_patched = True  # type: ignore[attr-defined]


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

        _patch_faster_whisper_caching()
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
