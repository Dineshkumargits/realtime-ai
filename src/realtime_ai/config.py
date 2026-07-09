"""Central, env-driven configuration.

Everything that differs between the Apple-Silicon dev box and the NVIDIA prod
box is selected here so the rest of the code stays backend-agnostic.

Backend selection is by explicit env (STT_BACKEND / LLM_BACKEND / TTS_BACKEND).
If left as "auto", we pick sensible defaults from the detected platform:
  - macOS/arm64  -> mlx-whisper, ollama, kokoro-onnx(coreml/cpu)
  - linux + cuda -> faster-whisper(cuda), vllm(openai), kokoro-onnx(cuda)
"""

from __future__ import annotations

import platform
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_mac_arm() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_cuda() -> bool:
    # Cheap check that avoids importing torch at config time.
    import shutil

    return shutil.which("nvidia-smi") is not None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- server ----
    host: str = "0.0.0.0"
    # Uncommon default port to avoid clashing with other apps on the host.
    port: int = 18080
    log_level: str = "INFO"
    # Public base URL the browser reaches this server at (used for ICE / logs).
    public_base_url: str = "http://localhost:18080"

    # ---- backend selection ----
    # "auto" resolves from the platform in resolved_* below.
    stt_backend: Literal["auto", "mlx_whisper", "faster_whisper"] = "auto"
    llm_backend: Literal["auto", "ollama", "openai_compat"] = "auto"
    tts_backend: Literal["auto", "kokoro"] = "auto"

    # ---- STT ----
    # mlx model repo (mac) or faster-whisper model name (cuda/cpu)
    stt_model_mac: str = "mlx-community/distil-whisper-large-v3"
    # Must be a CTranslate2-converted repo (faster-whisper needs model.bin, not
    # the raw HF/PyTorch checkpoint that "distil-whisper/distil-large-v3" is).
    stt_model_cuda: str = "Systran/faster-distil-whisper-large-v3"
    stt_compute_type: str = "int8"  # faster-whisper only
    stt_language: str = "en"

    # ---- LLM ----
    # For ollama and any OpenAI-compatible endpoint (vLLM, etc).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # placeholder; vLLM/ollama ignore or accept any
    # Default to a non-thinking instruct model for low latency. Reasoning models
    # (e.g. qwen3:8b) emit hidden <think> tokens that add seconds of latency and
    # can return empty content under a token cap — for those, prepend "/no_think"
    # to the system prompt (see llm_system_prefix) or avoid them for voice.
    llm_model: str = "llama3:latest"
    # Optional text prepended to the system prompt (e.g. "/no_think " for qwen3).
    llm_system_prefix: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512

    # ---- TTS ----
    # kokoro-onnx model + voices file (downloaded once into models/).
    tts_model_path: str = "models/kokoro-v1.0.onnx"
    tts_voices_path: str = "models/voices-v1.0.bin"
    tts_sample_rate: int = 24000
    tts_default_voice: str = "af_heart"
    # onnxruntime execution providers, first available wins.
    # mac: CoreMLExecutionProvider then CPU. cuda: CUDAExecutionProvider then CPU.
    tts_providers_mac: str = "CoreMLExecutionProvider,CPUExecutionProvider"
    tts_providers_cuda: str = "CUDAExecutionProvider,CPUExecutionProvider"

    # ---- audio / transport ----
    audio_in_sample_rate: int = 24000
    audio_out_sample_rate: int = 24000
    stun_server: str = "stun:stun.l.google.com:19302"

    # ---- TURN (Cloudflare) ----
    # Needed when this server sits behind NAT (no directly-routable public IP):
    # STUN alone lets peers discover a public IP, but inbound UDP still can't
    # reach a NAT'd box without port-forwarding, which isn't practical against
    # aiortc's per-session ephemeral ports. TURN relays media through a public
    # relay instead. Optional -- unset means STUN-only (today's behavior).
    # Create a TURN app at Cloudflare dashboard -> Realtime -> TURN.
    cf_turn_key_id: str = ""
    cf_turn_api_token: str = ""
    cf_turn_ttl_seconds: int = 86400  # 24h; refreshed periodically while the server runs

    @property
    def turn_enabled(self) -> bool:
        return bool(self.cf_turn_key_id and self.cf_turn_api_token)

    # ---- VAD (Silero) turn detection; mirrors the OpenAI server_vad knobs ----
    vad_threshold: float = 0.7
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 800

    # ---- session lifecycle ----
    ephemeral_token_ttl_s: int = 600
    max_concurrent_sessions: int = 16

    # ---------- resolved (auto) backends ----------
    @property
    def resolved_stt_backend(self) -> str:
        if self.stt_backend != "auto":
            return self.stt_backend
        return "mlx_whisper" if _is_mac_arm() else "faster_whisper"

    @property
    def resolved_llm_backend(self) -> str:
        if self.llm_backend != "auto":
            return self.llm_backend
        # Both go through an OpenAI-compatible client; ollama is just a base_url.
        return "ollama"

    @property
    def resolved_tts_backend(self) -> str:
        return "kokoro" if self.tts_backend == "auto" else self.tts_backend

    @property
    def resolved_tts_providers(self) -> list[str]:
        raw = self.tts_providers_mac if _is_mac_arm() else self.tts_providers_cuda
        return [p.strip() for p in raw.split(",") if p.strip()]

    @property
    def resolved_stt_model(self) -> str:
        return self.stt_model_mac if _is_mac_arm() else self.stt_model_cuda

    @property
    def is_mac(self) -> bool:
        return _is_mac_arm()

    @property
    def has_cuda(self) -> bool:
        return _has_cuda()


@lru_cache
def get_settings() -> Settings:
    return Settings()
