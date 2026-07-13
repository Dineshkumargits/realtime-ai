"""Ephemeral-token session store.

Mirrors OpenAI's two-step handshake:

  1. API calls POST /v1/realtime/client_secrets with the session config
     (instructions, voice, turn detection ...). We mint a short-lived token and
     stash the config under it.
  2. Browser calls POST /v1/realtime/calls with `Authorization: Bearer <token>`.
     We look up the config and spin up a pipeline configured with it.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class SessionConfig:
    """Parsed subset of the OpenAI `session` object that we actually use."""

    instructions: str = ""
    voice: str | None = None
    model: str = "gpt-realtime-2"
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    vad_threshold: float | None = None
    vad_prefix_padding_ms: int | None = None
    vad_silence_duration_ms: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai_session(cls, session: dict[str, Any]) -> "SessionConfig":
        audio = session.get("audio", {}) or {}
        out = audio.get("output", {}) or {}
        inp = audio.get("input", {}) or {}
        in_rate = ((inp.get("format") or {}).get("rate")) or 24000
        out_rate = ((out.get("format") or {}).get("rate")) or 24000
        turn_detection = inp.get("turn_detection") or {}
        return cls(
            instructions=session.get("instructions", "") or "",
            voice=out.get("voice"),
            model=session.get("model", "gpt-realtime-2"),
            input_sample_rate=int(in_rate),
            output_sample_rate=int(out_rate),
            vad_threshold=turn_detection.get("threshold"),
            vad_prefix_padding_ms=turn_detection.get("prefix_padding_ms"),
            vad_silence_duration_ms=turn_detection.get("silence_duration_ms"),
            raw=session,
        )


@dataclass
class _Entry:
    config: SessionConfig
    expires_at: float
    used: bool = False


class SessionManager:
    def __init__(self, token_ttl_s: int = 600) -> None:
        self._ttl = token_ttl_s
        self._store: dict[str, _Entry] = {}

    def mint(self, config: SessionConfig) -> str:
        self._gc()
        token = f"ek_{secrets.token_urlsafe(32)}"
        self._store[token] = _Entry(config=config, expires_at=time.time() + self._ttl)
        logger.info(f"Minted ephemeral token (voice={config.voice}, model={config.model})")
        return token

    def resolve(self, token: str) -> SessionConfig | None:
        self._gc()
        entry = self._store.get(token)
        if entry is None:
            return None
        # One token -> one call. Mark used but keep briefly for renegotiation.
        entry.used = True
        return entry.config

    def _gc(self) -> None:
        now = time.time()
        stale = [t for t, e in self._store.items() if e.expires_at < now]
        for t in stale:
            self._store.pop(t, None)
