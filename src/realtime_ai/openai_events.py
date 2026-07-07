"""OpenAI-Realtime event schema over the WebRTC `oai-events` data channel.

The existing client parses exactly two event types off this channel:

  - conversation.item.input_audio_transcription.completed  -> user speech text
  - response.done                                          -> assistant text + usage

We reproduce those shapes so the client needs zero changes. An Observer taps the
Pipecat pipeline (without altering frame flow) and pushes the events out.
"""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict

from loguru import logger

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


def make_transcription_completed(text: str, item_id: str) -> dict:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": text,
    }


def make_response_done(text: str, usage: dict | None) -> dict:
    prompt = (usage or {}).get("prompt_tokens", 0)
    completion = (usage or {}).get("completion_tokens", 0)
    cached = (usage or {}).get("cache_read_input_tokens", 0) or 0
    return {
        "type": "response.done",
        "event_id": _event_id(),
        "response": {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "status": "completed",
            "output": [
                {
                    "role": "assistant",
                    "content": [{"type": "audio", "transcript": text}],
                }
            ],
            # Local models are text-in/text-out, so audio_tokens are 0. The client
            # maps these onto its cost columns; audio cost ends up ~0 (our point).
            "usage": {
                "total_tokens": prompt + completion,
                "input_tokens": prompt,
                "output_tokens": completion,
                "input_token_details": {
                    "text_tokens": prompt,
                    "audio_tokens": 0,
                    "cached_tokens": cached,
                },
                "output_token_details": {
                    "text_tokens": completion,
                    "audio_tokens": 0,
                },
            },
        },
    }


class OaiEventsChannel:
    """Wraps the browser-created RTCDataChannel named `oai-events`.

    The channel appears asynchronously (after ICE), so sends before it opens are
    queued and flushed on open.
    """

    LABEL = "oai-events"

    def __init__(self) -> None:
        self._channel = None
        self._queue: list[str] = []

    def attach(self, channel) -> None:
        self._channel = channel
        logger.debug(f"oai-events channel attached (state={channel.readyState})")

        @channel.on("open")
        def _on_open():
            self._flush()

        if channel.readyState == "open":
            self._flush()

    def _flush(self) -> None:
        if self._channel is None:
            return
        for payload in self._queue:
            self._channel.send(payload)
        self._queue.clear()

    def send_event(self, event: dict) -> None:
        payload = json.dumps(event)
        ch = self._channel
        if ch is not None and ch.readyState == "open":
            ch.send(payload)
        else:
            self._queue.append(payload)


class OaiEventObserver(BaseObserver):
    """Taps pipeline frames and emits OpenAI events on the oai-events channel.

    The pipeline pushes each frame across multiple processor hops, so every frame
    is seen several times; we dedupe by frame id.
    """

    def __init__(self, channel: OaiEventsChannel) -> None:
        super().__init__()
        self._channel = channel
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._assistant_buf: list[str] = []
        self._last_usage: dict | None = None

    def _once(self, frame_id: str) -> bool:
        """Return True the first time a frame id is seen."""
        if frame_id in self._seen:
            return False
        self._seen[frame_id] = None
        if len(self._seen) > 2048:
            self._seen.popitem(last=False)
        return True

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame

        if isinstance(frame, TranscriptionFrame):
            if self._once(frame.id) and frame.text and frame.text.strip():
                self._channel.send_event(
                    make_transcription_completed(frame.text.strip(), f"item_{frame.id}")
                )

        elif isinstance(frame, LLMFullResponseStartFrame):
            if self._once(frame.id):
                self._assistant_buf.clear()

        elif isinstance(frame, LLMTextFrame):
            if self._once(frame.id) and frame.text:
                self._assistant_buf.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._once(frame.id):
                text = "".join(self._assistant_buf).strip()
                self._assistant_buf.clear()
                if text:
                    self._channel.send_event(make_response_done(text, self._last_usage))
                self._last_usage = None

        elif isinstance(frame, MetricsFrame):
            for md in frame.data:
                if isinstance(md, LLMUsageMetricsData):
                    u = md.value
                    self._last_usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                        "cache_read_input_tokens": u.cache_read_input_tokens or 0,
                    }
