"""Per-session Pipecat cascade pipeline.

    transport.input() -> STT -> user_aggregator -> LLM -> TTS
                       -> transport.output() -> assistant_aggregator

Turn-taking + barge-in interruptions are handled by the Silero VAD attached to
the user aggregator (Pipecat 1.5 turn architecture): when the user starts
speaking while the bot is talking, the framework flushes TTS output and
truncates the assistant turn automatically.
"""

from __future__ import annotations

import array
import asyncio

from loguru import logger

import time

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from realtime_ai.backends import create_llm, create_stt, create_tts
from realtime_ai.config import Settings
from realtime_ai.openai_events import OaiEventObserver, OaiEventsChannel
from realtime_ai.session_manager import SessionConfig


# Silero VAD (and Whisper) want 16 kHz. WebRTC carries Opus, so the transport
# resamples the mic to whatever we set here regardless of the client's declared
# PCM rate. Output stays at Kokoro's native rate.
VAD_STT_INPUT_RATE = 16000


class _AudioProbe(FrameProcessor):
    """Diagnostic pass-through: logs whether mic audio is actually reaching the
    pipeline from the browser. Confirms/rules out media-path issues (NAT/TURN,
    muted mic, silent track) independent of STT/VAD behavior. Cheap -- one log
    line every ~2s, not per-frame.
    """

    def __init__(self) -> None:
        super().__init__()
        self._frames = 0
        self._bytes = 0
        self._peak = 0
        self._last_log = time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._frames += 1
            self._bytes += len(frame.audio)
            # int16 PCM peak amplitude -- distinguishes real speech from
            # silence/near-silence arriving over a working media path (e.g. a
            # muted/wrong input device, or gain lost across a TURN relay).
            samples = array.array("h")
            samples.frombytes(
                frame.audio[: len(frame.audio) - (len(frame.audio) % 2)]
            )
            if samples:
                self._peak = max(self._peak, max(abs(s) for s in samples))
            now = time.monotonic()
            if now - self._last_log >= 2.0:
                logger.debug(
                    f"AudioProbe: {self._frames} frames / {self._bytes} bytes / "
                    f"peak={self._peak} (of 32767) from browser in last "
                    f"{now - self._last_log:.1f}s"
                )
                self._frames = 0
                self._bytes = 0
                self._peak = 0
                self._last_log = now
        await self.push_frame(frame, direction)


def _vad_analyzer(settings: Settings, config: SessionConfig) -> SileroVADAnalyzer:
    """Client-requested turn_detection (session.audio.input.turn_detection)
    overrides the server's own VAD_* defaults when present -- e.g. a client
    asking for a longer silence_duration_ms to avoid cutting a speaker off on
    a mid-sentence pause.
    """
    threshold = config.vad_threshold if config.vad_threshold is not None else settings.vad_threshold
    prefix_padding_ms = (
        config.vad_prefix_padding_ms
        if config.vad_prefix_padding_ms is not None
        else settings.vad_prefix_padding_ms
    )
    silence_duration_ms = (
        config.vad_silence_duration_ms
        if config.vad_silence_duration_ms is not None
        else settings.vad_silence_duration_ms
    )
    return SileroVADAnalyzer(
        sample_rate=VAD_STT_INPUT_RATE,
        params=VADParams(
            confidence=threshold,
            start_secs=prefix_padding_ms / 1000.0,
            stop_secs=silence_duration_ms / 1000.0,
        ),
    )


async def run_session(
    connection: SmallWebRTCConnection,
    config: SessionConfig,
    settings: Settings,
    events_channel: OaiEventsChannel,
) -> None:
    """Build and run one voice session until the peer disconnects."""
    logger.info(f"Building pipeline (voice={config.voice}, sr={config.output_sample_rate})")

    stt = create_stt(settings)
    llm = create_llm(settings)
    tts = create_tts(settings, voice=config.voice)

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=VAD_STT_INPUT_RATE,
            audio_out_sample_rate=config.output_sample_rate,
        ),
    )

    system_prompt = (settings.llm_system_prefix or "") + (config.instructions or "")
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=_vad_analyzer(settings, config)),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            _AudioProbe(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        # Our client isn't an RTVI client; suppress Pipecat's RTVI chatter so the
        # oai-events data channel carries only the two OpenAI event types.
        enable_rtvi=False,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=VAD_STT_INPUT_RATE,
            audio_out_sample_rate=config.output_sample_rate,
        ),
        observers=[OaiEventObserver(events_channel)],
    )

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    logger.info(f"Pipeline running for pc_id={connection.pc_id}")
    await runner.run()
    logger.info(f"Pipeline finished for pc_id={connection.pc_id}")


def spawn_session(
    connection: SmallWebRTCConnection,
    config: SessionConfig,
    settings: Settings,
    events_channel: OaiEventsChannel,
) -> asyncio.Task:
    """Start run_session as a background task; errors are logged, not raised."""

    async def _guarded() -> None:
        try:
            await run_session(connection, config, settings, events_channel)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"Session crashed for pc_id={connection.pc_id}")

    return asyncio.create_task(_guarded())
