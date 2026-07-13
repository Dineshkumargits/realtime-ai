"""FastAPI app exposing an OpenAI-Realtime-compatible surface.

Endpoints the existing client + API already speak:

  POST /v1/realtime/client_secrets   (API -> here)  mint ephemeral token
  POST /v1/realtime/calls?model=...  (browser -> here)  SDP offer -> SDP answer

Plus GET /health for probes.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    ConnectionMode,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

import asyncio

from realtime_ai.config import Settings, get_settings
from realtime_ai.openai_events import OaiEventsChannel
from realtime_ai.pipeline import spawn_session
from realtime_ai.session_manager import SessionConfig, SessionManager
from realtime_ai.turn import fetch_turn_ice_servers


class AppState:
    settings: Settings
    sessions: SessionManager
    webrtc: SmallWebRTCRequestHandler
    active: int = 0
    # Raw RTCIceServer-shaped dicts, exposed to the browser via
    # /v1/realtime/client_secrets so its RTCPeerConnection uses the same
    # TURN relay as our own server-side aiortc peer connection.
    ice_servers: list[dict] = []


state = AppState()


def _build_webrtc_handler(ice_servers: list[dict]) -> SmallWebRTCRequestHandler:
    return SmallWebRTCRequestHandler(
        ice_servers=[
            IceServer(
                urls=s["urls"],
                username=s.get("username"),
                credential=s.get("credential"),
            )
            for s in ice_servers
        ],
        connection_mode=ConnectionMode.MULTIPLE,
    )


async def _refresh_turn_loop(s: Settings) -> None:
    """Re-mint Cloudflare TURN credentials before they expire.

    Swaps in a fresh SmallWebRTCRequestHandler; existing connections only
    consult ice_servers during initial ICE gathering, so this is safe to do
    without disrupting sessions already connected.
    """
    if not s.turn_enabled:
        return
    # Refresh at 80% of TTL, floored at 60s so misconfigured tiny TTLs don't spin.
    interval = max(60, int(s.cf_turn_ttl_seconds * 0.8))
    while True:
        await asyncio.sleep(interval)
        try:
            servers = await fetch_turn_ice_servers(s)
            state.ice_servers = servers
            state.webrtc = _build_webrtc_handler(servers)
            logger.info("Refreshed Cloudflare TURN credentials")
        except Exception as exc:
            logger.error(f"TURN credential refresh failed, keeping existing: {exc}")


def _enable_ice_debug_logging(settings: Settings) -> None:
    """Surface aioice/aiortc stdlib logs (TURN allocation, candidate checks)
    -- but only at LOG_LEVEL=DEBUG.

    These libraries log through stdlib `logging`, which has no handler in this
    process (we use loguru) -- so TURN failures were being silently dropped.
    At DEBUG, aiortc logs a line per RTP/SCTP packet (dozens/sec), which
    drowns out everything else; only worth it when actively diagnosing a
    TURN/media-path issue, not by default.
    """
    import logging

    class _LoguruHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logger.opt(depth=6).log("DEBUG", f"[{record.name}] {record.getMessage()}")

    debug = settings.log_level.upper() == "DEBUG"
    for name in ("aioice", "aiortc"):
        lib_logger = logging.getLogger(name)
        if debug:
            lib_logger.setLevel(logging.DEBUG)
            lib_logger.addHandler(_LoguruHandler())
            lib_logger.propagate = False
        else:
            lib_logger.setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    state.settings = s
    logger.remove()
    logger.add(sys.stderr, level=s.log_level.upper())
    _enable_ice_debug_logging(s)
    state.sessions = SessionManager(token_ttl_s=s.ephemeral_token_ttl_s)
    state.ice_servers = await fetch_turn_ice_servers(s)
    state.webrtc = _build_webrtc_handler(state.ice_servers)
    refresh_task = asyncio.create_task(_refresh_turn_loop(s))
    logger.info(
        f"realtime-ai up | STT={s.resolved_stt_backend} LLM={s.resolved_llm_backend}"
        f"({s.llm_model}) TTS={s.resolved_tts_backend} | mac={s.is_mac} cuda={s.has_cuda}"
        f" | turn={'on' if s.turn_enabled else 'off'}"
    )
    yield
    refresh_task.cancel()
    await state.webrtc.close()


app = FastAPI(title="realtime-ai", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    s = state.settings
    return {
        "status": "ok",
        "active_sessions": state.active,
        "backends": {
            "stt": s.resolved_stt_backend,
            "llm": f"{s.resolved_llm_backend}:{s.llm_model}",
            "tts": s.resolved_tts_backend,
        },
    }


@app.post("/v1/realtime/client_secrets")
async def client_secrets(payload: dict) -> dict:
    """Mint an ephemeral token for a session config (OpenAI-compatible)."""
    session = payload.get("session", payload) or {}
    config = SessionConfig.from_openai_session(session)
    token = state.sessions.mint(config)
    return {
        "value": token,
        "expires_at": int(time.time()) + state.settings.ephemeral_token_ttl_s,
        "client_secret": {  # the API reads data.client_secret.value
            "value": token,
            "expires_at": int(time.time()) + state.settings.ephemeral_token_ttl_s,
        },
        "session": session,
        # Not part of OpenAI's schema -- our client reads this to build a
        # matching iceServers list so it uses the same TURN relay (if any) as
        # our server-side aiortc connection. Real OpenAI clients ignore
        # unknown fields, so this stays backward compatible.
        "ice_servers": state.ice_servers,
    }


@app.get("/v1/models")
async def list_models() -> Response:
    """Passthrough to the LLM backend's model list (some clients probe this)."""
    base = state.settings.llm_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{base}/models")
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """OpenAI-compatible text chat completions, forwarded to Ollama/vLLM.

    Used by the app's session-grading (`endSession`) call and any other text
    LLM use so everything runs through this one self-hosted host. Supports both
    streaming and non-streaming; the incoming OpenAI Authorization is dropped
    (the local backend doesn't need it).
    """
    base = state.settings.llm_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = await request.json()
    stream = bool(payload.get("stream"))

    if stream:

        async def gen():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, json=payload)
    return JSONResponse(content=r.json(), status_code=r.status_code)


@app.post("/v1/realtime/calls")
async def realtime_calls(
    request: Request,
    authorization: str | None = Header(default=None),
) -> Response:
    """Accept a raw SDP offer, run a pipeline, return the raw SDP answer."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    config = state.sessions.resolve(token) if token else None
    if config is None:
        # Allow tokenless local testing with a default persona.
        logger.warning("No/expired ephemeral token; using default session config")
        config = SessionConfig(instructions="You are a helpful voice assistant.")

    if state.active >= state.settings.max_concurrent_sessions:
        raise HTTPException(status_code=503, detail="Server at capacity")

    offer_sdp = (await request.body()).decode("utf-8")
    if "v=0" not in offer_sdp:
        raise HTTPException(status_code=400, detail="Body must be an SDP offer")

    _log_sdp_candidates("offer(browser)", offer_sdp)

    events_channel = OaiEventsChannel()

    async def on_connection(connection: SmallWebRTCConnection) -> None:
        # Capture the browser-created `oai-events` data channel so we can push
        # OpenAI events; pipecat keeps its own handler for the same channel.
        @connection.pc.on("datachannel")
        def _on_datachannel(channel):
            if channel.label == events_channel.LABEL:
                events_channel.attach(channel)

        @connection.event_handler("closed")
        async def _on_closed(_conn):
            state.active = max(0, state.active - 1)
            logger.info(f"Session closed; active={state.active}")

        state.active += 1
        logger.info(f"Session started; active={state.active}")
        spawn_session(connection, config, state.settings, events_channel)

    answer = await state.webrtc.handle_web_request(
        SmallWebRTCRequest(sdp=offer_sdp, type="offer"),
        on_connection,
    )
    if not answer:
        raise HTTPException(status_code=500, detail="Failed to produce SDP answer")

    _log_sdp_candidates("answer(server)", answer["sdp"])

    return Response(content=answer["sdp"], media_type="application/sdp")


def _log_sdp_candidates(label: str, sdp: str) -> None:
    """Log the ICE candidates in an SDP, summarized by type.

    The decisive TURN diagnostic: if the server's answer has no `typ relay`
    candidate, aiortc never got a Cloudflare TURN allocation, and a NAT'd
    server cannot be reached no matter what the browser offers.
    """
    candidates = [line.strip() for line in sdp.splitlines() if line.startswith("a=candidate")]
    types = [c.split(" typ ")[1].split(" ")[0] for c in candidates if " typ " in c]
    logger.info(f"SDP {label}: {len(candidates)} candidates, types={types}")
    for c in candidates:
        logger.info(f"SDP {label} candidate: {c}")
