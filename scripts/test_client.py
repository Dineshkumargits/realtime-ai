"""Headless end-to-end test of the realtime-ai server via aiortc.

Mimics exactly what the browser does:
  1. POST /v1/realtime/client_secrets  -> ephemeral token
  2. Add a mic track (Kokoro-synthesized user speech + trailing silence),
     open the `oai-events` data channel, POST the SDP offer to /v1/realtime/calls,
     apply the answer.
  3. Collect oai-events + count inbound bot audio frames.

Pass criteria: we receive a `response.done` event (assistant replied) AND
inbound bot audio frames > 0.
"""

from __future__ import annotations

import asyncio
import fractions
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("REALTIME_BASE", "http://localhost:18080")
USER_LINE = "Hi, I saw your ad for a home elevator. How much does it cost?"
INSTRUCTIONS = (
    "You are a friendly potential customer on a phone call with a salesperson. "
    "Keep replies to one or two short sentences."
)


class PcmTrack(AudioStreamTrack):
    """Emits the given PCM once, then silence forever (holds the call open)."""

    kind = "audio"

    def __init__(self, pcm_i16: np.ndarray, sr: int, tail_s: float = 4.0):
        super().__init__()
        self.sr = sr
        self.spf = int(sr * 0.02)  # 20 ms
        tail = np.zeros(int(sr * tail_s), dtype=np.int16)
        self.data = np.concatenate([pcm_i16.astype(np.int16), tail])
        self.pos = 0
        self.ts = 0

    async def recv(self) -> AudioFrame:
        await asyncio.sleep(0.02)
        if self.pos < len(self.data):
            chunk = self.data[self.pos : self.pos + self.spf]
            self.pos += self.spf
            if len(chunk) < self.spf:
                chunk = np.pad(chunk, (0, self.spf - len(chunk)))
        else:
            chunk = np.zeros(self.spf, dtype=np.int16)
        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sr
        frame.pts = self.ts
        frame.time_base = fractions.Fraction(1, self.sr)
        self.ts += self.spf
        return frame


def synth_user_speech(sr: int = 24000) -> np.ndarray:
    from kokoro_onnx import Kokoro

    k = Kokoro(str(ROOT / "models/kokoro-v1.0.onnx"), str(ROOT / "models/voices-v1.0.bin"))
    samples, ksr = k.create(USER_LINE, voice="am_adam", speed=1.0, lang="en-us")
    samples = np.asarray(samples, dtype=np.float32)
    if ksr != sr:  # linear resample
        n = int(len(samples) * sr / ksr)
        idx = np.clip((np.arange(n) * ksr / sr).astype(int), 0, len(samples) - 1)
        samples = samples[idx]
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16)


async def wait_ice_complete(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _():
        if pc.iceGatheringState == "complete":
            done.set()

    await asyncio.wait_for(done.wait(), timeout=10)


async def main() -> int:
    print(f"[1] mint token via {BASE}/v1/realtime/client_secrets")
    r = requests.post(
        f"{BASE}/v1/realtime/client_secrets",
        json={
            "session": {
                "type": "realtime",
                "model": "gpt-realtime-2",
                "instructions": INSTRUCTIONS,
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "coral"},
                },
            }
        },
        timeout=10,
    )
    r.raise_for_status()
    token = r.json()["client_secret"]["value"]
    print(f"    token={token[:16]}...")

    print("[2] synth user speech (Kokoro)")
    pcm = synth_user_speech()

    events: list[dict] = []
    audio_frames = {"n": 0}

    pc = RTCPeerConnection()
    pc.addTrack(PcmTrack(pcm, 24000))
    dc = pc.createDataChannel("oai-events")

    @dc.on("message")
    def on_msg(msg):
        try:
            ev = json.loads(msg)
            events.append(ev)
            t = ev.get("type")
            if t == "conversation.item.input_audio_transcription.completed":
                print(f"    <user> {ev.get('transcript')!r}")
            elif t == "response.done":
                tr = ev["response"]["output"][0]["content"][0].get("transcript")
                print(f"    <bot>  {tr!r}")
        except Exception:
            pass

    @pc.on("track")
    def on_track(track):
        async def drain():
            while True:
                try:
                    await track.recv()
                    audio_frames["n"] += 1
                except Exception:
                    break

        asyncio.ensure_future(drain())

    print("[3] SDP offer -> /v1/realtime/calls")
    await pc.setLocalDescription(await pc.createOffer())
    await wait_ice_complete(pc)
    resp = requests.post(
        f"{BASE}/v1/realtime/calls?model=gpt-realtime-2",
        data=pc.localDescription.sdp,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/sdp"},
        timeout=15,
    )
    resp.raise_for_status()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=resp.text, type="answer"))
    print("    connected; listening 35s...")

    await asyncio.sleep(35)
    await pc.close()

    from collections import Counter

    types = Counter(e.get("type") for e in events)
    got_response = any(e.get("type") == "response.done" for e in events)
    print(f"\n[result] events={len(events)} bot_audio_frames={audio_frames['n']} "
          f"response.done={got_response}")
    print(f"[event types] {dict(types)}")
    ok = got_response and audio_frames["n"] > 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
