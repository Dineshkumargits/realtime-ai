# realtime-ai

A self-hosted, **OpenAI-Realtime-API-compatible** voice server. Drop-in
replacement for `api.openai.com/v1/realtime/*` so the existing sales-training
client + API keep working with only a base-URL change.

Pipeline (Pipecat, async cascading, streaming end-to-end):

```
mic PCM ─▶ Silero VAD ─▶ STT ─▶ context ─▶ LLM ─▶ TTS ─▶ PCM out
                                    (streaming tokens → word-chunk TTS)
```

## Backend matrix (env-switched, one codebase)

| Stage | Mac dev (this box)              | NVIDIA prod                          |
|-------|---------------------------------|--------------------------------------|
| STT   | mlx-whisper (Metal)             | faster-whisper CUDA int8             |
| VAD   | Silero (onnx)                   | Silero (onnx)                        |
| LLM   | Ollama `qwen3:8b` (`/v1`)       | vLLM FP8 (`/v1`, OpenAI-compatible)  |
| TTS   | Kokoro-ONNX (CoreML/CPU)        | Kokoro-ONNX (CUDA EP)                |
| Transport | Pipecat SmallWebRTC (aiortc)| same                                 |

Selection lives in [config.py](src/realtime_ai/config.py); `*_BACKEND=auto`
resolves from the platform.

## OpenAI-compat surface (what the client/API already speak)

- `POST /v1/realtime/client_secrets` → `{ client_secret: { value } }`
- `POST /v1/realtime/calls?model=...` (raw SDP offer in, SDP answer out)
- data channel `oai-events`:
  `conversation.item.input_audio_transcription.completed`, `response.done`

## Quick start (Mac dev)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt
python scripts/download_models.py        # Kokoro onnx + voices
cp .env.example .env
ollama serve &                           # if not already running
uvicorn realtime_ai.server:app --host 0.0.0.0 --port 8080
```

Then point the client + API at `http://localhost:8080` (see `docs/INTEGRATION.md`).

## Prod (NVIDIA)

```bash
pip install -r requirements-cuda.txt
# run vLLM separately, set LLM_BASE_URL to it
```

See `docs/` for the Docker + Cloudflare-tunnel setup.
