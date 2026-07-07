# Integrating realtime-ai (drop-in for OpenAI Realtime)

The client and API already speak the OpenAI Realtime WebRTC protocol. Switching
to the self-hosted server is a **base-URL change** via env vars. Both default to
OpenAI when the var is unset, so this is fully reversible.

## 1. Start realtime-ai

```bash
cd realtime-ai
source .venv/bin/activate
ollama serve &                 # dev LLM (llama3:latest)
uvicorn realtime_ai.server:app --host 0.0.0.0 --port 8080
curl localhost:8080/health
```

## 2. Point the API at it

`sales-training-api-v2` — set in the API's environment:

```bash
REALTIME_BASE_URL=http://localhost:8080     # or your tunnel/GPU host URL
```

The change is in [session.service.ts](../../NibavLifts/sales-whisper/sales-training-api-v2/src/modules/session/session.service.ts):
when `REALTIME_BASE_URL` is not `api.openai.com`, the API calls our
`/v1/realtime/client_secrets` **without** an OpenAI key.

## 3. Point the client at it

`apps/sales-training` — set a Vite env var (e.g. `.env` / `environments/*`):

```bash
VITE_REALTIME_BASE_URL=http://localhost:8080   # same host the browser reaches
```

The change is in [curriculum/index.tsx](../../NibavLifts/nibav-internal-apps/ui/apps/sales-training/src/pages/curriculum/index.tsx):
the SDP offer POST now targets `${VITE_REALTIME_BASE_URL}/v1/realtime/calls`.

> The browser talks to the server **directly** for WebRTC (SDP + media), so
> `VITE_REALTIME_BASE_URL` must be reachable from the user's browser, not just
> from the API.

## 3b. Session grading (endSession) — also self-hosted

`endSession` grades the transcript with a plain `POST /v1/chat/completions`
(`response_format: json_object`). The realtime-ai server proxies that endpoint
to Ollama/vLLM, so grading runs on the **same host** — and unlike WebRTC, this
is plain HTTP that traverses a Cloudflare tunnel fine.

Set in the API's environment:

```bash
EVAL_BASE_URL=http://localhost:8080     # realtime-ai host (proxies to the LLM)
EVALUATION_MODEL=llama3:latest          # grading model (see note below)
```

The change is in [session.service.ts](../../NibavLifts/sales-whisper/sales-training-api-v2/src/modules/session/session.service.ts)
(`endSession`): when `EVAL_BASE_URL` isn't OpenAI, it calls the proxy with no key.

Notes:
- **Model choice:** `llama3:latest` is fast and returns valid JSON (~11s). `qwen3:8b`
  reasons a bit deeper but is slower (~26s). Both produced valid grading JSON in
  testing (Ollama's json mode forces valid JSON even for the thinking model).
- **Cost display:** `calculateEvalCost` already returns `$0` for models whose name
  contains `llama`/`ollama`, so `EVALUATION_MODEL=llama3:latest` shows accurate
  (zero) grading cost. `qwen3:8b` would fall through to gpt-4o pricing — broaden
  that guard if you use it.

## 4. What stays identical

- The client's WebRTC handshake, `oai-events` data-channel parsing, and usage
  accounting are unchanged.
- The API still stores `client_secret.value`, `sessionId`, transcript, and calls
  `endSession` for grading exactly as before.
- Voice keys (`alloy`, `coral`, ...) are mapped to Kokoro voices server-side
  (`backends/tts.py`); unknown keys fall back to the default voice.

## Voice mapping

Edit `OPENAI_TO_KOKORO_VOICE` in
[tts.py](../src/realtime_ai/backends/tts.py) to match your `AiVoices.key`
values to Kokoro voices (`af_heart`, `am_adam`, `bf_emma`, ...). List available
Kokoro voices from the `voices-v1.0.bin` file.

## Latency expectations

| Stage | Mac M3 Pro (dev) | NVIDIA (prod) |
|-------|------------------|---------------|
| STT (distil-whisper-large-v3) | ~2.5s/utterance | ~0.2–0.4s |
| LLM first token (llama3:8b / vLLM) | ~1–2s warm (12s cold) | ~0.1–0.3s |
| TTS first audio (Kokoro) | ~0.3–3s | ~0.1s |

Local laptop latency is a few seconds/turn — expected. The prod GPU stack
(faster-whisper CUDA int8 + vLLM FP8 + Kokoro CUDA) brings this near OpenAI.
Keep Ollama/vLLM **warm** (a model that unloads adds a big cold-start).
