# Deployment & scaling

## ⚠️ WebRTC does NOT traverse a plain Cloudflare Tunnel

This is the single most important gotcha for the "tunnel via Cloudflare" plan.

`cloudflared` (Cloudflare Tunnel) proxies **HTTP/HTTPS/WebSocket only**. WebRTC
media (ICE/DTLS/SRTP) rides on **UDP**, which a standard tunnel cannot carry. So:

- ✅ Signaling (`POST /v1/realtime/calls`, `/client_secrets`, `/health`) works
  fine over a Cloudflare Tunnel.
- ❌ The audio itself (the actual RTP media) will **not** connect through it.

### Working options for public media

1. **Public host + direct UDP (simplest for a GPU box).** Put realtime-ai on a
   server with a public IP, open the UDP ephemeral range (or set a fixed range),
   and let aiortc's host/srflx candidates connect directly. Terminate HTTPS for
   the signaling endpoints with a normal reverse proxy (Caddy/Nginx). No tunnel.
2. **TURN server.** Run coturn with a public IP and add it to `STUN_SERVER`/ICE
   config. Media relays through TURN over UDP (or TCP/443 as fallback). Use this
   when the server is behind NAT. This is the standard production answer.
3. **Cloudflare Calls / Realtime (their WebRTC SFU),** if you specifically want
   Cloudflare in the media path — a different product from Tunnel.

Recommended: **GPU box with public IP + coturn**, signaling behind HTTPS. Keep
Cloudflare (Tunnel or proxy) only in front of the HTTP signaling if desired.

## Production backend stack (NVIDIA)

Set these env vars (see `.env.example`) and install `requirements-cuda.txt`:

```bash
STT_BACKEND=faster_whisper          # CUDA int8, auto-detected via nvidia-smi
STT_MODEL_CUDA=distil-whisper/distil-large-v3
LLM_BASE_URL=http://vllm:18000/v1   # vLLM OpenAI-compatible server
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  # non-thinking instruct model
TTS_BACKEND=kokoro                  # onnxruntime-gpu picks CUDAExecutionProvider
```

Run vLLM separately (its own container), FP8 for Blackwell:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --port 18000
```

## Concurrency & scaling (target: 10+ sessions)

Current design:
- `SmallWebRTCRequestHandler` (MULTIPLE mode) already supports N simultaneous
  peer connections; each gets its own Pipecat pipeline + observer.
- `MAX_CONCURRENT_SESSIONS` gates admission (503 when full).

To scale cleanly to 10+ on one GPU:
1. **Share heavy models across sessions.** MLX-Whisper caches weights globally, but
   `KokoroTTSService` and `SileroVADAnalyzer` currently load per session. For high
   concurrency, load one Kokoro ONNX session + one Silero model and inject them
   (subclass the services). On a GPU this also maximizes batching headroom.
2. **Batch LLM with vLLM.** vLLM continuous batching serves many streams from one
   model instance — point all sessions at it; no per-session LLM weights.
3. **Shared token store for multi-instance.** The ephemeral-token map is in-memory,
   so `client_secrets` and `calls` must hit the **same** process. For multiple
   replicas behind a LB, move `SessionManager` to Redis (sticky sessions also work).
4. **One worker process per GPU.** Pipecat pipelines are asyncio; a single process
   saturates a GPU. Scale out by adding GPUs/replicas, not threads.

## Docker / containerization

**Performance:** containers are not VMs — they share the host kernel, so there is
~0% CPU/GPU compute overhead. GPU work runs natively through the NVIDIA Container
Toolkit, within ~1–2% of bare metal. You do **not** lose the machine's power.

> Note: this containerization targets the **Linux NVIDIA server**. Docker on
> macOS cannot pass through Metal/MLX, so these images run CPU-only on a Mac —
> develop on the Mac bare-metal (Metal), deploy the containers on Linux (CUDA).
> The backend abstraction makes it the same code either way.

### Host prerequisites (Linux GPU server, one-time)

```bash
# NVIDIA driver must already be installed (nvidia-smi works on the host).
# Install the NVIDIA Container Toolkit:
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi   # verify
```

### Run

```bash
docker compose -f docker-compose.cuda.yml up -d --build
curl localhost:18080/health
```

Single container (no compose), pointing at an external vLLM:

```bash
docker build -f Dockerfile.cuda -t realtime-ai:cuda .
docker run --gpus all --net host \
  -v realtime-models:/app/models -v realtime-hf:/app/.cache/huggingface \
  -e LLM_BASE_URL=http://localhost:18000/v1 --env-file .env realtime-ai:cuda
```

`--net host` (or an explicit UDP range + advertised public IP) is required so
WebRTC media reaches the container. Bridge networking drops the RTP media.

### VRAM budget (both services share one GPU)

Measured on a **16GB RTX 5060 Ti**: Qwen2.5-7B fp8 weights alone take ~8.2GB.
`--gpu-memory-utilization 0.6` (9.6GB) left only 0.07GB for KV cache and vLLM
refused to start (`KV cache memory ... larger than available`). Fixed by capping
context length (sales-training prompts don't need 32K tokens) and raising
utilization so there's real KV cache headroom for concurrent sessions:

```
--gpu-memory-utilization 0.80   # 12.8GB of 16GB for vLLM
--max-model-len 8192            # shrinks KV cache needed per session
```

| Component | Model | ~VRAM |
|-----------|-------|-------|
| vLLM | Qwen2.5-7B-Instruct FP8 (weights + KV cache) | ~12.8 GB (`--gpu-memory-utilization 0.80` on a 16GB card) |
| STT | faster-whisper distil-large-v3 int8 | ~1.5–2 GB |
| TTS | Kokoro-82M ONNX | <1 GB |

Leaves ~3.2GB for STT+TTS on a 16GB card — tight but workable for a handful of
concurrent sessions. On a 24GB+ card, raise `--max-model-len` back toward 32768
and/or lower utilization slightly for more headroom. For heavy concurrency
(10+), give vLLM its own GPU rather than sharing with STT/TTS.

### First boot

The image ships **without** model weights (kept lean). `docker/entrypoint.sh`
downloads Kokoro into the `models` volume on first start; faster-whisper + Silero
download into the `hf-cache` volume on first use. Both volumes persist, so
restarts are instant. Pre-warm by making one `/v1/realtime/calls` before traffic.
