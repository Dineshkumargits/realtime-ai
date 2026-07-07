# Deploy runbook: Linux GPU server + Cloudflare

From-scratch steps to run realtime-ai on a Linux NVIDIA server, fronted by a
Cloudflare Tunnel. Read §0 first — it decides your networking.

---

## 0. Architecture (read this or the audio won't work)

Two independent paths reach the server, and **only one** goes through Cloudflare:

```
                 ┌─────────────── Cloudflare Tunnel (HTTPS) ───────────────┐
 Browser ── signaling: /v1/realtime/{client_secrets,calls}, /v1/chat/... ──┤
 (BDM)                                                                       ▼
                                                          realtime-ai :8080 (HTTP)
 Browser ═══════════ WebRTC media (RTP over UDP) ═══════════▶ server_public_ip:UDP
                     (does NOT go through Cloudflare)
```

- **Signaling = HTTP** → goes through the Cloudflare Tunnel. ✅
- **Audio = UDP (RTP/DTLS/SRTP)** → Cloudflare Tunnel **cannot** carry it. ❌
  The browser sends audio **directly** to the server's public IP over UDP.

So you need: a Cloudflare Tunnel for the HTTP endpoints **and** the server's
public IP reachable on UDP for media. If your BDMs sit behind restrictive
corporate firewalls that block UDP, add a TURN server (§8).

Prereq: the server has a **public IP** and you can open UDP ports on it.

---

## 1. Server prerequisites (one-time)

```bash
# NVIDIA driver (must already work):
nvidia-smi

# Docker:
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # re-login after this

# NVIDIA Container Toolkit (GPU inside containers):
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is visible to containers:
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

## 2. Firewall (UDP for media is the key part)

```bash
sudo ufw allow OpenSSH
# WebRTC media. --net host makes aiortc use the host's ephemeral UDP range.
sudo ufw allow 32768:60999/udp
sudo ufw enable
```

HTTP (:8080) stays **closed** to the internet — Cloudflare Tunnel reaches it via
`localhost`. (If you later add TURN, also open 3478/udp+tcp and the relay range.)

## 3. Get the code onto the server

```bash
# option A: git
git clone <your-repo-url> realtime-ai && cd realtime-ai
# option B: from your Mac
rsync -av --exclude .venv --exclude models --exclude .git \
  ~/Projects/ai/realtime-ai/ user@server:/opt/realtime-ai/
```

## 4. Configure

The compose file already sets the important envs. Adjust the LLM model / VRAM in
[docker-compose.cuda.yml](../docker-compose.cuda.yml) if needed. No `.env` is
required for a default run.

## 5. Build & start

```bash
docker compose -f docker-compose.cuda.yml up -d --build
docker compose -f docker-compose.cuda.yml logs -f realtime-ai   # watch first boot
```

First boot downloads model weights into the `models` + `hf-cache` volumes
(one-time). Wait for `Application startup complete`.

## 6. Smoke test on the server (before Cloudflare)

```bash
curl -s http://localhost:8080/health          # {"status":"ok",...}
# text LLM path (grading):
curl -s http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"say hi"}]}'
```

If both respond, the server is healthy. Media is tested end-to-end in §9.

---

## 7. Cloudflare Tunnel (dashboard) for the HTTP endpoints

You need a domain already added to Cloudflare.

1. **Zero Trust dashboard** → **Networks → Tunnels → Create a tunnel**.
2. Type **Cloudflared**, name it e.g. `realtime-ai`, **Save**.
3. Choose **Debian/64-bit**; the dashboard shows an install command containing a
   long token. Run it **on the server**:
   ```bash
   # installs cloudflared as a host systemd service, bound to your tunnel
   curl -fsSL https://pkg.cloudflare.com/cloudflared-stable-linux-amd64.deb -o cf.deb
   sudo dpkg -i cf.deb
   sudo cloudflared service install <TOKEN_FROM_DASHBOARD>
   ```
   The connector should show **HEALTHY** in the dashboard.
4. **Public Hostnames → Add a public hostname**:
   - Subdomain: `realtime` (→ `realtime.yourdomain.com`)
   - Service: **HTTP** → `localhost:8080`
   - Save.

> cloudflared runs on the host, so `localhost:8080` reaches the `--net host`
> realtime-ai container. No extra ports opened.

Test: `curl https://realtime.yourdomain.com/health` from anywhere → `ok`.

## 8. Point the client + API at the tunnel

**API** (`sales-training-api-v2` env):
```bash
REALTIME_BASE_URL=https://realtime.yourdomain.com
EVAL_BASE_URL=https://realtime.yourdomain.com
EVALUATION_MODEL=Qwen/Qwen2.5-7B-Instruct   # or llama-family for $0 cost display
```

**Client** (`apps/sales-training` build-time env):
```bash
VITE_REALTIME_BASE_URL=https://realtime.yourdomain.com
```
Rebuild/redeploy the client so the var is baked in.

---

## 9. Verify end-to-end

Open the sales-training app, start a simulation, and speak. In the server logs
you should see `Transcription: [...]`, an LLM generation, and `Bot started
speaking`, and you should **hear** the reply.

- ✅ You see the transcript in logs but hear **no audio** → media (UDP) isn't
  connecting. Go to §10.
- ✅ Everything works → done.

## 10. If audio doesn't connect (media path)

Symptom: signaling succeeds (transcript appears) but no audio flows, or ICE
fails in the browser console.

Cause: the browser can't reach the server on UDP (server behind strict NAT, or
the BDM is behind a corporate firewall that blocks UDP).

Fix — add a **TURN** server so media relays over a well-known port:

**Option A — Cloudflare TURN (stays in Cloudflare, recommended):**
1. Cloudflare dashboard → **Realtime → TURN** → create a TURN app → get **Key ID**
   + **API token**.
2. Generate short-lived TURN credentials and add them to the ICE config in
   **both** places:
   - server: `STUN_SERVER` / ICE list in [config.py](../src/realtime_ai/config.py)
   - client: the `iceServers` array in
     [curriculum/index.tsx](../../NibavLifts/nibav-internal-apps/ui/apps/sales-training/src/pages/curriculum/index.tsx)
   (This needs a small change to have the API mint TURN creds alongside the
   ephemeral token and pass them to the client — ask and I'll wire it.)

**Option B — self-hosted coturn:** run coturn with the server's public IP,
open `3478/udp,tcp` + a relay range, and add
`turn:server_public_ip:3478` with a shared secret to both ICE lists.

For BDMs on open networks, §7 alone (direct UDP) usually works; add TURN when you
see field users who can't connect.

---

## Troubleshooting quick table

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` fails in container | toolkit not configured | re-run §1 nvidia-ctk step |
| `/health` ok but 502 via tunnel | cloudflared can't reach :8080 | service URL must be `localhost:8080`; realtime-ai on `--net host` |
| Transcript in logs, no audio | UDP media blocked | §10 TURN |
| Re-downloads models each restart | volumes not persisted | keep the `models`/`hf-cache` named volumes |
| vLLM OOM | VRAM over-allocated | lower `--gpu-memory-utilization` |
| First call very slow | cold model load | pre-warm with one call after startup |
