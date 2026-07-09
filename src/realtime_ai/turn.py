"""Cloudflare TURN credential minting.

Used when this server has no directly-routable public IP (behind a home/office
router's NAT): STUN alone tells peers the server's public IP, but inbound UDP
still can't reach the box without router port-forwarding, which isn't
practical against aiortc's per-session ephemeral ports. TURN relays media
through a public Cloudflare relay instead, sidestepping the NAT entirely.

Credentials are short-lived (see Settings.cf_turn_ttl_seconds) and must be
minted server-side, then handed to BOTH ICE agents: our own aiortc peer
connection (server.py's SmallWebRTCRequestHandler) and the browser's
RTCPeerConnection (returned to the API/client via /v1/realtime/client_secrets
and threaded through to the frontend's iceServers).
"""

from __future__ import annotations

import httpx
from loguru import logger

from realtime_ai.config import Settings

CF_TURN_ENDPOINT = "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate"


async def fetch_turn_ice_servers(settings: Settings) -> list[dict]:
    """Return a list of RTCIceServer-shaped dicts (STUN + TURN if configured).

    Falls back to STUN-only (today's behavior) if TURN isn't configured or the
    Cloudflare API call fails -- never raises, so a TURN outage degrades
    gracefully to the existing direct-UDP path instead of breaking sessions.
    """
    stun_only = [{"urls": [settings.stun_server]}]
    if not settings.turn_enabled:
        return stun_only

    url = CF_TURN_ENDPOINT.format(key_id=settings.cf_turn_key_id)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.cf_turn_api_token}"},
                json={"ttl": settings.cf_turn_ttl_seconds},
            )
        resp.raise_for_status()
        data = resp.json()
        ice_servers = data.get("iceServers")
        if not ice_servers:
            raise ValueError(f"Cloudflare TURN response missing iceServers: {data}")
        # Cloudflare returns either one dict or a list; normalize to a list.
        servers = ice_servers if isinstance(ice_servers, list) else [ice_servers]
        logger.info(f"Minted Cloudflare TURN credentials (ttl={settings.cf_turn_ttl_seconds}s)")
        return servers + stun_only
    except Exception as exc:
        logger.error(f"Cloudflare TURN credential mint failed, falling back to STUN-only: {exc}")
        return stun_only
