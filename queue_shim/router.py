from __future__ import annotations

import os
from typing import Final

import httpx

LITELLM_BASE: Final = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
HEALTH_ENDPOINT: Final = f"{LITELLM_BASE}/health"
COMPLETIONS_ENDPOINT: Final = f"{LITELLM_BASE}/v1/chat/completions"


async def litellm_healthy(client: httpx.AsyncClient, health_url: str = HEALTH_ENDPOINT) -> bool:
    try:
        resp = await client.get(health_url, timeout=3.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
