from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Final, Mapping

import httpx
from redis.asyncio import Redis

from .router import LITELLM_BASE, litellm_healthy
from .storage import CLAIM_TTL, JOB_PREFIX, Storage

__all__ = ["worker_main"]

WORKER_ID: Final = f"{socket.gethostname()}-{os.getpid()}"
SCAN_INTERVAL: Final = float(os.environ.get("SCAN_INTERVAL", "0.5"))
SICK_INTERVAL: Final = 2.0


async def worker_main(
    redis: Redis, litellm_base: str = LITELLM_BASE, http_client: httpx.AsyncClient | None = None
) -> None:
    storage = Storage(redis)
    completions_url = f"{litellm_base}/v1/chat/completions"
    health_url = f"{litellm_base}/health"

    async def run(client: httpx.AsyncClient) -> None:
        while True:
            if await litellm_healthy(client, health_url):
                for raw in await storage.list_pending(50):
                    key = raw.removeprefix(JOB_PREFIX)
                    claimed = await storage.claim(key, WORKER_ID, ttl=CLAIM_TTL)
                    if claimed is None:
                        continue
                    await _process(client, completions_url, claimed.model, claimed.body, storage, key)
            else:
                await asyncio.sleep(SICK_INTERVAL)
            await asyncio.sleep(SCAN_INTERVAL)

    if http_client is not None:
        await run(http_client)
        return
    async with httpx.AsyncClient(timeout=300.0) as http:
        await run(http)


async def _process(
    http: httpx.AsyncClient,
    completions_url: str,
    model: str,
    body: Mapping[str, object],
    storage: Storage,
    key: str,
) -> None:
    payload: dict[str, object] = dict(body)
    payload["model"] = model
    try:
        if payload.get("stream", False):
            chunks: list[str] = []
            async with http.stream("POST", completions_url, json=payload, timeout=300.0) as resp:
                async for line in resp.aiter_lines():
                    chunks.append(line)
            await storage.mark_done(key, "\n\n".join(chunks))
        else:
            resp = await http.post(completions_url, json=payload, timeout=300.0)
            await storage.mark_done(key, json.dumps(resp.json(), sort_keys=True, separators=(",", ":")))
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        await storage.mark_failed(key, str(exc))


def main() -> None:
    import redis.asyncio

    r = redis.asyncio.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    asyncio.run(worker_main(r))


if __name__ == "__main__":
    main()
