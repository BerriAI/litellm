from __future__ import annotations

import json
import os
import uuid
from typing import Final

import httpx
import redis.asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from queue_shim.router import COMPLETIONS_ENDPOINT, HEALTH_ENDPOINT, LITELLM_BASE, litellm_healthy
from queue_shim.storage import CLAIM_TTL, Storage

WORKER_ID: Final = f"shim-{uuid.uuid4().hex[:8]}"
POLL_INTERVAL: Final = float(os.environ.get("POLL_INTERVAL", "1.0"))


def _replay(job, stream: bool):
    result = job.result or ""
    if stream:
        async def gen() -> str:
            for chunk in result.split("\n\n"):
                yield chunk + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(json.loads(result) if result else {})


def create_app(
    redis: redis.asyncio.Redis,
    litellm_base: str = LITELLM_BASE,
    completions_url: str = COMPLETIONS_ENDPOINT,
    health_url: str = HEALTH_ENDPOINT,
    poll_interval: float = POLL_INTERVAL,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    app = FastAPI()
    storage = Storage(redis)

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> object:
        client = http_client or app.state.http
        body = await request.json()
        model = str(body.get("model", ""))
        stream = bool(body.get("stream", False))

        key, _is_new = await storage.enqueue(model, body, stream)
        job = await storage.get(key)
        if job is None:
            return JSONResponse({"error": "job not found"}, status_code=404)

        if job.status in ("done", "failed") and job.result:
            return _replay(job, stream)

        if job.status == "pending" and await litellm_healthy(client, health_url):
            claimed = await storage.claim(key, WORKER_ID, ttl=CLAIM_TTL)
            if claimed is not None:
                if stream:
                    return await _forward_stream(client, completions_url, body, key, storage)
                return await _forward_oneshot(client, completions_url, body, key, storage)

        return JSONResponse({"job_id": key, "status": job.status}, status_code=202)

    @app.get("/queue")
    async def queue() -> JSONResponse:
        keys = await storage.list_pending(10000)
        items = []
        for k in keys:
            key = k.removeprefix("job:")
            job = await storage.get(key)
            if job:
                items.append({"key": k, "status": job.status, "model": job.model})
        return JSONResponse({"pending": len(items), "items": items})

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.http = http_client if http_client is not None else httpx.AsyncClient()

    return app


async def _forward_oneshot(
    client: httpx.AsyncClient,
    completions_url: str,
    body: dict[str, object],
    key: str,
    storage: Storage,
) -> object:
    try:
        resp = await client.post(completions_url, json=body, timeout=60.0)
        resp.raise_for_status()
        result = resp.json()
        result_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
        await storage.mark_done(key, result_json)
        return JSONResponse(result)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        await storage.mark_failed(key, str(exc))
        return JSONResponse({"error": str(exc)}, status_code=502)


async def _forward_stream(
    client: httpx.AsyncClient,
    completions_url: str,
    body: dict[str, object],
    key: str,
    storage: Storage,
) -> object:
    async def gen() -> str:
        chunks: list[str] = []
        try:
            async with client.stream("POST", completions_url, json=body, timeout=60.0) as resp:
                async for line in resp.aiter_lines():
                    chunks.append(line + "\n")
            await storage.mark_done(key, "\n\n".join(chunks))
            for chunk in chunks:
                yield chunk
            yield "\n\n"
        except httpx.HTTPError as exc:
            await storage.mark_failed(key, str(exc))
            yield f"error: {exc}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def main() -> None:
    redis = redis.asyncio.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    uvicorn.run(create_app(redis), host="0.0.0.0", port=4001)


if __name__ == "__main__":
    main()
