from __future__ import annotations

import asyncio
import json
from typing import Final

import httpx
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient

from queue_shim.app import create_app
from queue_shim.storage import Storage, idem_key

COMPLETIONS_URL: Final = "http://litellm.test/v1/chat/completions"
HEALTH_URL: Final = "http://litellm.test/health"
LITELLM_BASE: Final = "http://litellm.test"
MODEL: Final = "test-model"


def _body() -> dict[str, object]:
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}


def _make_client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _list_pending(storage: Storage) -> list[str]:
    return asyncio.run(storage.list_pending(100))


def _get_job(storage: Storage, key: str):
    return asyncio.run(storage.get(key))


def _mark_done(storage: Storage, key: str, result: str) -> None:
    asyncio.run(storage.mark_done(key, result))


def test_unhealthy_request_held_then_completes_when_healthy() -> None:
    redis = FakeRedis(decode_responses=True)
    storage = Storage(redis)

    healthy_holder: list[bool] = [False]
    completions_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if healthy_holder[0]:
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(503, json={"status": "unhealthy"})
        if request.url.path == "/v1/chat/completions":
            completions_calls.append("called")
            return httpx.Response(200, json={"id": "cmpl-hold", "choices": [{"message": {"content": "recovered"}}]})
        return httpx.Response(404)

    http = _make_client(handler)
    app = create_app(
        redis,
        litellm_base=LITELLM_BASE,
        completions_url=COMPLETIONS_URL,
        health_url=HEALTH_URL,
        poll_interval=0.05,
        http_client=http,
    )

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 202

        pending = [k.removeprefix("job:") for k in _list_pending(storage)]
        assert len(pending) == 1

        key = pending[0]
        job = _get_job(storage, key)
        assert job is not None
        assert job.status == "pending"

    healthy_holder[0] = True
    with TestClient(app) as client:
        resp2 = client.post("/v1/chat/completions", json=_body())
        assert resp2.status_code == 200
        assert "recovered" in resp2.text
        assert completions_calls, "litellm completions endpoint must have been called after recovery"


def test_idempotent_retry_dedupes() -> None:
    redis = FakeRedis(decode_responses=True)
    storage = Storage(redis)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"id": "cmpl-dedupe", "choices": [{"message": {"content": "deduped"}}]})

    http = _make_client(handler)
    app = create_app(
        redis,
        litellm_base=LITELLM_BASE,
        completions_url=COMPLETIONS_URL,
        health_url=HEALTH_URL,
        poll_interval=0.05,
        http_client=http,
    )

    with TestClient(app) as client:
        resp1 = client.post("/v1/chat/completions", json=_body())
        resp2 = client.post("/v1/chat/completions", json=_body())

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert json.loads(resp1.text) == json.loads(resp2.text)

        pending = _list_pending(storage)
        assert len(pending) == 0, "done job must not remain in the pending queue"


def test_done_job_replays_immediately() -> None:
    redis = FakeRedis(decode_responses=True)
    storage = Storage(redis)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"id": "cmpl-replay", "choices": [{"message": {"content": "replay"}}]})

    http = _make_client(handler)
    app = create_app(
        redis,
        litellm_base=LITELLM_BASE,
        completions_url=COMPLETIONS_URL,
        health_url=HEALTH_URL,
        poll_interval=0.05,
        http_client=http,
    )

    with TestClient(app) as client:
        resp1 = client.post("/v1/chat/completions", json=_body())
        assert resp1.status_code == 200

        pending = [k.removeprefix("job:") for k in _list_pending(storage)]
        assert len(pending) == 0, "done job must not remain in the pending queue"

        body = _body()
        key = idem_key(body["model"], body)
        job = _get_job(storage, key)
        assert job is not None
        assert job.status == "done"

        resp2 = client.post("/v1/chat/completions", json=_body())
        assert resp2.status_code == 200
        assert "replay" in resp2.text
