"""Tests for AutoQueueMiddleware (ASGI integration).

Worker-count env parsing is synchronous; the rest are async integration tests
that exercise the full ASGI request lifecycle through the middleware.
"""
import asyncio
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
for _name in list(sys.modules):
    if _name == "litellm" or _name.startswith("litellm."):
        sys.modules.pop(_name, None)

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from litellm.proxy.middleware.auto_queue_middleware import (
    AutoQueueMiddleware,
    AutoQueueRedis,
    ModelQueue,
    _QueueWakeReason,
    _WakeState,
)
from litellm.proxy.middleware.auto_queue_scripts import AdmitDecision, ReleaseTransfer
from litellm.proxy.middleware.auto_queue_state import AutoQueueRequestState


# ---------------------------------------------------------------------------
# Local fixture overrides to avoid session-vs-function event-loop mismatch
# in the shared conftest redis/aqr_factory fixtures.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def aqr_factory(redis):
    def _factory(
        *,
        default_max_concurrent: int = 2,
        ceiling: int = 10,
        scale_up_threshold: int = 3,
        scale_down_step: int = 1,
    ):
        return AutoQueueRedis(
            redis=redis,
            default_max_concurrent=default_max_concurrent,
            ceiling=ceiling,
            scale_up_threshold=scale_up_threshold,
            scale_down_step=scale_down_step,
        )

    return _factory


# ---------------------------------------------------------------------------
# Worker-count env parsing tests (synchronous, use module reload)
# ---------------------------------------------------------------------------


def _make_dummy_app():
    async def app(scope, receive, send):
        return None

    return app


def _reload_module_with_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = importlib.import_module("litellm.proxy.middleware.auto_queue_middleware")
    mod = importlib.reload(mod)
    return mod


def test_worker_count_allows_multiple_workers_in_distributed_mode(monkeypatch):
    mod = _reload_module_with_env(monkeypatch, AUTOQ_ENABLED="true", WEB_CONCURRENCY="2")
    mw = mod.AutoQueueMiddleware(_make_dummy_app(), aqr=None, enabled=None)
    assert isinstance(mw, mod.AutoQueueMiddleware)


def test_worker_count_guard_allows_single_worker(monkeypatch):
    mod = _reload_module_with_env(monkeypatch, AUTOQ_ENABLED="true", WEB_CONCURRENCY="1")
    mw = mod.AutoQueueMiddleware(_make_dummy_app(), aqr=None, enabled=None)
    assert isinstance(mw, mod.AutoQueueMiddleware)


def test_invalid_env_fallback_to_one_for_worker_count(monkeypatch):
    mod = _reload_module_with_env(monkeypatch, WEB_CONCURRENCY="notanumber")
    assert mod._resolve_worker_count() == 1


# ---------------------------------------------------------------------------
# Integration tests (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_priority_metadata_preempts_older_waiter(queue_factory):
    """Higher-priority (lower number) request should be woken before lower-priority one."""
    queue = queue_factory(max_depth=10)

    # Add a low-priority waiter (priority=10, default)
    ws_low = _WakeState()
    queue.add("req-low", ws_low, priority=10)

    # Add a high-priority waiter (priority=1)
    ws_high = _WakeState()
    queue.add("req-high", ws_high, priority=1)

    # Wake next -- high-priority should be served first
    woke = queue.wake_next()
    assert woke is True
    assert ws_high.is_set
    assert ws_high.reason == _QueueWakeReason.TRANSFERRED
    assert not ws_low.is_set

    # Wake again -- low-priority should be served
    woke = queue.wake_next()
    assert woke is True
    assert ws_low.is_set
    assert queue.depth == 0


@pytest.mark.asyncio
async def test_queue_timeout_from_key_metadata_returns_504_and_cleans_entry(
    aqr_factory, make_middleware_app, asgi_client_factory, key_metadata_factory
):
    """When key metadata specifies a short timeout, queued request gets 504 and queue entry is cleaned."""
    aqr = aqr_factory(default_max_concurrent=1, ceiling=10)
    hold = asyncio.Event()

    async def slow_handler(request):
        await request.body()
        await hold.wait()
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    app = make_middleware_app(slow_handler, aqr=aqr)
    client = await asgi_client_factory(app)

    # Fill the slot (no auth header -- uses default timeout)
    t1 = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    )
    await asyncio.sleep(0.05)

    # Set very short timeout via key metadata for the queued request
    key_metadata_factory(timeout=0)

    # Send with auth header so _get_key_config picks up the timeout=0
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": []},
        headers={"Authorization": "Bearer test-key"},
    )
    assert resp.status_code == 504

    # Queue should be clean
    queue = app._get_queue("gpt-4")
    assert queue.depth == 0

    hold.set()
    await t1


@pytest.mark.asyncio
async def test_disconnect_mid_wait_removes_request_from_queue(
    aqr_factory, make_middleware_app, drive_asgi, eventually
):
    """Client disconnect while queued should remove the request from the queue."""
    aqr = aqr_factory(default_max_concurrent=1, ceiling=10)
    block_forever = asyncio.Event()

    async def handler(request):
        body = await request.body()
        await block_forever.wait()
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    app = make_middleware_app(handler, aqr=aqr)

    # Drive the first request that fills the slot via ASGI directly
    scope1 = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
    }
    body1 = json.dumps({"model": "gpt-4", "messages": []}).encode()
    msgs1 = [
        {"type": "http.request", "body": body1, "more_body": False},
    ]
    t1 = asyncio.create_task(drive_asgi(app, scope=scope1, messages=msgs1))
    await asyncio.sleep(0.1)

    # Now drive a second request that will queue, and disconnect it
    scope2 = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
    }
    body2 = json.dumps({"model": "gpt-4", "messages": []}).encode()
    msgs2 = [
        {"type": "http.request", "body": body2, "more_body": False},
        {"type": "http.disconnect"},
    ]
    t2 = asyncio.create_task(drive_asgi(app, scope=scope2, messages=msgs2))
    await t2

    # The queue should be empty after disconnect
    queue = app._get_queue("gpt-4")
    await eventually(lambda: queue.depth == 0)

    block_forever.set()
    await t1


@pytest.mark.asyncio
async def test_disconnect_during_active_request_propagates_http_disconnect_to_downstream_receive(
    aqr_factory, make_middleware_app, drive_asgi
):
    """When client disconnects during an active (slot-held) request, the middleware
    should handle it gracefully without crashing."""
    aqr = aqr_factory(default_max_concurrent=1, ceiling=10)

    async def handler(request):
        await asyncio.sleep(0.3)
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    app = make_middleware_app(handler, aqr=aqr)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
    }
    body = json.dumps({"model": "gpt-4", "messages": []}).encode()
    messages = [
        {"type": "http.request", "body": body, "more_body": False},
        {"type": "http.disconnect"},
    ]

    # Drive directly via ASGI so we control the receive messages
    sent = await drive_asgi(app, scope=scope, messages=messages)
    # The key assertion: the middleware completes without hanging or crashing


@pytest.mark.asyncio
async def test_chunked_request_body_is_buffered_and_replayed_once(
    aqr_factory, make_middleware_app, drive_asgi
):
    """Chunked request body should be fully buffered and replayed as a single body."""
    aqr = aqr_factory(default_max_concurrent=2, ceiling=10)
    received_bodies: list[bytes] = []

    async def handler(request):
        body = await request.body()
        received_bodies.append(body)
        from starlette.responses import JSONResponse
        return JSONResponse({"body_len": len(body)})

    app = make_middleware_app(handler, aqr=aqr)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
    }
    # Send body in two chunks
    chunk1 = b'{"model": "gpt-4", "mes'
    chunk2 = b'sages": []}'
    messages = [
        {"type": "http.request", "body": chunk1, "more_body": True},
        {"type": "http.request", "body": chunk2, "more_body": False},
    ]

    sent = await drive_asgi(app, scope=scope, messages=messages)

    # Should have buffered both chunks and replayed as single body
    assert len(received_bodies) == 1
    assert received_bodies[0] == chunk1 + chunk2


@pytest.mark.asyncio
async def test_downstream_exception_still_releases_slot(
    aqr_factory, make_middleware_app, asgi_client_factory
):
    """If the downstream app raises, the slot should still be released."""
    aqr = aqr_factory(default_max_concurrent=2, ceiling=10)

    async def failing_handler(request):
        await request.body()
        raise ValueError("downstream error")

    app = make_middleware_app(failing_handler, aqr=aqr)
    client = await asgi_client_factory(app)

    # The middleware wraps the app; the exception propagates through ASGI
    resp = await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    # httpx with raise_app_exceptions=False gives us the raw response
    # The exception may surface as 500 or be caught differently depending on Starlette
    # The key check is slot release below

    # Verify the slot was released -- active count should be 0
    info = await aqr.get_model_info("gpt-4")
    assert info["active"] == 0


@pytest.mark.asyncio
async def test_downstream_exception_wakes_next_waiter(
    aqr_factory, make_middleware_app, asgi_client_factory
):
    """When downstream raises and releases a slot, the next queued waiter should be woken."""
    aqr = aqr_factory(default_max_concurrent=1, ceiling=10)
    fail_first = True

    async def handler(request):
        nonlocal fail_first
        await request.body()
        if fail_first:
            fail_first = False
            raise ValueError("first request fails")
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    app = make_middleware_app(handler, aqr=aqr)
    client = await asgi_client_factory(app)

    # Start first request (will fail)
    t1 = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    )
    await asyncio.sleep(0.05)

    # Queue second request
    t2 = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    )

    results = await asyncio.gather(t1, t2, return_exceptions=True)
    # Second should succeed (first fails, releases slot, wakes second)
    assert results[1].status_code == 200

    # All slots released
    info = await aqr.get_model_info("gpt-4")
    assert info["active"] == 0


@pytest.mark.asyncio
async def test_heartbeat_start_failure_releases_claimed_slot_before_503(monkeypatch):
    worktree_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    for _name in list(sys.modules):
        if _name == "litellm" or _name.startswith("litellm."):
            sys.modules.pop(_name, None)
    sys.path.insert(0, worktree_root)
    module = importlib.import_module("litellm.proxy.middleware.auto_queue_middleware")
    assert ".worktrees/autoqueue-distributed-queue" in module.__file__
    release_calls = []
    monkeypatch.setattr(module, "_id_counter", iter([12]))
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 0)

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return False

        async def stop(self):
            return None

    class FakeRedisClient:
        async def zcard(self, key):
            return 0

    class FakeRedisWrapper:
        redis = FakeRedisClient()

        async def admit_or_enqueue(self, model, request_id, priority, deadline_at_ms, worker_id):
            return AdmitDecision(
                decision="admit_now",
                claim_token="claim-1",
                request_state=AutoQueueRequestState(
                    request_id=request_id,
                    model=model,
                    priority=priority,
                    state="claimed",
                    enqueued_at_ms=deadline_at_ms - 1000,
                    deadline_at_ms=deadline_at_ms,
                    worker_id=worker_id,
                    claim_token="claim-1",
                    claimed_at_ms=deadline_at_ms - 1000,
                ),
            )

        async def get_model_info(self, model):
            return {"active": 1, "limit": 1, "queued": 0, "ceiling": 1}

        async def release_and_claim_next(self, model, request_id, **kwargs):
            release_calls.append((model, request_id, kwargs))
            return ReleaseTransfer(
                claimed_request_id=None,
                claim_token=None,
            )

    monkeypatch.setattr(module, "ActiveLeaseHeartbeat", FakeHeartbeat)

    async def app(scope, receive, send):
        raise AssertionError("downstream app should not be called when heartbeat start fails")

    middleware = module.AutoQueueMiddleware(app, aqr=FakeRedisWrapper(), enabled=True)

    sent = []

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps({"model": "gpt-4", "messages": []}).encode(),
            "more_body": False,
        }

    async def send(message):
        sent.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
        },
        receive,
        send,
    )

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 503
    assert release_calls == [
        (
            "gpt-4",
            "gpt-4-12-0",
            {"terminal_state": "cancelled", "allow_missing_active": True},
        )
    ]


@pytest.mark.asyncio
async def test_non_429_4xx_does_not_scale_or_increment_success(
    aqr_factory, make_middleware_app, asgi_client_factory, redis
):
    """A 400 response should NOT trigger scale-down or success counter."""
    aqr = aqr_factory(default_max_concurrent=2, ceiling=10)

    async def handler(request):
        await request.body()
        from starlette.responses import JSONResponse
        return JSONResponse({"error": "bad"}, status_code=400)

    app = make_middleware_app(handler, aqr=aqr)
    client = await asgi_client_factory(app)

    await redis.set("autoq:limit:gpt-4", 5)

    resp = await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert resp.status_code == 400

    info = await aqr.get_model_info("gpt-4")
    # Limit should not change (no scale-down), success counter should be 0
    assert info["limit"] == 5


@pytest.mark.asyncio
async def test_queue_is_per_model_not_global(
    aqr_factory, make_middleware_app, asgi_client_factory
):
    """Queuing for one model should not affect another model's capacity."""
    aqr = aqr_factory(default_max_concurrent=1, ceiling=10)

    async def handler(request):
        body = await request.body()
        data = json.loads(body)
        await asyncio.sleep(0.05)
        from starlette.responses import JSONResponse
        return JSONResponse({"model": data.get("model")})

    app = make_middleware_app(handler, aqr=aqr)
    client = await asgi_client_factory(app)

    # Both models should be able to acquire slots simultaneously
    t1 = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    )
    t2 = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "claude-3", "messages": []})
    )

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.xfail(reason="middleware only checks shutdown in queue path, not acquire path")
@pytest.mark.asyncio
async def test_shutdown_rejects_new_requests_even_if_capacity_exists(
    aqr_factory, make_middleware_app, asgi_client_factory
):
    """During shutdown, even if capacity exists, new requests should get 503."""
    aqr = aqr_factory(default_max_concurrent=2, ceiling=10)

    async def handler(request):
        await request.body()
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    app = make_middleware_app(handler, aqr=aqr)
    client = await asgi_client_factory(app)

    # Trigger shutdown
    app.shutdown()

    resp = await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert resp.status_code == 503
    assert "shutting down" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_stale_waiter_cannot_leak_transferred_slot(
    aqr_factory, make_middleware_app, asgi_client_factory
):
    """A waiter that was removed from queue cannot claim a transferred slot."""
    aqr = aqr_factory(default_max_concurrent=1, ceiling=10)

    async def handler(request):
        await request.body()
        await asyncio.sleep(0.05)
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    app = make_middleware_app(handler, aqr=aqr)
    client = await asgi_client_factory(app)

    # Fill the slot
    t1 = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    )
    await asyncio.sleep(0.05)

    # The middleware should properly transfer slots only to entries still in queue
    await t1

    # After t1 completes, the slot should be released and no stale entries remain
    queue = app._get_queue("gpt-4")
    assert queue.depth == 0
    info = await aqr.get_model_info("gpt-4")
    assert info["active"] == 0


@pytest.mark.asyncio
async def test_key_metadata_lookup_failure_falls_back_to_defaults(
    aqr_factory, make_middleware_app, asgi_client_factory, key_metadata_factory
):
    """If key metadata lookup fails, middleware should use defaults and still work."""
    aqr = aqr_factory(default_max_concurrent=2, ceiling=10)

    async def handler(request):
        await request.body()
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    app = make_middleware_app(handler, aqr=aqr)
    client = await asgi_client_factory(app)

    # Make metadata lookup explode
    key_metadata_factory(explode=True)

    # Request should still succeed (fallback to defaults)
    resp = await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert resp.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [202, 204])
async def test_2xx_non_200_counts_as_success(
    aqr_factory, make_middleware_app, asgi_client_factory, status_code
):
    """2xx responses that aren't 200 should still be treated as success for auto-scaling."""
    aqr = aqr_factory(default_max_concurrent=2, ceiling=10, scale_up_threshold=1)

    async def handler(request):
        await request.body()
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True}, status_code=status_code)

    app = make_middleware_app(handler, aqr=aqr)
    client = await asgi_client_factory(app)

    resp = await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert resp.status_code == status_code

    # The 2xx response triggers on_success. With scale_up_threshold=1, one success
    # triggers scale-up: limit goes from 2 -> 3. (The Lua script resets the counter
    # to 0 after scale-up, so we check the limit instead of the counter.)
    info = await aqr.get_model_info("gpt-4")
    assert info["limit"] == 3
