"""tests/test_litellm/proxy/middleware/test_auto_queue_middleware.py"""
import asyncio
import json
import os

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.auto_queue_middleware import (
    AutoQueueMiddleware,
    AutoQueueRedis,
    ModelQueue,
)


# -- Fixtures -----------------------------------------------------------------


@pytest_asyncio.fixture
async def redis():
    r = fakeredis.aioredis.FakeRedis(db=3)
    yield r
    await r.flushdb()
    await r.aclose()


@pytest.fixture
def aqr(redis):
    return AutoQueueRedis(
        redis=redis,
        default_max_concurrent=5,
        ceiling=20,
        scale_up_threshold=3,
        scale_down_step=1,
    )


# -- Slot acquire / release ---------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_slot_succeeds_when_under_limit(aqr):
    assert await aqr.try_acquire("gpt-4") is True


@pytest.mark.asyncio
async def test_acquire_slot_fails_when_at_limit(aqr):
    for _ in range(5):
        await aqr.try_acquire("gpt-4")
    assert await aqr.try_acquire("gpt-4") is False


@pytest.mark.asyncio
async def test_release_slot_frees_capacity(aqr):
    for _ in range(5):
        await aqr.try_acquire("gpt-4")
    await aqr.release("gpt-4")
    assert await aqr.try_acquire("gpt-4") is True


@pytest.mark.asyncio
async def test_release_never_goes_negative(aqr):
    await aqr.release("gpt-4")
    await aqr.release("gpt-4")
    info = await aqr.get_model_info("gpt-4")
    assert info["active"] == 0


@pytest.mark.asyncio
async def test_get_model_info_defaults(aqr):
    info = await aqr.get_model_info("gpt-4")
    assert info == {"active": 0, "limit": 5, "queued": 0, "ceiling": 20}


# -- release_and_transfer ------------------------------------------------------


@pytest.mark.asyncio
async def test_release_and_transfer_with_waiters(aqr):
    """When waiters exist, slot is transferred (active count unchanged)."""
    await aqr.try_acquire("gpt-4")
    transferred = await aqr.release_and_transfer("gpt-4", has_waiters=True)
    assert transferred is True
    info = await aqr.get_model_info("gpt-4")
    assert info["active"] == 1  # stayed the same -- transferred, not released


@pytest.mark.asyncio
async def test_release_and_transfer_without_waiters(aqr):
    """Without waiters, slot is released normally."""
    await aqr.try_acquire("gpt-4")
    transferred = await aqr.release_and_transfer("gpt-4", has_waiters=False)
    assert transferred is False
    info = await aqr.get_model_info("gpt-4")
    assert info["active"] == 0


@pytest.mark.asyncio
async def test_release_and_transfer_no_active_slots(aqr):
    """When no active slots, nothing happens."""
    transferred = await aqr.release_and_transfer("gpt-4", has_waiters=True)
    assert transferred is False


# -- Auto-scale ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_scale_up_after_threshold_successes(aqr):
    for _ in range(3):  # threshold = 3
        await aqr.on_success("gpt-4")
    info = await aqr.get_model_info("gpt-4")
    assert info["limit"] == 6  # 5 + 1


@pytest.mark.asyncio
async def test_scale_up_capped_at_ceiling(aqr):
    # Set limit just below ceiling
    await aqr.redis.set("autoq:limit:gpt-4", 20)
    for _ in range(3):
        await aqr.on_success("gpt-4")
    info = await aqr.get_model_info("gpt-4")
    assert info["limit"] == 20  # stays at ceiling


@pytest.mark.asyncio
async def test_scale_down_on_429(aqr):
    # First set a known limit
    await aqr.redis.set("autoq:limit:gpt-4", 10)
    await aqr.on_429("gpt-4")
    info = await aqr.get_model_info("gpt-4")
    assert info["limit"] == 9


@pytest.mark.asyncio
async def test_scale_down_never_below_1(aqr):
    await aqr.redis.set("autoq:limit:gpt-4", 1)
    await aqr.on_429("gpt-4")
    info = await aqr.get_model_info("gpt-4")
    assert info["limit"] == 1


@pytest.mark.asyncio
async def test_scale_down_resets_success_counter(aqr):
    await aqr.on_success("gpt-4")
    await aqr.on_success("gpt-4")
    await aqr.on_429("gpt-4")
    # Next 3 successes should trigger scale up (counter was reset)
    for _ in range(3):
        await aqr.on_success("gpt-4")
    # Limit was default 5, 429 brought to 4, then 3 successes -> 5
    info = await aqr.get_model_info("gpt-4")
    assert info["limit"] == 5


# -- In-memory priority queue --------------------------------------------------


@pytest.fixture
def queue():
    return ModelQueue(max_depth=3)


def test_queue_add_and_wake(queue):
    event = asyncio.Event()
    queue.add("req-1", event, priority=10)
    assert queue.depth == 1
    woken = queue.wake_next()
    assert woken is True
    assert event.is_set()
    assert queue.depth == 0


def test_queue_priority_ordering(queue):
    low = asyncio.Event()
    high = asyncio.Event()
    queue.add("req-low", low, priority=10)
    queue.add("req-high", high, priority=1)
    queue.wake_next()
    assert high.is_set()  # higher priority (lower number) woken first
    assert not low.is_set()


def test_queue_fifo_within_same_priority(queue):
    first = asyncio.Event()
    second = asyncio.Event()
    queue.add("req-1", first, priority=10)
    queue.add("req-2", second, priority=10)
    queue.wake_next()
    assert first.is_set()
    assert not second.is_set()


def test_queue_max_depth_exceeded(queue):
    for i in range(3):
        queue.add(f"req-{i}", asyncio.Event(), priority=10)
    assert queue.is_full is True


def test_queue_remove_by_id(queue):
    event = asyncio.Event()
    queue.add("req-1", event, priority=10)
    queue.remove("req-1")
    assert queue.depth == 0
    assert queue.wake_next() is False


def test_queue_wake_all_sets_all_events(queue):
    events = [asyncio.Event() for _ in range(3)]
    for i, e in enumerate(events):
        queue.add(f"req-{i}", e, priority=10)
    queue.wake_all()
    assert all(e.is_set() for e in events)
    assert queue.depth == 0


# -- Middleware helpers --------------------------------------------------------


@pytest_asyncio.fixture
async def aqr_for_middleware(redis):
    return AutoQueueRedis(
        redis=redis,
        default_max_concurrent=2,
        ceiling=10,
        scale_up_threshold=3,
        scale_down_step=1,
    )


def _make_echo_app(aqr_instance):
    """App that echoes back the request body, proving body replay works."""
    async def echo(request: Request):
        body = await request.body()
        data = json.loads(body) if body else {}
        return JSONResponse({"model": data.get("model", "none"), "echoed": True})

    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[
        Route("/v1/chat/completions", echo, methods=["POST"]),
        Route("/health", health),
    ])
    app.add_middleware(AutoQueueMiddleware, aqr=aqr_instance, enabled=True)
    return app


# -- Passthrough ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_llm_routes_pass_through(aqr_for_middleware):
    client = TestClient(_make_echo_app(aqr_for_middleware))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_body_is_replayed_to_downstream(aqr_for_middleware):
    client = TestClient(_make_echo_app(aqr_for_middleware))
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert resp.status_code == 200
    assert resp.json()["model"] == "gpt-4"
    assert resp.json()["echoed"] is True


@pytest.mark.asyncio
async def test_non_json_body_passes_through(aqr_for_middleware):
    """Non-JSON request bodies should pass through without queuing."""
    client = TestClient(_make_echo_app(aqr_for_middleware), raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        content=b"not json at all",
        headers={"content-type": "text/plain"},
    )
    # The middleware passes non-JSON through without queuing.
    # The downstream echo handler fails to parse -> 500, but the point is
    # that the middleware did not block or queue it.
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_disabled_middleware_passes_through(redis):
    """When disabled, the middleware is a transparent passthrough."""
    aqr = AutoQueueRedis(redis=redis, default_max_concurrent=1)

    async def handler(request: Request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/chat/completions", handler, methods=["POST"])])
    app.add_middleware(AutoQueueMiddleware, aqr=aqr, enabled=False)
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert resp.status_code == 200
    # Should not have touched Redis
    active = int(await redis.get("autoq:active:gpt-4") or 0)
    assert active == 0


# -- Full flow -----------------------------------------------------------------


def _make_slow_app(aqr_instance, delay: float = 0.1, status: int = 200):
    """App that takes `delay` seconds and returns `status`."""
    async def handler(request: Request):
        body = await request.body()
        await asyncio.sleep(delay)
        return JSONResponse({"ok": True}, status_code=status)

    app = Starlette(routes=[
        Route("/v1/chat/completions", handler, methods=["POST"]),
    ])
    app.add_middleware(AutoQueueMiddleware, aqr=aqr_instance, enabled=True)
    return app


@pytest.mark.asyncio
async def test_request_acquires_and_releases_slot(redis, aqr_for_middleware):
    app = _make_slow_app(aqr_for_middleware, delay=0.0)
    client = TestClient(app)
    client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    # After request completes, slot should be released
    active = int(await redis.get("autoq:active:gpt-4") or 0)
    assert active == 0


@pytest.mark.asyncio
async def test_429_response_triggers_scale_down(redis, aqr_for_middleware):
    app = _make_slow_app(aqr_for_middleware, delay=0.0, status=429)
    client = TestClient(app)
    await redis.set("autoq:limit:gpt-4", 10)
    client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    limit = int(await redis.get("autoq:limit:gpt-4") or 0)
    assert limit == 9


@pytest.mark.asyncio
async def test_success_response_increments_success_counter(redis, aqr_for_middleware):
    app = _make_slow_app(aqr_for_middleware, delay=0.0, status=200)
    client = TestClient(app)
    client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    count = int(await redis.get("autoq:success:gpt-4") or 0)
    assert count == 1


@pytest.mark.asyncio
async def test_queue_full_returns_503(redis):
    """When max_concurrent=1 and max_queue_depth=1, third request gets 503."""
    aqr = AutoQueueRedis(
        redis=redis, default_max_concurrent=1, ceiling=10,
        scale_up_threshold=3, scale_down_step=1,
    )
    hold_event = asyncio.Event()

    async def slow_handler(request: Request):
        body = await request.body()
        await hold_event.wait()
        return JSONResponse({"ok": True})

    app = Starlette(routes=[
        Route("/v1/chat/completions", slow_handler, methods=["POST"]),
    ])
    mw = AutoQueueMiddleware(app, aqr=aqr, max_queue_depth=1, enabled=True)

    async with AsyncClient(transport=ASGITransport(app=mw), base_url="http://test") as client:
        # Fill the slot
        task1 = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
        )
        await asyncio.sleep(0.05)

        # Fill the queue (1 deep)
        task2 = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
        )
        await asyncio.sleep(0.05)

        # This should get 503 -- queue full
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
        assert resp.status_code == 503

        hold_event.set()
        await task1
        await task2


# -- Disconnect and shutdown ---------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_rejects_new_requests(redis, aqr_for_middleware):
    inner_app = Starlette(routes=[
        Route("/v1/chat/completions", lambda r: JSONResponse({"ok": True}), methods=["POST"]),
    ])
    mw = AutoQueueMiddleware(inner_app, aqr=aqr_for_middleware, enabled=True)
    mw._shutting_down = True

    async with AsyncClient(transport=ASGITransport(app=mw), base_url="http://test") as client:
        # Fill the only slots so the next request tries to queue
        await redis.set("autoq:active:gpt-4", 2)  # at limit
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
        assert resp.status_code == 503
        assert "shutting down" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_shutdown_wakes_all_queued(redis, aqr_for_middleware):
    inner_app = Starlette(routes=[
        Route("/v1/chat/completions", lambda r: JSONResponse({"ok": True}), methods=["POST"]),
    ])
    mw = AutoQueueMiddleware(inner_app, aqr=aqr_for_middleware, enabled=True)
    q = mw._get_queue("gpt-4")
    event = asyncio.Event()
    q.add("req-1", event, priority=10)
    mw.shutdown()
    assert event.is_set()
    assert mw._shutting_down is True


# -- Integration: concurrent requests -----------------------------------------


@pytest.mark.asyncio
async def test_concurrent_requests_queued_and_served(redis):
    """3 concurrent requests with max_concurrent=1: served sequentially."""
    aqr = AutoQueueRedis(redis=redis, default_max_concurrent=1, ceiling=5,
                          scale_up_threshold=100, scale_down_step=1)

    call_order = []

    async def ordered_handler(request: Request):
        body = await request.body()
        data = json.loads(body)
        call_order.append(data["id"])
        await asyncio.sleep(0.05)
        return JSONResponse({"id": data["id"]})

    inner = Starlette(routes=[
        Route("/v1/chat/completions", ordered_handler, methods=["POST"]),
    ])
    mw = AutoQueueMiddleware(inner, aqr=aqr, enabled=True)

    async with AsyncClient(transport=ASGITransport(app=mw), base_url="http://test") as client:
        tasks = [
            asyncio.create_task(
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4", "messages": [], "id": i})
            )
            for i in range(3)
        ]
        responses = await asyncio.gather(*tasks)

    assert all(r.status_code == 200 for r in responses)
    # All 3 should have been served (order may vary due to async scheduling)
    assert sorted(call_order) == [0, 1, 2]

    # Verify all slots released
    active = int(await redis.get("autoq:active:gpt-4") or 0)
    assert active == 0


@pytest.mark.asyncio
async def test_queue_status_reflects_active_requests(redis):
    aqr = AutoQueueRedis(redis=redis, default_max_concurrent=1, ceiling=5,
                          scale_up_threshold=100, scale_down_step=1)
    hold = asyncio.Event()

    async def blocking_handler(request: Request):
        body = await request.body()
        await hold.wait()
        return JSONResponse({"ok": True})

    inner = Starlette(routes=[
        Route("/v1/chat/completions", blocking_handler, methods=["POST"]),
    ])
    mw = AutoQueueMiddleware(inner, aqr=aqr, enabled=True)

    async with AsyncClient(transport=ASGITransport(app=mw), base_url="http://test") as client:
        # Start a request that blocks
        task = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
        )
        await asyncio.sleep(0.05)

        # Verify active count via Redis directly (status endpoint removed from middleware)
        active = int(await redis.get("autoq:active:gpt-4") or 0)
        assert active == 1

        hold.set()
        await task


# -- Slot transfer race condition regression -----------------------------------


@pytest.mark.asyncio
async def test_slot_transfer_prevents_stealing(redis):
    """Verify that release_and_transfer atomically holds the slot for the waiter."""
    aqr = AutoQueueRedis(redis=redis, default_max_concurrent=1, ceiling=5,
                          scale_up_threshold=100, scale_down_step=1)

    # Acquire a slot
    assert await aqr.try_acquire("gpt-4") is True

    # Transfer to a waiter
    transferred = await aqr.release_and_transfer("gpt-4", has_waiters=True)
    assert transferred is True

    # Active count should still be 1 (transferred, not released)
    info = await aqr.get_model_info("gpt-4")
    assert info["active"] == 1

    # A new arrival should NOT be able to steal the slot
    stolen = await aqr.try_acquire("gpt-4")
    assert stolen is False  # slot is held for the waiter
