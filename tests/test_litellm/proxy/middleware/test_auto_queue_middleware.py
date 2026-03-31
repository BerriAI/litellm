"""tests/test_litellm/proxy/middleware/test_auto_queue_middleware.py"""
import asyncio
import os

import fakeredis.aioredis
import pytest
import pytest_asyncio

from litellm.proxy.middleware.auto_queue_middleware import AutoQueueRedis


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


from litellm.proxy.middleware.auto_queue_middleware import ModelQueue


# -- In-memory priority queue --------------------------------------------------


@pytest.fixture
def queue():
    return ModelQueue(max_depth=3)


@pytest.mark.asyncio
async def test_queue_add_and_wake(queue):
    event = asyncio.Event()
    queue.add("req-1", event, priority=10)
    assert queue.depth == 1
    woken = queue.wake_next()
    assert woken is True
    assert event.is_set()
    assert queue.depth == 0


@pytest.mark.asyncio
async def test_queue_priority_ordering(queue):
    low = asyncio.Event()
    high = asyncio.Event()
    queue.add("req-low", low, priority=10)
    queue.add("req-high", high, priority=1)
    queue.wake_next()
    assert high.is_set()  # higher priority (lower number) woken first
    assert not low.is_set()


@pytest.mark.asyncio
async def test_queue_fifo_within_same_priority(queue):
    first = asyncio.Event()
    second = asyncio.Event()
    queue.add("req-1", first, priority=10)
    queue.add("req-2", second, priority=10)
    queue.wake_next()
    assert first.is_set()
    assert not second.is_set()


@pytest.mark.asyncio
async def test_queue_max_depth_exceeded(queue):
    for i in range(3):
        queue.add(f"req-{i}", asyncio.Event(), priority=10)
    assert queue.is_full is True


@pytest.mark.asyncio
async def test_queue_remove_by_id(queue):
    event = asyncio.Event()
    queue.add("req-1", event, priority=10)
    queue.remove("req-1")
    assert queue.depth == 0
    assert queue.wake_next() is False


@pytest.mark.asyncio
async def test_queue_wake_all_sets_all_events(queue):
    events = [asyncio.Event() for _ in range(3)]
    for i, e in enumerate(events):
        queue.add(f"req-{i}", e, priority=10)
    queue.wake_all()
    assert all(e.is_set() for e in events)
    assert queue.depth == 0


import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.auto_queue_middleware import AutoQueueMiddleware


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
    app.add_middleware(AutoQueueMiddleware, aqr=aqr_instance)
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
async def test_queue_status_endpoint(aqr_for_middleware):
    client = TestClient(_make_echo_app(aqr_for_middleware))
    resp = client.get("/queue/status")
    assert resp.status_code == 200
    assert "models" in resp.json()
