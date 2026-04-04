import asyncio
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
for _name in list(sys.modules):
    if _name == "litellm" or _name.startswith("litellm."):
        sys.modules.pop(_name, None)

import fakeredis.aioredis
import pytest
import pytest_asyncio

import litellm.proxy.middleware.auto_queue_scripts as auto_queue_scripts
from litellm.proxy.middleware.auto_queue_scripts import DistributedAutoQueueRedis
from litellm.proxy.middleware.auto_queue_state import (
    AutoQueueRequestState,
    active_key,
    active_lease_key,
    claim_key,
    request_state_from_hash,
    queue_key,
    queue_score,
    request_key,
)


# Local overrides to avoid session-vs-function event-loop mismatch in the
# shared conftest redis/aqr_factory fixtures.
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
        return DistributedAutoQueueRedis(
            redis=redis,
            default_max_concurrent=default_max_concurrent,
            ceiling=ceiling,
            scale_up_threshold=scale_up_threshold,
            scale_down_step=scale_down_step,
        )

    return _factory


pytestmark = pytest.mark.asyncio


async def _redis_int(redis, key: str) -> int:
    raw = await redis.get(key)
    return int(raw or 0)


async def test_try_acquire_is_atomic_under_parallel_contention(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "gpt-4o"
    start = asyncio.Event()

    async def contender():
        await start.wait()
        return await aqr.try_acquire(model)

    tasks = [asyncio.create_task(contender()) for _ in range(16)]
    start.set()

    results = await asyncio.gather(*tasks)

    assert sum(results) == 1
    assert await _redis_int(redis, f"autoq:active:{model}") == 1


async def test_try_acquire_is_isolated_per_model(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    assert await aqr.try_acquire("gpt-4o") is True
    assert await aqr.try_acquire("claude-3") is True
    assert await _redis_int(redis, "autoq:active:gpt-4o") == 1
    assert await _redis_int(redis, "autoq:active:claude-3") == 1


async def test_scale_up_resets_counter_after_threshold_window(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=2, scale_up_threshold=3)
    model = "gpt-4o"

    for _ in range(6):
        await aqr.on_success(model)

    assert await _redis_int(redis, f"autoq:limit:{model}") == 4
    assert await _redis_int(redis, f"autoq:success:{model}") == 0


async def test_scale_down_without_existing_limit_uses_default_limit(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=5, scale_down_step=2)
    model = "gpt-4o"

    await aqr.on_429(model)

    assert await _redis_int(redis, f"autoq:limit:{model}") == 3


async def test_release_never_goes_negative_under_parallel_release(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "gpt-4o"
    active_key = f"autoq:active:{model}"
    await redis.set(active_key, 2)

    start = asyncio.Event()

    async def releaser():
        await start.wait()
        return await aqr.release(model)

    tasks = [asyncio.create_task(releaser()) for _ in range(5)]
    start.set()

    results = await asyncio.gather(*tasks)

    assert all(r >= 0 for r in results)
    assert await _redis_int(redis, active_key) == 0


async def test_request_state_round_trip_preserves_fields():
    state = AutoQueueRequestState(
        request_id="req-1",
        model="glm-5.1",
        priority=10,
        state="queued",
        enqueued_at_ms=1_700_000_000_000,
        deadline_at_ms=1_700_000_000_500,
        worker_id="worker-a",
        claim_token=None,
    )

    restored = AutoQueueRequestState.from_json(state.to_json())

    await asyncio.sleep(0)
    assert restored == state


async def test_admit_or_enqueue_places_second_request_in_redis_queue(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)

    first = await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-1",
        priority=10,
        deadline_at_ms=1_700_000_000_000,
        worker_id="worker-a",
    )
    second = await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-2",
        priority=10,
        deadline_at_ms=1_700_000_000_500,
        worker_id="worker-b",
    )

    assert first.decision == "admit_now"
    assert first.request_state is not None
    assert first.request_state.state == "active"
    assert second.decision == "queued"
    assert second.request_state is not None
    assert second.request_state.state == "queued"
    assert await redis.zrange(queue_key("glm-5.1"), 0, -1) == [b"req-2"]
    assert request_state_from_hash(await redis.hgetall(request_key("req-2"))).state == "queued"
    assert request_state_from_hash(await redis.hgetall(request_key("req-1"))).state == "active"


async def test_admit_or_enqueue_respects_deterministic_tie_order(aqr_factory, redis, monkeypatch):
    aqr = aqr_factory(default_max_concurrent=0)
    now_values = iter([1_700_000_000_000, 1_700_000_000_001])
    monkeypatch.setattr(auto_queue_scripts, "current_time_ms", lambda: next(now_values))

    await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-b",
        priority=10,
        deadline_at_ms=1_700_000_000_000,
        worker_id="worker-b",
    )
    await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-a",
        priority=10,
        deadline_at_ms=1_700_000_000_001,
        worker_id="worker-a",
    )

    assert queue_score(10, 1_700_000_000_000) < queue_score(10, 1_700_000_000_001)
    assert await redis.zrange(queue_key("glm-5.1"), 0, -1) == [b"req-b", b"req-a"]


async def test_admit_or_enqueue_is_atomic_under_parallel_contention(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "glm-5.1"
    start = asyncio.Event()

    async def contender(i: int):
        await start.wait()
        return await aqr.admit_or_enqueue(
            model=model,
            request_id=f"req-{i}",
            priority=10,
            deadline_at_ms=1_700_000_000_000 + i,
            worker_id=f"worker-{i}",
        )

    tasks = [asyncio.create_task(contender(i)) for i in range(12)]
    start.set()
    results = await asyncio.gather(*tasks)

    assert sum(1 for result in results if result.decision == "admit_now") == 1
    assert await redis.zcard(queue_key(model)) == 11


async def test_release_transfers_claim_to_head_of_queue(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)

    await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-1",
        priority=10,
        deadline_at_ms=1_700_000_000_000,
        worker_id="worker-a",
    )
    await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-2",
        priority=10,
        deadline_at_ms=1_700_000_000_500,
        worker_id="worker-b",
    )

    transfer = await aqr.release_and_claim_next("glm-5.1", "req-1")

    assert transfer.claimed_request_id == "req-2"
    assert transfer.claim_token is not None
    assert await redis.zrange(queue_key("glm-5.1"), 0, -1) == []
    assert await redis.get(claim_key("req-2")) is not None
    assert await redis.get(active_key("glm-5.1")) == b"1"
    assert request_state_from_hash(await redis.hgetall(request_key("req-2"))).state == "claimed"


async def test_release_and_claim_next_refuses_when_active_is_zero(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "glm-5.1"

    await redis.set(active_key(model), 0)
    await redis.zadd(queue_key(model), {"req-queued": 1})

    transfer = await aqr.release_and_claim_next(model, "req-1")

    assert transfer.claimed_request_id is None
    assert transfer.claim_token is None
    assert await redis.zrange(queue_key(model), 0, -1) == [b"req-queued"]
    assert await redis.get(active_key(model)) == b"0"


async def test_release_and_claim_next_skips_stale_queue_members(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "glm-5.1"

    await aqr.admit_or_enqueue(
        model=model,
        request_id="req-1",
        priority=10,
        deadline_at_ms=1_700_000_000_000,
        worker_id="worker-a",
    )
    await aqr.admit_or_enqueue(
        model=model,
        request_id="req-2",
        priority=10,
        deadline_at_ms=1_700_000_000_500,
        worker_id="worker-b",
    )
    await redis.zadd(queue_key(model), {"req-stale": 0})

    transfer = await aqr.release_and_claim_next(model, "req-1")

    assert transfer.claimed_request_id == "req-2"
    assert await redis.zrange(queue_key(model), 0, -1) == []
    assert await redis.hgetall(request_key("req-stale")) == {}
    assert request_state_from_hash(await redis.hgetall(request_key("req-2"))).state == "claimed"


async def test_admit_or_enqueue_stores_active_lease_safely_for_quoted_worker_ids(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    worker_id = 'worker-"quoted\\\\slash"'

    result = await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-1",
        priority=10,
        deadline_at_ms=1_700_000_000_000,
        worker_id=worker_id,
    )

    assert result.decision == "admit_now"
    lease = await redis.hgetall(active_lease_key("req-1"))
    assert lease[b"worker_id"].decode() == worker_id
    assert lease[b"claim_token"].decode() == result.claim_token


async def test_activate_claim_promotes_claimed_request_to_active(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)

    await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-1",
        priority=10,
        deadline_at_ms=1_700_000_000_000,
        worker_id="worker-a",
    )
    await aqr.admit_or_enqueue(
        model="glm-5.1",
        request_id="req-2",
        priority=10,
        deadline_at_ms=1_700_000_000_500,
        worker_id="worker-b",
    )

    transfer = await aqr.release_and_claim_next("glm-5.1", "req-1")

    assert transfer.claimed_request_id == "req-2"
    assert transfer.claim_token is not None
    activated = await aqr.activate_claim("glm-5.1", "req-2", transfer.claim_token)

    assert activated is True
    assert await redis.get(active_key("glm-5.1")) == b"1"
    assert request_state_from_hash(await redis.hgetall(request_key("req-2"))).state == "active"


async def test_middleware_preserves_autoq_configuration(monkeypatch):
    module = importlib.import_module("litellm.proxy.middleware.auto_queue_middleware")

    captured = {}

    class FakeRedisWrapper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "DistributedAutoQueueRedis", FakeRedisWrapper)
    monkeypatch.setattr(module.aioredis, "Redis", lambda **kwargs: object())
    monkeypatch.setattr(module, "DEFAULT_MAX_CONCURRENT", 7)
    monkeypatch.setattr(module, "CEILING", 11)
    monkeypatch.setattr(module, "SCALE_UP_THRESHOLD", 13)
    monkeypatch.setattr(module, "SCALE_DOWN_STEP", 3)

    async def app(scope, receive, send):
        return None

    middleware = module.AutoQueueMiddleware(app, enabled=False)
    middleware._aqr = None
    middleware._ensure_aqr()

    await asyncio.sleep(0)
    assert captured["default_max_concurrent"] == 7
    assert captured["ceiling"] == 11
    assert captured["scale_up_threshold"] == 13
    assert captured["scale_down_step"] == 3
