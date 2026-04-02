import asyncio

import fakeredis.aioredis
import pytest
import pytest_asyncio

from litellm.proxy.middleware.auto_queue_middleware import AutoQueueRedis


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
        return AutoQueueRedis(
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
