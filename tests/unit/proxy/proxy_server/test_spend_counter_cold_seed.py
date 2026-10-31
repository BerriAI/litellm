"""Overlapping cold starts of one spend counter inside one process add that process's cached spend once."""

import asyncio
import threading
from typing import Final, cast

import pytest

import litellm.proxy.proxy_server as ps
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.spend_tracking.spend_counter_batch import active_spend_counter_batch, spend_counter_batch_scope

SOURCE: Final = "end_user_id:customer-1"


class _SharedRedis:
    """The counter commands a cold start sends, yielding like a network call so overlapping requests interleave."""

    def __init__(self) -> None:
        self.values: Final[dict[str, float]] = {}  # mutable-ok: stands in for the Redis keyspace
        self.failing = False

    async def async_batch_get_cache(self, key_list: list[str], **_: object) -> dict[str, float | None]:
        await asyncio.sleep(0)
        return {key: self.values.get(key) for key in key_list}

    async def async_get_cache(self, key: str, **_: object) -> float | None:
        await asyncio.sleep(0)
        return self.values.get(key)

    async def async_increment(self, key: str, value: float, **_: object) -> float:
        await asyncio.sleep(0)
        if self.failing:
            raise ConnectionError("redis down")
        self.values[key] = self.values.get(key, 0.0) + value
        return self.values[key]

    async def async_delete_cache(self, key: str) -> None:
        await asyncio.sleep(0)
        if not self.failing:
            self.values.pop(key, None)


def _process(monkeypatch: pytest.MonkeyPatch, redis: _SharedRedis | None, cached_spend: float) -> DualCache:
    counters: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=cast(RedisCache, redis))
    objects: Final = DualCache(in_memory_cache=InMemoryCache())
    objects.in_memory_cache.set_cache(key=SOURCE, value={"spend": cached_spend})
    monkeypatch.setattr(ps, "spend_counter_cache", counters)
    monkeypatch.setattr(ps, "user_api_key_cache", objects)
    monkeypatch.setattr(ps, "prisma_client", object())
    return counters


async def _cold_start(redis: _SharedRedis | None, counter: str) -> None:
    with spend_counter_batch_scope(cast(RedisCache, redis) if redis is not None else None, frozenset((counter,))):
        await ps.ensure_spend_counter_initialized(counter_key=counter, source_cache_key=SOURCE)


async def _read_cold(counter: str) -> None:
    batch: Final = active_spend_counter_batch()
    assert batch is not None
    assert await batch.read(counter) == (None, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_redis", (True, False))
async def test_overlapping_cold_starts_in_one_process_add_its_cached_spend_once(
    monkeypatch: pytest.MonkeyPatch, with_redis: bool
) -> None:
    counter: Final = f"spend:end_user:once-{with_redis}"
    redis: Final = _SharedRedis() if with_redis else None
    counters: Final = _process(monkeypatch, redis, cached_spend=6.0)

    await asyncio.gather(*(_cold_start(redis, counter) for _ in range(5)))

    seeded: Final = redis.values.get(counter) if redis is not None else counters.in_memory_cache.get_cache(key=counter)
    assert seeded == 6.0


@pytest.mark.asyncio
async def test_a_request_that_read_the_counter_before_this_process_seeded_it_does_not_add_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = "spend:end_user:stale-reader"
    redis: Final = _SharedRedis()
    _process(monkeypatch, redis, cached_spend=6.0)
    read: Final = asyncio.Event()

    async def _stale_request() -> None:
        with spend_counter_batch_scope(cast(RedisCache, redis), frozenset((counter,))):
            await _read_cold(counter)
            read.set()
            await seeded.wait()
            await ps.ensure_spend_counter_initialized(counter_key=counter, source_cache_key=SOURCE)

    seeded: Final = asyncio.Event()

    async def _first_request() -> None:
        await read.wait()
        await _cold_start(redis, counter)
        seeded.set()

    await asyncio.gather(_stale_request(), _first_request())

    assert redis.values[counter] == 6.0


@pytest.mark.asyncio
async def test_a_counter_lost_again_after_this_process_seeded_it_is_seeded_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = "spend:end_user:lost-again"
    redis: Final = _SharedRedis()
    _process(monkeypatch, redis, cached_spend=6.0)
    read: Final = asyncio.Event()
    seeded: Final = asyncio.Event()

    async def _stale_request() -> None:
        with spend_counter_batch_scope(cast(RedisCache, redis), frozenset((counter,))):
            await _read_cold(counter)
            read.set()
            await seeded.wait()
            del redis.values[counter]
            await ps.ensure_spend_counter_initialized(counter_key=counter, source_cache_key=SOURCE)

    async def _first_request() -> None:
        await read.wait()
        await _cold_start(redis, counter)
        seeded.set()

    await asyncio.gather(_stale_request(), _first_request())

    assert redis.values[counter] == 6.0


@pytest.mark.asyncio
async def test_another_process_seeding_after_this_request_read_the_counter_still_sums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = "spend:end_user:two-processes"
    redis: Final = _SharedRedis()
    read: Final = asyncio.Event()
    a_seeded: Final = asyncio.Event()

    async def _process_b() -> None:
        with spend_counter_batch_scope(cast(RedisCache, redis), frozenset((counter,))):
            await _read_cold(counter)
            read.set()
            await a_seeded.wait()
            _process(monkeypatch, redis, cached_spend=5.0)
            await ps.ensure_spend_counter_initialized(counter_key=counter, source_cache_key=SOURCE)

    async def _process_a() -> None:
        await read.wait()
        _process(monkeypatch, redis, cached_spend=3.0)
        await _cold_start(redis, counter)
        a_seeded.set()

    await asyncio.gather(_process_b(), _process_a())

    assert redis.values[counter] == 8.0


@pytest.mark.asyncio
async def test_a_failed_seed_fails_the_requests_that_joined_it_and_the_next_cold_start_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = "spend:end_user:failed-seed"
    redis: Final = _SharedRedis()
    _process(monkeypatch, redis, cached_spend=6.0)
    redis.failing = True

    outcomes: Final = await asyncio.gather(*(_cold_start(redis, counter) for _ in range(3)), return_exceptions=True)
    redis.failing = False
    await _cold_start(redis, counter)

    assert [type(outcome) for outcome in outcomes] == [ConnectionError] * 3
    assert redis.values[counter] == 6.0


class _HeldReplyRedis(_SharedRedis):
    """Applies an increment at once but holds its reply until released, like a slow network round trip."""

    def __init__(self) -> None:
        super().__init__()
        self.applied: Final = asyncio.Event()
        self.release: Final = asyncio.Event()

    async def async_increment(self, key: str, value: float, **_: object) -> float:
        self.values[key] = self.values.get(key, 0.0) + value
        total: Final = self.values[key]
        self.applied.set()
        await self.release.wait()
        return total


@pytest.mark.asyncio
async def test_a_counter_deleted_before_an_earlier_add_returns_is_seeded_by_a_later_cold_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = "spend:end_user:deleted-mid-add"
    redis: Final = _HeldReplyRedis()
    _process(monkeypatch, redis, cached_spend=6.0)

    first: Final = asyncio.ensure_future(_cold_start(redis, counter))
    await redis.applied.wait()
    del redis.values[counter]
    later: Final = asyncio.ensure_future(_cold_start(redis, counter))
    for _ in range(20):
        await asyncio.sleep(0)
    redis.release.set()
    await asyncio.gather(first, later)

    assert redis.values[counter] == 6.0


def test_a_cold_start_on_another_event_loop_adds_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    counter: Final = "spend:end_user:two-loops"
    held: Final = threading.Event()
    release: Final = threading.Event()

    class _ThreadHeldRedis(_SharedRedis):
        async def async_increment(self, key: str, value: float, **_: object) -> float:
            if threading.current_thread() is not threading.main_thread():
                held.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
            return await super().async_increment(key, value)

    redis: Final = _ThreadHeldRedis()
    _process(monkeypatch, redis, cached_spend=6.0)
    other_loop: Final = threading.Thread(target=lambda: asyncio.run(_cold_start(redis, counter)), daemon=True)
    other_loop.start()
    assert held.wait(5)

    try:
        asyncio.run(_cold_start(redis, counter))
    finally:
        release.set()
        other_loop.join(5)

    assert redis.values[counter] == 12.0
