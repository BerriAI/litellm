from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from redis import Redis

from litellm.caching.redis_batch import RedisBatch, active_post_call_redis_batch, request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.types.caching import (
    CacheFailure,
    CachePublish,
    CacheQueueUpdate,
    CapacityExceeded,
    ConcurrencyRead,
    ConcurrencyRelease,
    ConcurrencySlot,
    CounterIncrement,
    CounterUpdate,
    GuardedIncrement,
    LimitedIncrement,
    ThrottleKeys,
    WindowCount,
    WindowCounter,
    WindowReservation,
)
from tests.integration._support.redis_process import owned_redis


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[tuple[RedisCache, Redis[bytes]]]:
    with owned_redis(tmp_path) as server:
        cache: Final = RedisCache(host=server.host, port=server.port, namespace="typed")
        with Redis(host=server.host, port=server.port) as raw:
            yield cache, raw


@pytest.mark.asyncio
async def test_window_reservations_are_atomic_and_guarded_updates_keep_window_identity(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    reservations: Final = (
        WindowReservation("{scope}:window", "{scope}:requests", 7, 1, 60, 60),
        WindowReservation("{scope}:window", "{scope}:tokens", 70, 10, 60, 60),
    )
    results: Final = await asyncio.gather(
        *(cache.async_reserve_window_capacity(reservations=reservations) for _ in range(20))
    )
    assert sum(isinstance(result, tuple) for result in results) == 7
    assert sum(result == CapacityExceeded(0, 7, 7) for result in results) == 13
    assert raw.mget("typed:{scope}:requests", "typed:{scope}:tokens") == [b"7", b"70"]
    assert raw.exists("{scope}:requests") == 0
    winner: Final = next(result for result in results if isinstance(result, tuple))
    assert isinstance(winner[0].window_start, str)
    assert await cache.async_increment_guarded_counters(
        increments=(GuardedIncrement("{scope}:window", "{scope}:tokens", winner[0].window_start, -5, 0),)
    ) == (CounterUpdate(True, 65),)
    assert await cache.async_increment_guarded_counters(
        increments=(GuardedIncrement("{scope}:window", "{scope}:tokens", "stale", 100, 0),)
    ) == (CounterUpdate(False, 65),)
    assert 0 < raw.ttl("typed:{scope}:requests") <= 60


@pytest.mark.asyncio
@pytest.mark.parametrize("now", (100, 100.5))
async def test_legacy_window_increment_resets_siblings_and_preserves_string_starts(
    backend: tuple[RedisCache, Redis[bytes]], now: float,
) -> None:
    cache, raw = backend
    raw.set("typed:lane:tokens", 99)
    counter: Final = WindowCounter("lane:window", "lane:requests")
    assert await cache.async_increment_window_counters(counters=(counter,), now=now, window_seconds=60) == (
        WindowCount(str(now), 1),
    )
    assert raw.get("typed:lane:tokens") is None
    assert await cache.async_increment_window_counters(counters=(counter,), now=now + 1, window_seconds=60) == (
        WindowCount(str(now), 2),
    )
    assert await cache.async_read_atomic_counts(keys=("lane:requests", "missing")) == (2, None)


@pytest.mark.asyncio
async def test_concurrency_prunes_stale_slots_and_rejects_without_partial_acquisition(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    raw.zadd("typed:gauge", {"stale": 0})
    assert await cache.async_acquire_concurrency(slots=(ConcurrencySlot("gauge", 1, 60, "owner"),)) == (1,)
    assert await cache.async_acquire_concurrency(
        slots=(
            ConcurrencySlot("other", 2, 60, "candidate"),
            ConcurrencySlot("gauge", 1, 60, "candidate"),
        )
    ) == CapacityExceeded(1, 1, 1)
    assert raw.exists("typed:other") == 0
    assert await cache.async_count_concurrency(counters=(ConcurrencyRead("gauge", 60),)) == (1,)
    assert await cache.async_release_concurrency(slots=(ConcurrencyRelease("gauge", "not-owner"),)) == (1,)
    assert await cache.async_release_concurrency(slots=(ConcurrencyRelease("gauge", "owner"),)) == (0,)


@pytest.mark.asyncio
async def test_counter_and_record_capabilities_preserve_ttls_and_tombstones(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    assert await cache.async_increment_iterations(key="iterations", ttl_seconds=120) == 1
    raw.expire("typed:iterations", 30)
    assert await cache.async_increment_iterations(key="iterations", ttl_seconds=120) == 2
    assert 0 < raw.ttl("typed:iterations") <= 30
    assert await cache.async_increment_spend(key="spend", amount=0.25, ttl_seconds=120) == 0.25
    raw.expire("typed:spend", 30)
    assert await cache.async_increment_spend(key="spend", amount=0.5, ttl_seconds=120) == 0.75
    assert 0 < raw.ttl("typed:spend") <= 30
    assert await cache.async_increment_counters(increments=(CounterIncrement("tokens", 1.5, 120),)) == (1.5,)
    raw.expire("typed:tokens", 30)
    assert await cache.async_increment_counters(increments=(CounterIncrement("tokens", 2.5, 120),)) == (4.0,)
    assert 0 < raw.ttl("typed:tokens") <= 30
    assert await cache.async_reserve_counter(key="enqueued", amount=5, limit=8, ttl_seconds=60) == LimitedIncrement(
        True, 5
    )
    assert await cache.async_reserve_counter(key="enqueued", amount=4, limit=8, ttl_seconds=60) == LimitedIncrement(
        False, 5
    )
    assert await cache.async_refund_counter(key="enqueued", amount=7) is None
    assert raw.get("typed:enqueued") is None
    assert await cache.async_save_record(key="record", value='{"tokens":5}', ttl_seconds=90) is None
    assert await cache.async_pop_record(key="record", tombstone_ttl_seconds=25) == '{"tokens":5}'
    assert await cache.async_pop_record(key="record", tombstone_ttl_seconds=25) == ""
    assert await cache.async_pop_record(key="missing", tombstone_ttl_seconds=25) is None
    assert 0 < raw.ttl("typed:record") <= 25


@pytest.mark.asyncio
async def test_affinity_keeps_eligible_winners_and_refreshes_legacy_model_pins(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    assert await cache.async_claim_affinity(key="pin", value={"model_id": "a"}, ttl_seconds=60) == {"model_id": "a"}
    assert await cache.async_claim_affinity(key="pin", value={"model_id": "b"}, ttl_seconds=60) == {"model_id": "a"}
    assert await cache.async_claim_affinity(
        key="pin",
        value={"model_id": "b"},
        ttl_seconds=60,
        eligible_values=({"model_id": "b"},),
    ) == {"model_id": "b"}
    raw.set("typed:legacy", "old")
    assert (
        await cache.async_claim_affinity(
            key="legacy",
            value={"model_id": "new"},
            ttl_seconds=60,
            eligible_values=({"model_id": "old"},),
            preserve_legacy_model_ids=True,
        )
        == "old"
    )
    assert 0 < raw.ttl("typed:legacy") <= 60


@pytest.mark.asyncio
async def test_throttle_source_count_stops_after_pair_is_blocked(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    keys: Final = ThrottleKeys("pair", "pair:block", "source", "source:block")
    assert await cache.async_read_throttle_ttls(keys=keys) == (-2, -2)
    assert await cache.async_record_throttle_failure(
        keys=keys,
        pair_limit=1,
        source_limit=5,
        window_seconds=60,
        block_seconds=90,
    ) == (0, 0)
    assert await cache.async_record_throttle_failure(
        keys=keys,
        pair_limit=1,
        source_limit=5,
        window_seconds=60,
        block_seconds=90,
    ) == (90, 0)
    assert raw.get("typed:source") == b"1"
    pair_ttl, source_ttl = await cache.async_read_throttle_ttls(keys=keys)
    assert 0 < pair_ttl <= 90
    assert source_ttl == -2


@pytest.mark.asyncio
async def test_refresh_lease_only_renews_or_deletes_for_its_owner(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    assert await cache.async_acquire_lease(key="lock", token="owner", ttl_seconds=60) == "acquired"
    assert await cache.async_acquire_lease(key="lock", token="other", ttl_seconds=60) == "held"
    assert await cache.async_extend_lease(key="lock", token="other", ttl_seconds=120) is False
    assert await cache.async_release_lease(key="lock", token="other") is False
    assert await cache.async_lease_held(key="lock") is True
    assert await cache.async_extend_lease(key="lock", token="owner", ttl_seconds=120) is True
    assert 60000 < raw.pttl("typed:lock") <= 120000
    assert await cache.async_release_lease(key="lock", token="owner") is True
    assert await cache.async_lease_held(key="lock") is False


@pytest.mark.asyncio
async def test_typed_batches_retry_noscript_isolate_item_failures_and_settle_post_call_hooks(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    assert await cache.async_increment_iterations(key="warmup", ttl_seconds=60) == 1
    raw.script_flush()
    raw.rpush("typed:wrong-type", "value")
    batch: Final = RedisBatch(cache)
    good: Final = cache.async_increment_iterations(key="good", ttl_seconds=60, batch=batch)
    broken: Final = cache.async_increment_iterations(key="wrong-type", ttl_seconds=60, batch=batch)
    assert await good == 1
    failure: Final = await broken
    assert isinstance(failure, CacheFailure)
    assert failure.kind == "unavailable"
    settled: Final[asyncio.Future[int | CacheFailure]] = asyncio.get_running_loop().create_future()
    with request_redis_batch_scope(post_call_deadline=60) as request:
        post_call: Final = active_post_call_redis_batch(cache)
        operation: Final = cache.async_increment_iterations(key="post-call", ttl_seconds=60, batch=post_call)
        operation.on_settled(lambda result: settled.set_result(result.result()))
        assert raw.get("typed:post-call") is None
        await request.flush_post_call()
    assert await settled == 1
    assert await operation == 1
    assert raw.get("typed:post-call") == b"1"


@pytest.mark.asyncio
async def test_pubsub_and_update_queues_preserve_namespace_payloads_and_order(
    backend: tuple[RedisCache, Redis[bytes]],
) -> None:
    cache, raw = backend
    subscription: Final = await cache.async_subscribe(channel="updates")
    assert not isinstance(subscription, CacheFailure)
    try:
        assert await subscription.receive(timeout_seconds=1) is None
        assert await cache.async_publish_many(messages=(
            CachePublish("updates", "first"), CachePublish("typed:updates", "second"),
        )) == (1, 1)
        assert await subscription.receive(timeout_seconds=1) == b"first"
        assert await subscription.receive(timeout_seconds=1) == b"second"
        assert await cache.async_enqueue_updates(updates=(
            CacheQueueUpdate("queue", ("first", "second")), CacheQueueUpdate("typed:queue", ("third",)),
        )) == (2, 3)
        assert raw.lrange("typed:queue", 0, -1) == [b"first", b"second", b"third"]
        assert raw.exists("queue") == 0
        assert await cache.async_delete_record(key="queue") is True
        assert await cache.async_delete_record(key="queue") is False
    finally:
        assert await subscription.close() is None
