import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache_coalesces_concurrent_redis_reads():
    dual_cache = DualCache(
        redis_cache=MagicMock(spec=RedisCache), default_redis_batch_cache_expiry=10
    )
    keys = ["shared_a", "shared_b"]
    start_gate = asyncio.Event()

    async def _mock_async_batch_get_cache(key_list, parent_otel_span=None):
        await asyncio.sleep(0.05)
        return {k: None for k in key_list}

    with patch.object(
        dual_cache.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(side_effect=_mock_async_batch_get_cache),
    ) as mock_async_batch_get_cache:

        async def worker():
            await start_gate.wait()
            return await dual_cache.async_batch_get_cache(keys=keys)

        tasks = [asyncio.create_task(worker()) for _ in range(50)]
        start_gate.set()
        await asyncio.gather(*tasks)

        assert mock_async_batch_get_cache.call_count == 1


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache_rolls_back_redis_reservation_on_error():
    dual_cache = DualCache(
        redis_cache=MagicMock(spec=RedisCache), default_redis_batch_cache_expiry=10
    )
    keys = ["shared_a", "shared_b"]

    with patch.object(
        dual_cache.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(side_effect=RuntimeError("redis unavailable")),
    ) as mock_async_batch_get_cache:
        first_result = await dual_cache.async_batch_get_cache(keys=keys)
        second_result = await dual_cache.async_batch_get_cache(keys=keys)

        assert first_result is None
        assert second_result is None
        assert mock_async_batch_get_cache.call_count == 2
        assert "shared_a" not in dual_cache.last_redis_batch_access_time
        assert "shared_b" not in dual_cache.last_redis_batch_access_time


@pytest.mark.asyncio
async def test_dual_cache_async_set_cache_injects_default_in_memory_ttl():
    """
    Test that async_set_cache injects default_in_memory_ttl into kwargs
    when no explicit ttl is provided, matching the sync set_cache behavior.

    Regression test for: async_set_cache was missing the TTL injection that
    sync set_cache has, causing InMemoryCache to use its own default_ttl (600s)
    instead of DualCache's default_in_memory_ttl.
    """
    in_memory_cache = InMemoryCache(default_ttl=600)
    dual_cache = DualCache(
        in_memory_cache=in_memory_cache,
        default_in_memory_ttl=60,
    )

    before = time.time()
    await dual_cache.async_set_cache(key="test_key", value="test_value")
    after = time.time()

    # The TTL stored should reflect default_in_memory_ttl (60s), not
    # InMemoryCache's default_ttl (600s)
    expiry = in_memory_cache.ttl_dict["test_key"]
    assert expiry >= before + 60
    assert expiry <= after + 60


@pytest.mark.asyncio
async def test_dual_cache_async_set_cache_respects_explicit_ttl():
    """
    Test that async_set_cache does NOT override an explicitly provided ttl.
    """
    in_memory_cache = InMemoryCache(default_ttl=600)
    dual_cache = DualCache(
        in_memory_cache=in_memory_cache,
        default_in_memory_ttl=60,
    )

    before = time.time()
    await dual_cache.async_set_cache(key="test_key", value="test_value", ttl=30)
    after = time.time()

    # The explicit ttl=30 should be used, not default_in_memory_ttl (60)
    expiry = in_memory_cache.ttl_dict["test_key"]
    assert expiry >= before + 30
    assert expiry <= after + 30


@pytest.mark.asyncio
async def test_dual_cache_async_set_cache_pipeline_injects_default_in_memory_ttl():
    """
    Test that async_set_cache_pipeline injects default_in_memory_ttl into kwargs
    when no explicit ttl is provided.
    """
    in_memory_cache = InMemoryCache(default_ttl=600)
    dual_cache = DualCache(
        in_memory_cache=in_memory_cache,
        default_in_memory_ttl=60,
    )

    cache_list = [("key_a", "value_a"), ("key_b", "value_b")]

    before = time.time()
    await dual_cache.async_set_cache_pipeline(cache_list=cache_list)
    after = time.time()

    for key in ["key_a", "key_b"]:
        expiry = in_memory_cache.ttl_dict[key]
        assert expiry >= before + 60
        assert expiry <= after + 60


@pytest.mark.asyncio
async def test_dual_cache_sync_and_async_set_cache_use_same_ttl():
    """
    Test that sync set_cache and async async_set_cache produce the same TTL
    when no explicit ttl is provided, ensuring parity between the two paths.
    """
    in_memory_sync = InMemoryCache(default_ttl=600)
    dual_cache_sync = DualCache(
        in_memory_cache=in_memory_sync,
        default_in_memory_ttl=60,
    )

    in_memory_async = InMemoryCache(default_ttl=600)
    dual_cache_async = DualCache(
        in_memory_cache=in_memory_async,
        default_in_memory_ttl=60,
    )

    dual_cache_sync.set_cache(key="test_key", value="test_value")
    await dual_cache_async.async_set_cache(key="test_key", value="test_value")

    sync_expiry = in_memory_sync.ttl_dict["test_key"]
    async_expiry = in_memory_async.ttl_dict["test_key"]

    # Both should use default_in_memory_ttl=60, so their expiry times
    # should be within a small tolerance of each other
    assert abs(sync_expiry - async_expiry) < 1.0


def test_circuit_breaker_opens_after_threshold():
    """Circuit opens after N consecutive Redis failures."""
    from litellm.caching.redis_cache import RedisCircuitBreaker

    cb = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
    for _ in range(3):
        cb.record_failure()

    assert cb._state == "open"


@pytest.mark.asyncio
async def test_circuit_breaker_open_skips_redis():
    """When circuit is open, the guard decorator raises immediately without calling the method."""
    from litellm.caching.redis_cache import (
        RedisCircuitBreaker,
        _redis_circuit_breaker_guard,
    )

    class FakeRedis:
        def __init__(self):
            self._circuit_breaker = RedisCircuitBreaker(
                failure_threshold=3, recovery_timeout=60
            )
            self._circuit_breaker._state = "open"
            self._circuit_breaker._opened_at = time.time()
            self.call_count = 0

        @_redis_circuit_breaker_guard
        async def do_thing(self):
            self.call_count += 1
            return "result"

    fr = FakeRedis()
    with pytest.raises(Exception, match="circuit breaker is open"):
        await fr.do_thing()

    assert fr.call_count == 0  # method body never executed


def test_circuit_breaker_closes_on_recovery():
    """After recovery_timeout expires, probe is allowed and success closes the circuit."""
    from litellm.caching.redis_cache import RedisCircuitBreaker

    cb = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
    cb._state = "open"
    cb._opened_at = time.time() - 9999  # recovery timeout long expired

    # is_open() should return False to allow a probe through, and transition to HALF_OPEN
    assert cb.is_open() is False
    assert cb._state == "half_open"

    # Successful probe closes the circuit
    cb.record_success()
    assert cb._state == "closed"
