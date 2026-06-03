import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.redis_cache import RedisCircuitBreaker, RedisCache


def make_breaker(failure_threshold: int = 3, recovery_timeout: int = 60):
    return RedisCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )


def test_opens_after_threshold_failures():
    cb = make_breaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open()
    assert cb._state == RedisCircuitBreaker.OPEN


def test_success_resets_failure_count():
    cb = make_breaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert not cb.is_open()


def test_recovery_timeout_controls_half_open_transition():
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()

    cb._opened_at = time.time() - 59
    assert cb.is_open()
    assert cb._state == RedisCircuitBreaker.OPEN

    cb._opened_at = time.time() - 61
    assert not cb.is_open()
    assert cb._state == RedisCircuitBreaker.HALF_OPEN


def test_half_open_concurrent_requests_fast_fail():
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    cb._opened_at = time.time() - 61

    assert not cb.is_open()
    assert cb._state == RedisCircuitBreaker.HALF_OPEN

    assert cb.is_open()
    assert cb.is_open()


def test_half_open_probe_success_closes_breaker():
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    cb._opened_at = time.time() - 61
    cb.is_open()

    cb.record_success()
    assert cb._state == RedisCircuitBreaker.CLOSED
    assert not cb.is_open()
    assert cb._failure_count == 0


def test_half_open_probe_failure_reopens_and_resets_timer():
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    original_opened_at = time.time() - 61
    cb._opened_at = original_opened_at
    cb.is_open()

    cb.record_failure()
    assert cb._state == RedisCircuitBreaker.OPEN
    assert cb._opened_at > original_opened_at




@pytest.fixture
def redis_cache_with_mock_client():
    with patch("asyncio.get_running_loop", side_effect=RuntimeError):
        cache = RedisCache(host="https://my-test-host")
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    cache.init_async_client = MagicMock(return_value=mock_client)
    return cache, mock_client


@pytest.mark.asyncio
async def test_guard_opens_after_threshold_connection_errors(
    redis_cache_with_mock_client,
):
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.side_effect = ConnectionError("Redis unreachable")

    for _ in range(5):
        with pytest.raises(ConnectionError):
            await cache.async_increment(key="key", value=1.0)

    assert cache._circuit_breaker._state == RedisCircuitBreaker.OPEN

    with pytest.raises(Exception, match="circuit breaker is open"):
        await cache.async_increment(key="key", value=1.0)

    assert mock_client.incrbyfloat.await_count == 5


@pytest.mark.asyncio
async def test_guard_fast_fails_without_network_call_when_open(
    redis_cache_with_mock_client,
):
    cache, mock_client = redis_cache_with_mock_client
    cache._circuit_breaker._state = RedisCircuitBreaker.OPEN
    cache._circuit_breaker._opened_at = time.time()
    cache._circuit_breaker._failure_count = 5

    with pytest.raises(Exception, match="circuit breaker is open"):
        await cache.async_increment(key="key", value=1.0)

    mock_client.incrbyfloat.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_breaker_lets_calls_through_after_many_failures(
    redis_cache_with_mock_client,
):
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.side_effect = ConnectionError("Redis unreachable")

    with patch("litellm.caching.redis_cache.REDIS_CIRCUIT_BREAKER_ENABLED", False):
        for _ in range(10):
            with pytest.raises(ConnectionError):
                await cache.async_increment(key="key", value=1.0)

    assert mock_client.incrbyfloat.await_count == 10


@pytest.mark.asyncio
async def test_guard_closes_after_recovery_timeout_and_successful_probe(
    redis_cache_with_mock_client,
):
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.return_value = 1.0
    mock_client.ttl.return_value = -1

    cache._circuit_breaker._state = RedisCircuitBreaker.OPEN
    cache._circuit_breaker._opened_at = time.time() - 61
    cache._circuit_breaker._failure_count = 5

    result = await cache.async_increment(key="key", value=1.0)
    assert result == 1.0
    assert cache._circuit_breaker._state == RedisCircuitBreaker.CLOSED


@pytest.mark.asyncio
async def test_guard_reopens_when_probe_fails(redis_cache_with_mock_client):
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.side_effect = ConnectionError("Still down")

    cache._circuit_breaker._state = RedisCircuitBreaker.OPEN
    cache._circuit_breaker._opened_at = time.time() - 61
    cache._circuit_breaker._failure_count = 5

    with pytest.raises(ConnectionError):
        await cache.async_increment(key="key", value=1.0)

    assert cache._circuit_breaker._state == RedisCircuitBreaker.OPEN


@pytest.mark.asyncio
async def test_async_get_cache_never_trips_breaker_on_failure(
    redis_cache_with_mock_client,
):
    """
    async_get_cache swallows all exceptions internally and returns None.
    This means Redis GET failures are invisible to the circuit breaker — only
    async_increment (and other re-raising methods) can open it.
    If this test ever fails, it means async_get_cache now propagates exceptions,
    which changes the breaker's trigger surface and needs deliberate handling.
    """
    cache, mock_client = redis_cache_with_mock_client
    mock_client.get.side_effect = ConnectionError("Redis unreachable")

    for _ in range(10):
        result = await cache.async_get_cache("some_key")
        assert result is None

    assert cache._circuit_breaker._failure_count == 0
    assert cache._circuit_breaker._state == RedisCircuitBreaker.CLOSED


@pytest.mark.asyncio
async def test_dual_cache_falls_back_to_in_memory_when_circuit_open():
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.in_memory_cache import InMemoryCache

    with patch("asyncio.get_running_loop", side_effect=RuntimeError):
        redis_cache = RedisCache(host="https://my-test-host")

    redis_cache._circuit_breaker._state = RedisCircuitBreaker.OPEN
    redis_cache._circuit_breaker._opened_at = time.time()

    in_memory = InMemoryCache()
    await in_memory.async_set_cache("test_key", "in_memory_value")

    dual = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    result = await dual.async_get_cache("test_key")
    assert result == "in_memory_value"
