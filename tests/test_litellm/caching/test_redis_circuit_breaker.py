"""
Tests for RedisCircuitBreaker and the _redis_circuit_breaker_guard decorator.

Covers:
- CLOSED -> OPEN transition after failure_threshold consecutive failures
- OPEN state fast-fails without touching Redis
- OPEN -> HALF_OPEN after recovery_timeout seconds
- HALF_OPEN -> CLOSED on probe success
- HALF_OPEN -> OPEN on probe failure (resets timer)
- Concurrent HALF_OPEN requests: only probe caller gets through, rest fast-fail
- REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT controls open duration (the 60s default)
- Success resets failure counter (failures must be consecutive)
- Sync methods (set_cache, get_cache) are NOT guarded — still attempt Redis when open
- DualCache falls back to in-memory when circuit breaker raises
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.redis_cache import RedisCircuitBreaker, RedisCache


# ---------------------------------------------------------------------------
# RedisCircuitBreaker unit tests (state machine only, no I/O)
# ---------------------------------------------------------------------------


def make_breaker(failure_threshold: int = 3, recovery_timeout: int = 60):
    return RedisCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )


def test_initial_state_is_closed():
    cb = make_breaker()
    assert not cb.is_open()
    assert cb._state == RedisCircuitBreaker.CLOSED


def test_single_failure_does_not_open():
    cb = make_breaker(failure_threshold=3)
    cb.record_failure()
    assert not cb.is_open()
    assert cb._state == RedisCircuitBreaker.CLOSED


def test_opens_after_threshold_failures():
    cb = make_breaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open()
    assert cb._state == RedisCircuitBreaker.OPEN


def test_success_resets_failure_count():
    """Failures must be consecutive — a success resets the counter."""
    cb = make_breaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()  # only 1 failure after reset — should not open
    assert not cb.is_open()


def test_open_state_is_open_returns_true():
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    assert cb.is_open()


def test_recovery_timeout_controls_half_open_transition():
    """
    This directly tests the ~60s default behavior reported in production.
    With recovery_timeout=60, the breaker stays OPEN until 60s elapses.
    We verify the transition by faking time.
    """
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    assert cb._state == RedisCircuitBreaker.OPEN

    # Simulate 59 seconds passing — still OPEN
    cb._opened_at = time.time() - 59
    assert cb.is_open()
    assert cb._state == RedisCircuitBreaker.OPEN

    # Simulate 61 seconds passing — transitions to HALF_OPEN, probe gets through
    cb._opened_at = time.time() - 61
    assert not cb.is_open()  # probe caller gets False (allowed through)
    assert cb._state == RedisCircuitBreaker.HALF_OPEN


def test_half_open_concurrent_requests_fast_fail():
    """
    Once in HALF_OPEN, only the probe caller gets is_open()=False.
    All subsequent callers see HALF_OPEN and get is_open()=True (fast-fail).
    """
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    cb._opened_at = time.time() - 61  # force transition on next is_open()

    # First caller becomes the probe
    assert not cb.is_open()
    assert cb._state == RedisCircuitBreaker.HALF_OPEN

    # All subsequent callers are fast-failed
    assert cb.is_open()
    assert cb.is_open()


def test_half_open_probe_success_closes_breaker():
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    cb._opened_at = time.time() - 61
    cb.is_open()  # advance to HALF_OPEN

    cb.record_success()
    assert cb._state == RedisCircuitBreaker.CLOSED
    assert not cb.is_open()
    assert cb._failure_count == 0


def test_half_open_probe_failure_reopens_and_resets_timer():
    cb = make_breaker(failure_threshold=1, recovery_timeout=60)
    cb.record_failure()
    original_opened_at = time.time() - 61
    cb._opened_at = original_opened_at
    cb.is_open()  # advance to HALF_OPEN

    cb.record_failure()  # probe fails
    assert cb._state == RedisCircuitBreaker.OPEN
    # Timer was reset — new _opened_at is later than the original
    assert cb._opened_at > original_opened_at


def test_failure_threshold_default_is_five():
    """
    Production default: REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5.
    Verify 4 failures don't open and 5 do.
    """
    with patch("asyncio.get_running_loop", side_effect=RuntimeError):
        cache = RedisCache(host="https://my-test-host")
    cb = cache._circuit_breaker
    assert cb.failure_threshold == 5

    for _ in range(4):
        cb.record_failure()
    assert not cb.is_open()

    cb.record_failure()
    assert cb.is_open()


def test_breaker_disabled_never_opens():
    """
    REDIS_CIRCUIT_BREAKER_ENABLED=false: breaker stays CLOSED regardless of failures.
    Callers always get is_open()=False and record_failure() is a no-op.
    """
    cb = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60, enabled=False)
    for _ in range(10):
        cb.record_failure()
    assert not cb.is_open()
    assert cb._state == RedisCircuitBreaker.CLOSED
    assert cb._failure_count == 0


def test_recovery_timeout_default_is_sixty_seconds():
    """
    Production default: REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60.
    This is exactly the ~60s user-visible latency reported in the issue.
    """
    with patch("asyncio.get_running_loop", side_effect=RuntimeError):
        cache = RedisCache(host="https://my-test-host")
    assert cache._circuit_breaker.recovery_timeout == 60


# ---------------------------------------------------------------------------
# _redis_circuit_breaker_guard integration tests (async method interception)
# ---------------------------------------------------------------------------


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
async def test_guard_allows_call_when_closed(redis_cache_with_mock_client):
    """Happy path: CLOSED state lets the call through and returns the result."""
    cache, mock_client = redis_cache_with_mock_client
    mock_client.get.return_value = b'"value"'
    result = await cache.async_get_cache("some_key")
    assert result == "value"
    mock_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_guard_counts_connection_error_as_failure(redis_cache_with_mock_client):
    """
    async_increment re-raises exceptions, so ConnectionError propagates through
    the guard and increments the failure counter.
    """
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.side_effect = ConnectionError("Redis unreachable")

    with pytest.raises(ConnectionError):
        await cache.async_increment(key="key", value=1.0)

    assert cache._circuit_breaker._failure_count == 1


@pytest.mark.asyncio
async def test_guard_opens_after_threshold_connection_errors(
    redis_cache_with_mock_client,
):
    """
    5 consecutive ConnectionErrors open the breaker; the 6th call is fast-failed.
    Uses async_increment because it re-raises exceptions (unlike async_get_cache
    which swallows them and returns None).
    """
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.side_effect = ConnectionError("Redis unreachable")

    for _ in range(5):
        with pytest.raises(ConnectionError):
            await cache.async_increment(key="key", value=1.0)

    assert cache._circuit_breaker._state == RedisCircuitBreaker.OPEN

    # 6th call — circuit is open, Redis must NOT be called
    with pytest.raises(Exception, match="circuit breaker is open"):
        await cache.async_increment(key="key", value=1.0)

    assert mock_client.incrbyfloat.await_count == 5  # not 6


@pytest.mark.asyncio
async def test_guard_fast_fails_without_network_call_when_open(
    redis_cache_with_mock_client,
):
    cache, mock_client = redis_cache_with_mock_client
    # Force breaker open directly
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
    """
    With REDIS_CIRCUIT_BREAKER_ENABLED=false, repeated failures never open the
    breaker — subsequent calls still reach Redis.
    """
    cache, mock_client = redis_cache_with_mock_client
    cache._circuit_breaker.enabled = False
    mock_client.incrbyfloat.side_effect = ConnectionError("Redis unreachable")

    for _ in range(10):
        with pytest.raises(ConnectionError):
            await cache.async_increment(key="key", value=1.0)

    # Breaker must still be CLOSED and failure counter untouched
    assert cache._circuit_breaker._state == RedisCircuitBreaker.CLOSED
    assert cache._circuit_breaker._failure_count == 0
    # Every call reached Redis — no fast-fails
    assert mock_client.incrbyfloat.await_count == 10


@pytest.mark.asyncio
async def test_guard_closes_after_recovery_timeout_and_successful_probe(
    redis_cache_with_mock_client,
):
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.return_value = 1.0
    mock_client.ttl.return_value = -1

    # Force OPEN with expired timer
    cache._circuit_breaker._state = RedisCircuitBreaker.OPEN
    cache._circuit_breaker._opened_at = time.time() - 61
    cache._circuit_breaker._failure_count = 5

    # Probe succeeds — breaker should close
    result = await cache.async_increment(key="key", value=1.0)
    assert result == 1.0
    assert cache._circuit_breaker._state == RedisCircuitBreaker.CLOSED


@pytest.mark.asyncio
async def test_guard_reopens_when_probe_fails(redis_cache_with_mock_client):
    cache, mock_client = redis_cache_with_mock_client
    mock_client.incrbyfloat.side_effect = ConnectionError("Still down")

    # Force OPEN with expired timer
    cache._circuit_breaker._state = RedisCircuitBreaker.OPEN
    cache._circuit_breaker._opened_at = time.time() - 61
    cache._circuit_breaker._failure_count = 5

    with pytest.raises(ConnectionError):
        await cache.async_increment(key="key", value=1.0)

    assert cache._circuit_breaker._state == RedisCircuitBreaker.OPEN


# ---------------------------------------------------------------------------
# Sync methods are NOT guarded (regression guard)
# ---------------------------------------------------------------------------


def test_sync_get_cache_attempts_redis_when_circuit_is_open():
    """
    Sync methods have no circuit breaker guard — they still hit Redis when open.
    This test documents the gap so any future fix is deliberate.
    """
    with patch("asyncio.get_running_loop", side_effect=RuntimeError):
        cache = RedisCache(host="https://my-test-host")

    cache._circuit_breaker._state = RedisCircuitBreaker.OPEN
    cache._circuit_breaker._opened_at = time.time()

    mock_sync_client = MagicMock()
    mock_sync_client.get.return_value = None
    cache.redis_client = mock_sync_client

    cache.get_cache("any_key")

    # Sync client WAS called even though breaker is open
    mock_sync_client.get.assert_called_once()


# ---------------------------------------------------------------------------
# DualCache fallback behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dual_cache_falls_back_to_in_memory_when_circuit_open():
    """
    When the Redis circuit breaker raises, DualCache must return the in-memory
    value rather than propagating the exception to the caller.
    """
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
