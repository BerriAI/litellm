import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fakeredis import FakeRedis, FakeServer
from fakeredis.aioredis import FakeRedis as FakeAsyncRedis

from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError
from litellm.constants import DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager


class MockRedisCache:
    def __init__(self):
        self.async_set_cache = AsyncMock()
        self.async_get_cache = AsyncMock()
        self.async_delete_cache = AsyncMock()


@pytest.fixture
def mock_redis():
    return MockRedisCache()


@pytest.fixture
def pod_lock_manager(mock_redis):
    return PodLockManager(redis_cache=mock_redis)


@pytest.mark.asyncio
async def test_acquire_lock_success(pod_lock_manager, mock_redis):
    """
    Test that the lock is acquired successfully when no existing lock exists
    """
    # Mock successful acquisition (SET NX returns True)
    mock_redis.async_set_cache.return_value = True

    result = await pod_lock_manager.acquire_lock(
        cronjob_id="test_job",
    )
    assert result == True

    # Verify set_cache was called with correct parameters
    lock_key = pod_lock_manager.get_redis_lock_key(cronjob_id="test_job")
    mock_redis.async_set_cache.assert_called_once_with(
        lock_key,
        pod_lock_manager.pod_id,
        nx=True,
        ttl=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS,
    )


@pytest.mark.asyncio
async def test_acquire_lock_existing_active(pod_lock_manager, mock_redis):
    """
    Test that the lock is not acquired if there's an active lock by different pod
    """
    # Mock failed acquisition (SET NX returns False)
    mock_redis.async_set_cache.return_value = False
    # Mock get_cache to return a different pod's ID
    mock_redis.async_get_cache.return_value = "different_pod_id"

    result = await pod_lock_manager.acquire_lock(
        cronjob_id="test_job",
    )
    assert result == False

    # Verify set_cache was called
    mock_redis.async_set_cache.assert_called_once()
    # Verify get_cache was called to check existing lock
    lock_key = pod_lock_manager.get_redis_lock_key(cronjob_id="test_job")
    mock_redis.async_get_cache.assert_called_once_with(lock_key)


@pytest.mark.asyncio
async def test_acquire_lock_expired(pod_lock_manager, mock_redis):
    """
    Test that the lock can be acquired if existing lock is expired
    """
    # Mock failed acquisition first (SET NX returns False)
    mock_redis.async_set_cache.return_value = False

    # Simulate an expired lock by having the TTL return a value
    # Since Redis auto-expires keys, an expired lock would be absent
    # So we'll simulate a retry after the first check fails

    # First check returns a value (lock exists)
    mock_redis.async_get_cache.return_value = "different_pod_id"

    # Then set succeeds on retry (simulating key expiring between checks)
    mock_redis.async_set_cache.side_effect = [False, True]

    result = await pod_lock_manager.acquire_lock(
        cronjob_id="test_job",
    )
    assert result == False  # First attempt fails

    # Reset mock for a second attempt
    mock_redis.async_set_cache.reset_mock()
    mock_redis.async_set_cache.return_value = True

    # Try again (simulating the lock expired)
    result = await pod_lock_manager.acquire_lock(
        cronjob_id="test_job",
    )
    assert result == True

    # Verify set_cache was called again
    mock_redis.async_set_cache.assert_called_once()


@pytest.mark.asyncio
async def test_release_lock_success(pod_lock_manager, mock_redis):
    """
    Test that the release lock works when the current pod holds the lock
    """
    # Mock get_cache to return this pod's ID
    mock_redis.async_get_cache.return_value = pod_lock_manager.pod_id
    # Mock successful deletion
    mock_redis.async_delete_cache.return_value = 1

    await pod_lock_manager.release_lock(
        cronjob_id="test_job",
    )

    # Verify get_cache was called
    lock_key = pod_lock_manager.get_redis_lock_key(cronjob_id="test_job")
    mock_redis.async_get_cache.assert_called_once_with(lock_key)
    # Verify delete_cache was called
    mock_redis.async_delete_cache.assert_called_once_with(lock_key)


@pytest.mark.asyncio
async def test_release_lock_different_pod(pod_lock_manager, mock_redis):
    """
    Test that the release lock doesn't delete when a different pod holds the lock
    """
    # Mock get_cache to return a different pod's ID
    mock_redis.async_get_cache.return_value = "different_pod_id"

    await pod_lock_manager.release_lock(
        cronjob_id="test_job",
    )

    # Verify get_cache was called
    lock_key = pod_lock_manager.get_redis_lock_key(cronjob_id="test_job")
    mock_redis.async_get_cache.assert_called_once_with(lock_key)
    # Verify delete_cache was NOT called
    mock_redis.async_delete_cache.assert_not_called()


@pytest.mark.asyncio
async def test_release_lock_no_lock(pod_lock_manager, mock_redis):
    """
    Test release lock behavior when no lock exists
    """
    # Mock get_cache to return None (no lock)
    mock_redis.async_get_cache.return_value = None

    await pod_lock_manager.release_lock(
        cronjob_id="test_job",
    )

    # Verify get_cache was called
    lock_key = pod_lock_manager.get_redis_lock_key(cronjob_id="test_job")
    mock_redis.async_get_cache.assert_called_once_with(lock_key)
    # Verify delete_cache was NOT called
    mock_redis.async_delete_cache.assert_not_called()


@pytest.mark.asyncio
async def test_redis_none(monkeypatch):
    """
    Test behavior when redis_cache is None
    """
    pod_lock_manager = PodLockManager(redis_cache=None)

    # Test acquire_lock with None redis_cache
    assert (
        await pod_lock_manager.acquire_lock(
            cronjob_id="test_job",
        )
        is None
    )

    # Test release_lock with None redis_cache (should not raise exception)
    await pod_lock_manager.release_lock(
        cronjob_id="test_job",
    )


@pytest.mark.asyncio
async def test_redis_error_handling(pod_lock_manager, mock_redis):
    """
    Test error handling in Redis operations
    """
    # Mock exceptions for Redis operations
    mock_redis.async_set_cache.side_effect = Exception("Redis error")
    mock_redis.async_get_cache.side_effect = Exception("Redis error")
    mock_redis.async_delete_cache.side_effect = Exception("Redis error")

    # Test acquire_lock error handling
    result = await pod_lock_manager.acquire_lock(
        cronjob_id="test_job",
    )
    assert result == False

    # Reset side effect for get_cache for the release test
    mock_redis.async_get_cache.side_effect = None
    mock_redis.async_get_cache.return_value = pod_lock_manager.pod_id

    # Test release_lock error handling (should not raise exception)
    await pod_lock_manager.release_lock(
        cronjob_id="test_job",
    )


@pytest.mark.asyncio
async def test_lock_refused_by_the_open_circuit_breaker_is_not_logged_as_an_error(pod_lock_manager, mock_redis, caplog):
    """Every cron job retries its lock on a timer, so an open breaker must not add an error line per cycle."""
    refused = RedisCircuitBreakerOpenError("Redis circuit breaker is open - skipping async_set_cache")
    mock_redis.async_set_cache.side_effect = refused
    mock_redis.async_get_cache.return_value = pod_lock_manager.pod_id
    mock_redis.async_delete_cache.side_effect = refused

    with caplog.at_level(logging.ERROR):
        acquired = await pod_lock_manager.acquire_lock(cronjob_id="test_job")
        await pod_lock_manager.release_lock(cronjob_id="test_job")

    assert acquired is False
    assert caplog.records == []


@pytest.mark.asyncio
async def test_bytes_handling(pod_lock_manager, mock_redis):
    """
    Test handling of bytes values from Redis
    """
    # Mock failed acquisition
    mock_redis.async_set_cache.return_value = False
    # Mock get_cache to return bytes
    mock_redis.async_get_cache.return_value = pod_lock_manager.pod_id.encode("utf-8")

    result = await pod_lock_manager.acquire_lock(
        cronjob_id="test_job",
    )
    assert result == True

    # Reset for release test
    mock_redis.async_get_cache.return_value = pod_lock_manager.pod_id.encode("utf-8")
    mock_redis.async_delete_cache.return_value = 1

    await pod_lock_manager.release_lock(
        cronjob_id="test_job",
    )
    mock_redis.async_delete_cache.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_lock_acquisition_simulation():
    """
    Simulate multiple pods trying to acquire the lock simultaneously
    """
    mock_redis = MockRedisCache()
    pod1 = PodLockManager(redis_cache=mock_redis)
    pod2 = PodLockManager(redis_cache=mock_redis)
    pod3 = PodLockManager(redis_cache=mock_redis)

    # Simulate first pod getting the lock
    mock_redis.async_set_cache.return_value = True

    # First pod should get the lock
    result1 = await pod1.acquire_lock(
        cronjob_id="test_job",
    )
    assert result1 == True

    # Simulate other pods failing to get the lock
    mock_redis.async_set_cache.return_value = False
    mock_redis.async_get_cache.return_value = pod1.pod_id

    # Other pods should fail to acquire
    result2 = await pod2.acquire_lock(
        cronjob_id="test_job",
    )
    result3 = await pod3.acquire_lock(
        cronjob_id="test_job",
    )

    # Since other pods don't have the lock, they should get False
    assert result2 == False
    assert result3 == False


@pytest.mark.asyncio
async def test_lock_takeover_race_condition(mock_redis):
    """
    Test scenario where multiple pods try to take over an expired lock using Redis
    """
    pod1 = PodLockManager(redis_cache=mock_redis)
    pod2 = PodLockManager(redis_cache=mock_redis)

    # Simulate first pod's acquisition succeeding
    mock_redis.async_set_cache.return_value = True

    # First pod should successfully acquire
    result1 = await pod1.acquire_lock(
        cronjob_id="test_job",
    )
    assert result1 == True

    # Simulate race condition: second pod tries but fails
    mock_redis.async_set_cache.return_value = False
    mock_redis.async_get_cache.return_value = pod1.pod_id

    # Second pod should fail to acquire
    result2 = await pod2.acquire_lock(
        cronjob_id="test_job",
    )
    assert result2 == False


@pytest_asyncio.fixture(loop_scope="function")
async def atomic_lock_backend() -> AsyncIterator[tuple[RedisCache, FakeAsyncRedis, AsyncMock]]:
    server: Final = FakeServer()
    transport: Final = FakeAsyncRedis(server=server)

    async def compare_delete(keys: Sequence[str], args: Sequence[str], client: object = None) -> int:
        if await transport.get(keys[0]) == args[0].encode():
            return await transport.delete(keys[0])
        return 0

    release: Final = AsyncMock(side_effect=compare_delete)
    with (
        patch("litellm._redis.get_redis_client", return_value=FakeRedis(server=server)),
        patch("litellm._redis.get_redis_async_client", return_value=transport),
        patch.object(transport, "register_script", return_value=release),
    ):
        cache: Final = RedisCache(host="127.0.0.1", port=6379, namespace="pod-test")
        yield cache, transport, release
    await transport.aclose()


@pytest.mark.asyncio
async def test_release_lock_deletes_json_encoded_lock_and_allows_the_next_pod(atomic_lock_backend) -> None:
    cache, transport, release = atomic_lock_backend
    holder: Final = PodLockManager(redis_cache=cache)
    other: Final = PodLockManager(redis_cache=cache)
    assert await holder.acquire_lock(cronjob_id="spend") is True
    assert await transport.get("pod-test:cronjob_lock:spend") == json.dumps(holder.pod_id).encode()

    await holder.release_lock(cronjob_id="spend")

    assert await transport.get("pod-test:cronjob_lock:spend") is None
    assert await other.acquire_lock(cronjob_id="spend") is True
    release.assert_awaited_once_with(keys=("pod-test:cronjob_lock:spend",), args=(json.dumps(holder.pod_id),), client=None)


@pytest.mark.asyncio
async def test_release_lock_preserves_lock_held_by_other_pod(atomic_lock_backend) -> None:
    cache, transport, _ = atomic_lock_backend
    holder: Final = PodLockManager(redis_cache=cache)
    other: Final = PodLockManager(redis_cache=cache)
    assert await holder.acquire_lock(cronjob_id="spend") is True

    await other.release_lock(cronjob_id="spend")

    assert await transport.get("pod-test:cronjob_lock:spend") == json.dumps(holder.pod_id).encode()


@pytest.mark.asyncio
async def test_release_lock_falls_back_to_get_del_when_atomic_execution_fails(atomic_lock_backend) -> None:
    cache, transport, release = atomic_lock_backend
    holder: Final = PodLockManager(redis_cache=cache)
    assert await holder.acquire_lock(cronjob_id="spend") is True
    release.side_effect = OSError("scripting unavailable")

    await holder.release_lock(cronjob_id="spend")

    assert await transport.get("pod-test:cronjob_lock:spend") is None


@pytest.mark.asyncio
async def test_acquire_lock_own_lock_not_reentrant(pod_lock_manager, mock_redis):
    """With allow_reentrant=False a live lock means the window's work is done, so even
    the holder gets False; the default stays reentrant for leader-election callers."""
    mock_redis.async_set_cache.return_value = False
    mock_redis.async_get_cache.return_value = pod_lock_manager.pod_id

    assert await pod_lock_manager.acquire_lock(cronjob_id="test_job", allow_reentrant=False) is False
    assert await pod_lock_manager.acquire_lock(cronjob_id="test_job") is True
