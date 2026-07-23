import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock

from litellm.caching.redis_cache import RedisCache


@pytest.fixture
def redis_no_ping():
    """Patch RedisCache initialization to prevent async ping tasks from being created"""
    with patch("asyncio.get_running_loop") as mock_get_loop:
        # Either raise an exception or return a mock that will handle the task creation
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.mark.parametrize("namespace", [None, "test"])
@pytest.mark.asyncio
async def test_redis_cache_async_increment(namespace, monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()

    # Make sure the mock can be used as an async context manager
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    assert redis_cache is not None

    expected_key = "test:test" if namespace else "test"

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        # Call async_set_cache
        await redis_cache.async_increment(key=expected_key, value=1)

        # Verify that the set method was called on the mock Redis instance
        mock_redis_instance.incrbyfloat.assert_called_once_with(
            name=expected_key, amount=1
        )


@pytest.mark.asyncio
async def test_redis_cache_async_increment_refresh_ttl_true_bumps_existing_ttl(
    monkeypatch, redis_no_ping
):
    """With refresh_ttl=True, every increment should call expire() to bump
    the TTL, even when the key already has a TTL (counter-style use)."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None
    mock_redis_instance.ttl.return_value = 42  # key already has ~42s left

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_increment(
            key="spend:team_member:u:t", value=0.05, refresh_ttl=True
        )

    mock_redis_instance.expire.assert_awaited_once_with("spend:team_member:u:t", 60)
    mock_redis_instance.ttl.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_cache_async_increment_default_uses_expire_nx(
    monkeypatch, redis_no_ping
):
    """Default (refresh_ttl=False) uses EXPIRE NX to preserve window-style
    semantics in a single RTT (no explicit TTL read)."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_increment(key="rate_limit:window", value=1)

    mock_redis_instance.expire.assert_awaited_once_with(
        "rate_limit:window", 60, nx=True
    )
    mock_redis_instance.ttl.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_cache_async_increment_default_falls_back_when_expire_nx_unsupported(
    monkeypatch, redis_no_ping
):
    """If expire(nx=True) is unsupported by the client, fallback to
    TTL-check + conditional EXPIRE for compatibility."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None
    mock_redis_instance.expire.side_effect = [
        TypeError("unexpected keyword argument 'nx'"),
        True,
    ]
    mock_redis_instance.ttl.return_value = -1

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_increment(key="rate_limit:window", value=1)

    assert mock_redis_instance.expire.await_count == 2
    assert mock_redis_instance.expire.await_args_list[0].args == (
        "rate_limit:window",
        60,
    )
    assert mock_redis_instance.expire.await_args_list[0].kwargs == {"nx": True}
    assert mock_redis_instance.expire.await_args_list[1].args == (
        "rate_limit:window",
        60,
    )
    assert mock_redis_instance.expire.await_args_list[1].kwargs == {}
    mock_redis_instance.ttl.assert_awaited_once_with("rate_limit:window")


@pytest.mark.asyncio
async def test_redis_cache_async_increment_default_fallback_existing_ttl_skips_second_expire(
    monkeypatch, redis_no_ping
):
    """When expire(nx=True) is unsupported and key already has a TTL, fallback
    should not issue a second expire() call."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None
    mock_redis_instance.expire.side_effect = [
        TypeError("unexpected keyword argument 'nx'"),
    ]
    mock_redis_instance.ttl.return_value = 42

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_increment(key="rate_limit:window", value=1)

    assert mock_redis_instance.expire.await_count == 1
    assert mock_redis_instance.expire.await_args_list[0].args == (
        "rate_limit:window",
        60,
    )
    assert mock_redis_instance.expire.await_args_list[0].kwargs == {"nx": True}
    mock_redis_instance.ttl.assert_awaited_once_with("rate_limit:window")


@pytest.mark.asyncio
async def test_redis_cache_async_increment_default_falls_back_on_redis6_expire_nx_syntax_error(
    monkeypatch, redis_no_ping
):
    """If Redis server rejects EXPIRE ... NX, fallback to TTL-check + conditional EXPIRE."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    class ResponseError(Exception):
        pass

    mock_redis_instance.expire.side_effect = [
        ResponseError("ERR syntax error"),
        True,
    ]
    mock_redis_instance.ttl.return_value = -1

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_increment(key="rate_limit:window", value=1)

    assert mock_redis_instance.expire.await_count == 2
    assert mock_redis_instance.expire.await_args_list[0].kwargs == {"nx": True}
    assert mock_redis_instance.expire.await_args_list[1].kwargs == {}
    mock_redis_instance.ttl.assert_awaited_once_with("rate_limit:window")


@pytest.mark.asyncio
async def test_redis_cache_async_increment_default_raises_non_compat_expire_error(
    monkeypatch, redis_no_ping
):
    """Non-compatibility expire errors should still propagate."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    class ResponseError(Exception):
        pass

    mock_redis_instance.expire.side_effect = ResponseError("READONLY You can't write")

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        with pytest.raises(ResponseError, match="READONLY"):
            await redis_cache.async_increment(key="rate_limit:window", value=1)

    mock_redis_instance.ttl.assert_not_awaited()


@pytest.mark.parametrize("namespace", [None, "litellm"])
@pytest.mark.asyncio
async def test_async_delete_cache_applies_namespace(
    namespace, monkeypatch, redis_no_ping
):
    """async_delete_cache must prefix keys with the namespace, matching every
    other cache operation. Without this, Redis NOPERM errors occur when an
    ACL restricts DEL to the litellm:* pattern."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_delete_cache(key="3997c4abcdef")

    expected_key = "litellm:3997c4abcdef" if namespace else "3997c4abcdef"
    mock_redis_instance.delete.assert_awaited_once_with(expected_key)


@pytest.mark.parametrize("namespace", [None, "litellm"])
def test_delete_cache_applies_namespace(namespace, monkeypatch, redis_no_ping):
    """delete_cache must prefix keys with the namespace, matching every other
    cache operation."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_client = MagicMock()
    redis_cache.redis_client = mock_redis_client

    redis_cache.delete_cache(key="3997c4abcdef")

    expected_key = "litellm:3997c4abcdef" if namespace else "3997c4abcdef"
    mock_redis_client.delete.assert_called_once_with(expected_key)


@pytest.mark.asyncio
async def test_redis_client_init_with_socket_timeout(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "my-fake-host")
    redis_cache = RedisCache(socket_timeout=1.0)
    assert redis_cache.redis_kwargs["socket_timeout"] == 1.0
    client = redis_cache.init_async_client()
    assert client is not None
    assert client.connection_pool.connection_kwargs["socket_timeout"] == 1.0


@pytest.mark.asyncio
async def test_redis_cache_async_batch_get_cache(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()

    # Make sure the mock can be used as an async context manager
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    # Setup the return value for mget
    mock_redis_instance.mget.return_value = [
        b'{"key1": "value1"}',
        None,
        b'{"key3": "value3"}',
    ]

    test_keys = ["key1", "key2", "key3"]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        # Call async_batch_get_cache
        result = await redis_cache.async_batch_get_cache(key_list=test_keys)

        # Verify mget was called with the correct keys
        mock_redis_instance.mget.assert_called_once()

        # Check that results were properly decoded
        assert result["key1"] == {"key1": "value1"}
        assert result["key2"] is None
        assert result["key3"] == {"key3": "value3"}


@pytest.mark.asyncio
async def test_handle_lpop_count_for_older_redis_versions(monkeypatch):
    """Test the helper method that handles LPOP with count for Redis versions < 7.0"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    # Create RedisCache instance
    redis_cache = RedisCache()

    # Create a mock pipeline
    mock_pipeline = AsyncMock()
    # Set up execute to return different values each time
    mock_pipeline.execute.side_effect = [
        [b"value1"],  # First execute returns first value
        [b"value2"],  # Second execute returns second value
    ]

    # Test the helper method
    result = await redis_cache.handle_lpop_count_for_older_redis_versions(
        pipe=mock_pipeline, key="test_key", count=2
    )

    # Verify results
    assert result == [b"value1", b"value2"]
    assert mock_pipeline.lpop.call_count == 2
    assert mock_pipeline.execute.call_count == 2


@pytest.mark.asyncio
async def test_async_rpush_pipeline_executes_all_operations(monkeypatch, redis_no_ping):
    """Verify that multiple rpush ops are batched into a single pipeline execute"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.rpush = MagicMock()
    mock_pipeline.execute = AsyncMock(return_value=[3, 5, 1])
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    from litellm.types.caching import RedisPipelineRpushOperation

    rpush_list = [
        RedisPipelineRpushOperation(key="key1", values=["a", "b"]),
        RedisPipelineRpushOperation(key="key2", values=["c"]),
        RedisPipelineRpushOperation(key="key3", values=["d", "e", "f"]),
    ]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        result = await redis_cache.async_rpush_pipeline(rpush_list=rpush_list)

    assert result == [3, 5, 1]
    assert mock_pipeline.rpush.call_count == 3
    mock_pipeline.rpush.assert_any_call("key1", "a", "b")
    mock_pipeline.rpush.assert_any_call("key2", "c")
    mock_pipeline.rpush.assert_any_call("key3", "d", "e", "f")
    mock_pipeline.execute.assert_called_once()


@pytest.mark.asyncio
async def test_async_rpush_pipeline_empty_list_returns_empty(
    monkeypatch, redis_no_ping
):
    """Empty rpush_list should return empty list without touching Redis"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        result = await redis_cache.async_rpush_pipeline(rpush_list=[])

    assert result == []
    mock_redis_instance.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_async_rpush_pipeline_raises_on_redis_error(monkeypatch, redis_no_ping):
    """Pipeline errors should propagate"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.rpush = MagicMock()
    mock_pipeline.execute = AsyncMock(side_effect=ConnectionError("Redis down"))
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    from litellm.types.caching import RedisPipelineRpushOperation

    rpush_list = [RedisPipelineRpushOperation(key="key1", values=["a"])]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        with pytest.raises(ConnectionError, match="Redis down"):
            await redis_cache.async_rpush_pipeline(rpush_list=rpush_list)


@pytest.mark.asyncio
async def test_async_lpop_pipeline_single_round_trip(monkeypatch, redis_no_ping):
    """Verify that multiple lpop ops are batched into a single pipeline execute"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    redis_cache.redis_version = "7.0.0"

    mock_redis_instance = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.lpop = MagicMock()
    mock_pipeline.execute = AsyncMock(
        return_value=[
            [b"val1", b"val2"],  # key1 results
            None,  # key2 empty
            [b"val3"],  # key3 results
        ]
    )
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    from litellm.types.caching import RedisPipelineLpopOperation

    lpop_list = [
        RedisPipelineLpopOperation(key="key1", count=10),
        RedisPipelineLpopOperation(key="key2", count=10),
        RedisPipelineLpopOperation(key="key3", count=5),
    ]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        results = await redis_cache.async_lpop_pipeline(lpop_list=lpop_list)

    assert len(results) == 3
    assert results[0] == ["val1", "val2"]
    assert results[1] is None
    assert results[2] == ["val3"]
    mock_pipeline.execute.assert_called_once()


@pytest.mark.asyncio
async def test_async_lpop_pipeline_redis_lt7_regroups_flat_results(
    monkeypatch, redis_no_ping
):
    """Verify Redis < 7 fallback issues individual LPOPs and regroups correctly"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    redis_cache.redis_version = "6.2.0"

    mock_redis_instance = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.lpop = MagicMock()

    # With count=3 for key1 and count=2 for key2, we get 5 individual LPOP commands
    # Simulate: key1 has 2 values then None, key2 has 1 value then None
    mock_pipeline.execute = AsyncMock(
        return_value=[
            b"val1",
            b"val2",
            None,  # 3 LPOPs for key1
            b"val3",
            None,  # 2 LPOPs for key2
        ]
    )
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    from litellm.types.caching import RedisPipelineLpopOperation

    lpop_list = [
        RedisPipelineLpopOperation(key="key1", count=3),
        RedisPipelineLpopOperation(key="key2", count=2),
    ]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        results = await redis_cache.async_lpop_pipeline(lpop_list=lpop_list)

    assert len(results) == 2
    assert results[0] == ["val1", "val2"]  # 2 values, None filtered out
    assert results[1] == ["val3"]  # 1 value, None filtered out
    # All 5 individual LPOPs should be queued, but only 1 execute() call
    assert mock_pipeline.lpop.call_count == 5
    mock_pipeline.execute.assert_called_once()


@pytest.mark.asyncio
async def test_async_rpush_pipeline_raises_on_per_command_error(
    monkeypatch, redis_no_ping
):
    """Verify that per-command errors in pipeline results are raised, not silently dropped"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.rpush = MagicMock()
    # Simulate: first RPUSH succeeds, second returns a per-command error
    mock_pipeline.execute = AsyncMock(return_value=[3, Exception("WRONGTYPE")])
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    from litellm.types.caching import RedisPipelineRpushOperation

    rpush_list = [
        RedisPipelineRpushOperation(key="key1", values=["a"]),
        RedisPipelineRpushOperation(key="key2", values=["b"]),
    ]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        with pytest.raises(Exception, match="WRONGTYPE"):
            await redis_cache.async_rpush_pipeline(rpush_list=rpush_list)


@pytest.mark.asyncio
async def test_async_lpop_pipeline_raises_on_per_command_error(
    monkeypatch, redis_no_ping
):
    """Verify that per-command errors in LPOP pipeline results are raised, not silently dropped"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    redis_cache.redis_version = "7.0.0"

    mock_redis_instance = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.lpop = MagicMock()
    # Simulate: first LPOP succeeds, second returns a per-command error
    mock_pipeline.execute = AsyncMock(return_value=[[b"val1"], Exception("WRONGTYPE")])
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    from litellm.types.caching import RedisPipelineLpopOperation

    lpop_list = [
        RedisPipelineLpopOperation(key="key1", count=10),
        RedisPipelineLpopOperation(key="key2", count=10),
    ]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        with pytest.raises(Exception, match="WRONGTYPE"):
            await redis_cache.async_lpop_pipeline(lpop_list=lpop_list)


@pytest.mark.asyncio
async def test_async_lpop_pipeline_empty_list(monkeypatch, redis_no_ping):
    """Empty lpop_list should return empty list without touching Redis"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        result = await redis_cache.async_lpop_pipeline(lpop_list=[])

    assert result == []
    mock_redis_instance.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_async_lpop_pipeline_propagates_redis_exception(
    monkeypatch, redis_no_ping
):
    """Pipeline errors should propagate"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    redis_cache.redis_version = "7.0.0"

    mock_redis_instance = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.lpop = MagicMock()
    mock_pipeline.execute = AsyncMock(side_effect=ConnectionError("Redis down"))
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    from litellm.types.caching import RedisPipelineLpopOperation

    lpop_list = [RedisPipelineLpopOperation(key="key1", count=10)]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        with pytest.raises(ConnectionError, match="Redis down"):
            await redis_cache.async_lpop_pipeline(lpop_list=lpop_list)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redis_version",
    [
        # Standard cases
        "7.0.0",  # Standard Redis string version
        7.0,  # Valkey/ElastiCache float version (THE BUG this fix addresses)
        7,  # Integer version (e.g., from some Redis forks)
        # Version < 7
        "6",  # String without dots, version < 7
        # Malformed versions (fallback to 7)
        "latest",  # Non-numeric version
        "",  # Empty string
        -7.0,  # Negative float
        # Format variations
        " 7.0.0 ",  # Whitespace (should be stripped)
        "7.0.0-rc1",  # Version with suffix
        "10.0.0",  # Double digit major version
    ],
)
async def test_async_lpop_with_float_redis_version(
    monkeypatch, redis_no_ping, redis_version
):
    """
    Test async_lpop with various Redis version formats (especially float).

    This test specifically addresses the issue where AWS ElastiCache Valkey
    returns redis_version as a float (e.g., 7.0) instead of a string (e.g., "7.0.0"),
    which caused a 'float' object has no attribute 'split' error when trying to
    use the Redis transaction buffer feature.

    The fix converts the version to a string and handles edge cases like:
    - Floats (7.0) and integers (7)
    - Strings with/without dots ("7" vs "7.0.0")
    - Malformed versions ("v7.0.0", "latest") - fallback to version 7
    - Whitespace (" 7.0.0 ")
    - Negative versions (fallback to version 7)

    Related: Database deadlock issues when use_redis_transaction_buffer is enabled.
    """
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")

    # Create RedisCache instance
    redis_cache = RedisCache()
    redis_cache.redis_version = redis_version  # Set the version to test

    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    # Mock lpop to return a test value (Redis >= 7.0 behavior)
    mock_redis_instance.lpop.return_value = [b"value1", b"value2"]

    # Mock pipeline for Redis < 7.0 (used when major_version < 7)
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    # Make pipeline() a regular method (not async) that returns the mock
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)

    # Mock handle_lpop_count_for_older_redis_versions for Redis < 7
    with patch.object(
        redis_cache,
        "handle_lpop_count_for_older_redis_versions",
        return_value=[b"value1", b"value2"],
    ):
        with patch.object(
            redis_cache, "init_async_client", return_value=mock_redis_instance
        ):
            # Call async_lpop with count - this should not raise AttributeError
            result = await redis_cache.async_lpop(key="test_key", count=2)

            # Verify the method completed without error
            assert result is not None


# LIT-3374: the namespace must be applied uniformly across every key-taking
# Redis operation, not just get/set/increment. Before the fix these paths wrote
# or read raw keys, so with a namespace configured the prefixed keys other
# operations created were silently missed.


@pytest.mark.parametrize(
    "namespace, raw_keys, expected_keys",
    [
        (None, ["{k:v}:tokens", "{k:v}:requests"], ["{k:v}:tokens", "{k:v}:requests"]),
        (
            "litellm_sandbox",
            ["{k:v}:tokens", "{k:v}:requests"],
            ["litellm_sandbox:{k:v}:tokens", "litellm_sandbox:{k:v}:requests"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_register_script_namespaces_keys(
    namespace, raw_keys, expected_keys, monkeypatch, redis_no_ping
):
    """The callable returned by async_register_script (used by the rate limiter
    Lua scripts, pod-lock release, and budget limiters) must namespace every key
    it is invoked with. The hash tag is preserved so cluster slotting is intact."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)

    registered_script = AsyncMock(return_value="ok")
    mock_redis_instance = MagicMock()
    mock_redis_instance.register_script = MagicMock(return_value=registered_script)

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        script = redis_cache.async_register_script("return 1")
        result = await script(keys=raw_keys, args=[60])

    assert result == "ok"
    registered_script.assert_awaited_once_with(
        keys=tuple(expected_keys), args=[60], client=None
    )


# LIT-3298: rate limits tripped at ~40M instead of 80M. async_register_script
# registered the Lua script once at startup and stored the object on the
# limiter, so a request running on a different event loop awaited a script bound
# to the startup loop's connection -> "got Future attached to a different loop".
# The limiter then fell back to a pipeline that reset the window TTL, so two
# minutes of tokens piled into one window. The script must instead be registered
# lazily against the calling loop's client and cached per loop.


@pytest.mark.parametrize("namespace", [None, "litellm_sandbox"])
def test_async_register_script_binds_per_event_loop(namespace, monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)

    clients_built = []

    def make_client():
        client = MagicMock()
        client.register_script = MagicMock(return_value=AsyncMock(return_value="ok"))
        clients_built.append(client)
        return client

    unique_script = "return 'lit3298'"

    with patch.object(redis_cache, "init_async_client", side_effect=make_client):
        script = redis_cache.async_register_script(unique_script)

        # Registration is deferred: no client is touched until the script runs.
        assert clients_built == []

        # Two loops kept alive at once so their ids can't be recycled into one
        # cache key. The buggy version reuses the first loop's bound object.
        loop_a = asyncio.new_event_loop()
        loop_b = asyncio.new_event_loop()
        try:
            result_a = loop_a.run_until_complete(
                script(keys=["{k:v}:tokens"], args=[60])
            )
            result_b = loop_b.run_until_complete(
                script(keys=["{k:v}:tokens"], args=[60])
            )
        finally:
            loop_a.close()
            loop_b.close()

    assert result_a == "ok"
    assert result_b == "ok"
    assert len(clients_built) == 2
    for client in clients_built:
        client.register_script.assert_called_once_with(unique_script)


@pytest.mark.asyncio
async def test_async_register_script_not_shared_across_namespaces(
    monkeypatch, redis_no_ping
):
    """Two caches with different namespaces registering the SAME script must
    each run against their own client and key prefix. A content-only executor
    cache would let the second cache reuse the first's executor and namespace."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    cache_a = RedisCache(namespace="ns_a")
    cache_b = RedisCache(namespace="ns_b")

    reg_a = AsyncMock(return_value="a")
    client_a = MagicMock()
    client_a.register_script = MagicMock(return_value=reg_a)
    reg_b = AsyncMock(return_value="b")
    client_b = MagicMock()
    client_b.register_script = MagicMock(return_value=reg_b)

    same_script = "return redis.call('GET', KEYS[1])"
    with patch.object(
        cache_a, "init_async_client", return_value=client_a
    ), patch.object(cache_b, "init_async_client", return_value=client_b):
        script_a = cache_a.async_register_script(same_script)
        script_b = cache_b.async_register_script(same_script)
        result_a = await script_a(keys=["k"], args=[])
        result_b = await script_b(keys=["k"], args=[])

    assert (result_a, result_b) == ("a", "b")
    reg_a.assert_awaited_once_with(keys=("ns_a:k",), args=[], client=None)
    reg_b.assert_awaited_once_with(keys=("ns_b:k",), args=[], client=None)


@pytest.mark.asyncio
async def test_async_register_script_cluster_path_uses_evalsha(
    monkeypatch, redis_no_ping
):
    """Redis Cluster exposes script_load/evalsha rather than register_script.
    The script is loaded once and invoked via evalsha with namespaced keys."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace="ns")

    cluster_client = MagicMock(spec=["script_load", "evalsha"])
    cluster_client.script_load = MagicMock(return_value="sha123")
    cluster_client.evalsha = AsyncMock(return_value="cluster-ok")

    with patch.object(
        redis_cache, "init_async_client", return_value=cluster_client
    ):
        script = redis_cache.async_register_script("return 'cluster'")
        result = await script(keys=["{k:v}:tokens"], args=[5, 60])

    assert result == "cluster-ok"
    cluster_client.script_load.assert_called_once_with("return 'cluster'")
    cluster_client.evalsha.assert_awaited_once_with(
        "sha123", 1, "ns:{k:v}:tokens", 5, 60
    )


@pytest.mark.asyncio
async def test_async_register_script_raises_for_unsupported_client(
    monkeypatch, redis_no_ping
):
    """A client exposing neither register_script nor script_load fails loudly
    rather than silently returning a no-op callable."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    bad_client = MagicMock(spec=[])

    with patch.object(redis_cache, "init_async_client", return_value=bad_client):
        script = redis_cache.async_register_script("return 'x'")
        with pytest.raises(ValueError, match="does not support Lua script"):
            await script(keys=["k"], args=[1])


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_delete_cache_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_delete_cache("k")
    mock_redis_instance.delete.assert_awaited_once_with(expected)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_delete_cache_keys_namespaces_keys(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.delete_cache_keys(["k"])
    mock_redis_instance.delete.assert_awaited_once_with(expected)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_get_ttl_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.ttl = AsyncMock(return_value=42)
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        ttl = await redis_cache.async_get_ttl("k")
    assert ttl == 42
    mock_redis_instance.ttl.assert_awaited_once_with(expected)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_lpop_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.lpop = AsyncMock(return_value=b"value")
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_lpop(key="k")
    mock_redis_instance.lpop.assert_awaited_once_with(expected, None)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_rpush_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.rpush = AsyncMock(return_value=1)
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_rpush("k", ["v"])
    mock_redis_instance.rpush.assert_awaited_once_with(expected, "v")


@pytest.mark.parametrize("namespace, expected_match", [(None, "k*"), ("ns", "ns:k*")])
@pytest.mark.asyncio
async def test_async_scan_iter_namespaces_pattern(
    namespace, expected_match, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)

    captured = {}

    def scan_iter(match, count):
        captured["match"] = match

        async def gen():
            for _ in ():
                yield _

        return gen()

    mock_redis_instance = MagicMock()
    mock_redis_instance.scan_iter = scan_iter
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_scan_iter(pattern="k")
    assert captured["match"] == expected_match


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
def test_increment_cache_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_client = MagicMock()
    mock_client.incr.return_value = 5
    mock_client.ttl.return_value = 100
    redis_cache.redis_client = mock_client
    redis_cache.increment_cache(key="k", value=1)
    mock_client.incr.assert_called_once_with(name=expected, amount=1)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
def test_delete_cache_namespaces_key(namespace, expected, monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_client = MagicMock()
    redis_cache.redis_client = mock_client
    redis_cache.delete_cache(key="k")
    mock_client.delete.assert_called_once_with(expected)
