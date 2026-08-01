import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.redis_cache import RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache


@patch("litellm._redis.init_redis_cluster")
def test_redis_cluster_batch_get(mock_init_redis_cluster):
    """
    Test that RedisClusterCache uses mget_nonatomic instead of mget for batch operations
    """
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_redis.mget_nonatomic.return_value = [None, None]  # Simulate no cache hits
    mock_init_redis_cluster.return_value = mock_redis

    # Create RedisClusterCache instance with mock client
    cache = RedisClusterCache(
        startup_nodes=[{"host": "localhost", "port": 6379}],
        password="hello",
    )

    # Test batch_get_cache
    keys = ["key1", "key2"]
    cache.batch_get_cache(keys)

    # Verify mget_nonatomic was called instead of mget
    mock_redis.mget_nonatomic.assert_called_once()
    assert not mock_redis.mget.called


@pytest.mark.asyncio
@patch("litellm._redis.init_redis_cluster")
async def test_redis_cluster_async_batch_get(mock_init_redis_cluster):
    """
    Test that RedisClusterCache uses mget_nonatomic instead of mget for async batch operations
    """
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_redis.mget_nonatomic.return_value = [None, None]  # Simulate no cache hits

    # Create RedisClusterCache instance with mock client
    cache = RedisClusterCache(
        startup_nodes=[{"host": "localhost", "port": 6379}],
        password="hello",
    )

    # Mock the init_async_client to return our mock redis client
    cache.init_async_client = MagicMock(return_value=mock_redis)

    # Test async_batch_get_cache
    keys = ["key1", "key2"]
    await cache.async_batch_get_cache(keys)

    # Verify mget_nonatomic was called instead of mget
    mock_redis.mget_nonatomic.assert_called_once()
    assert not mock_redis.mget.called


@patch("litellm._redis.get_redis_connection_pool")
@patch("litellm._redis.get_redis_client")
@patch("litellm.caching.redis_cache.RedisCache._setup_health_pings")
def test_cache_init_creates_cluster_cache_from_env_var(
    mock_health, mock_get_client, mock_get_pool, monkeypatch
):
    """
    Test that Cache() creates RedisClusterCache when REDIS_CLUSTER_NODES env var is set.

    Regression test for https://github.com/BerriAI/litellm/issues/22748
    """
    from litellm.caching.caching import Cache

    startup_nodes = [{"host": "127.0.0.1", "port": "7001"}]
    monkeypatch.setenv("REDIS_CLUSTER_NODES", json.dumps(startup_nodes))
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    mock_get_client.return_value = MagicMock()
    mock_get_pool.return_value = MagicMock()

    cache = Cache(type="redis")
    assert isinstance(cache.cache, RedisClusterCache)


@patch("litellm._redis.get_redis_connection_pool")
@patch("litellm._redis.get_redis_client")
@patch("litellm.caching.redis_cache.RedisCache._setup_health_pings")
def test_cache_init_creates_redis_cache_without_cluster_config(
    mock_health, mock_get_client, mock_get_pool, monkeypatch
):
    """
    Test that Cache() creates RedisCache when no cluster config is present.

    Ensures backward compatibility: without REDIS_CLUSTER_NODES or
    redis_startup_nodes, the standard RedisCache is still used.
    """
    from litellm.caching.caching import Cache

    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.delenv("REDIS_URL", raising=False)

    mock_get_client.return_value = MagicMock()
    mock_get_pool.return_value = MagicMock()

    cache = Cache(type="redis")
    assert isinstance(cache.cache, RedisCache)
    assert not isinstance(cache.cache, RedisClusterCache)


@pytest.mark.asyncio
@patch("litellm._redis.get_redis_connection_pool", return_value=None)
@patch("litellm._redis.get_redis_client")
@patch("litellm.caching.redis_cache.RedisCache._setup_health_pings")
async def test_redis_cluster_disconnect_handles_missing_connection_pool(
    _mock_health, mock_get_client, mock_get_pool
):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/31206.

    Redis cluster mode does not use async_redis_conn_pool, so shutdown should
    skip pool disconnect instead of raising AttributeError on None.
    """
    mock_sync_client = MagicMock()
    mock_get_client.return_value = mock_sync_client

    cache = RedisClusterCache(startup_nodes=[{"host": "localhost", "port": 6379}])

    await cache.disconnect()

    mock_get_pool.assert_called_once()
    mock_sync_client.close.assert_called_once()


@pytest.mark.asyncio
@patch("litellm._redis.get_redis_connection_pool", return_value=None)
@patch("litellm._redis.get_redis_client")
@patch("litellm.caching.redis_cache.RedisCache._setup_health_pings")
async def test_redis_cluster_disconnect_closes_async_client_when_initialized(
    _mock_health, mock_get_client, _mock_get_pool
):
    """Redis cluster shutdown should close the async client if one exists."""
    mock_sync_client = MagicMock()
    mock_get_client.return_value = mock_sync_client

    cache = RedisClusterCache(startup_nodes=[{"host": "localhost", "port": 6379}])
    mock_async_client = AsyncMock()
    cache.redis_async_client = mock_async_client

    await cache.disconnect()

    mock_async_client.aclose.assert_awaited_once()
    mock_sync_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_redis_disconnect_still_errors_when_non_cluster_pool_missing():
    """Do not silently mask a missing pool for non-cluster Redis configs."""
    cache = object.__new__(RedisCache)
    cache.async_redis_conn_pool = None
    cache.redis_kwargs = {}
    cache.redis_client = MagicMock()
    cache.redis_async_client = None

    with pytest.raises(RuntimeError, match="Redis connection pool is not initialized"):
        await cache.disconnect()


@pytest.mark.parametrize(
    "startup_nodes, env_var, expected_cache_type",
    [
        pytest.param(
            [dict(host="node1.localhost", port=6379)],
            None,
            RedisClusterCache,
            id="cluster-via-explicit-startup-nodes",
        ),
        pytest.param(
            None,
            '[{"host": "node1.localhost", "port": 6379}]',
            RedisClusterCache,
            id="cluster-via-env-var",
        ),
        pytest.param(
            None,
            None,
            RedisCache,
            id="standard-redis-when-no-cluster-config",
        ),
        pytest.param(
            [dict(host="explicit-node.localhost", port=6379)],
            '[{"host": "env-node.localhost", "port": 6379}]',
            RedisClusterCache,
            id="explicit-startup-nodes-takes-precedence-over-env-var",
        ),
    ],
)
def test_router_create_redis_cache_cluster_detection(
    startup_nodes, env_var, expected_cache_type, monkeypatch
):
    """
    Test that Router._create_redis_cache() creates RedisClusterCache when
    either startup_nodes is in config or REDIS_CLUSTER_NODES env var is set.
    Also verifies that explicit startup_nodes take precedence over env var.

    Regression test for https://github.com/BerriAI/litellm/issues/22748
    """
    from litellm import Router

    cache_config = dict(
        host="mockhost",
        port=6379,
        password="mock-password",
        startup_nodes=startup_nodes,
    )

    if env_var is not None:
        monkeypatch.setenv("REDIS_CLUSTER_NODES", env_var)
    else:
        monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    def _mock_redis_cache_init(*args, **kwargs): ...

    with patch.object(RedisCache, "__init__", _mock_redis_cache_init):
        redis_cache = Router._create_redis_cache(cache_config)
        assert isinstance(redis_cache, expected_cache_type)
