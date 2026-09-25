from importlib import import_module
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis, RedisCluster
from redis.asyncio.cluster import ClusterNode


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
@patch.object(import_module("litellm.caching.redis_cache").RedisCache, "_setup_health_pings")
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
@patch.object(import_module("litellm.caching.redis_cache").RedisCache, "_setup_health_pings")
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


def _isolated_redis_cache(host: str) -> RedisCache:
    """RedisCache whose sync client and pool are stubbed out."""
    with (
        patch("litellm._redis.get_redis_client", return_value=MagicMock()),
        patch("litellm._redis.get_redis_connection_pool", return_value=MagicMock()),
    ):
        return RedisCache(host=host, port=6379)


def _cluster_for_pubsub(startup_node_host: str = "10.9.9.9") -> RedisCluster:
    """Uninitialized RedisCluster carrying the connection kwargs a real one would."""
    return RedisCluster(
        startup_nodes=[ClusterNode(host=startup_node_host, port=7000)],
        password="cluster-secret",
        socket_timeout=7.0,
    )


def test_init_pubsub_client_derives_a_node_client_for_cluster_backend() -> None:
    """LIT-8543: a cluster-backed cache must return a pub/sub-capable client.

    The derived client pins a plain Redis connection pool to the cluster's
    default node, inheriting the connection kwargs minus cluster-only keys.
    """
    cache = _isolated_redis_cache("cluster-pubsub-default-node")
    cluster = _cluster_for_pubsub()
    node = ClusterNode(host="10.1.2.3", port=7001)
    cluster.nodes_manager.default_node = node
    cache.init_async_client = MagicMock(return_value=cluster)

    client = cache.init_pubsub_client()

    assert isinstance(client, Redis) and not isinstance(client, RedisCluster)
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["host"] == "10.1.2.3"
    assert kwargs["port"] == 7001
    assert kwargs["password"] == "cluster-secret"
    assert kwargs["socket_timeout"] == 7.0
    assert "response_callbacks" not in kwargs


def test_init_pubsub_client_falls_back_to_first_startup_node() -> None:
    """Before cluster initialization there is no default node; the first
    startup node is a valid pub/sub target."""
    cache = _isolated_redis_cache("cluster-pubsub-startup-fallback")
    cluster = _cluster_for_pubsub(startup_node_host="10.8.8.8")
    cache.init_async_client = MagicMock(return_value=cluster)

    client = cache.init_pubsub_client()

    assert isinstance(client, Redis)
    assert client.connection_pool.connection_kwargs["host"] == "10.8.8.8"


def test_init_pubsub_client_returns_the_same_cached_client_on_repeat_calls() -> None:
    cache = _isolated_redis_cache("cluster-pubsub-caching")
    cluster = _cluster_for_pubsub()
    cluster.nodes_manager.default_node = ClusterNode(host="10.1.2.3", port=7001)
    cache.init_async_client = MagicMock(return_value=cluster)

    first = cache.init_pubsub_client()
    second = cache.init_pubsub_client()

    assert first is second


def test_init_pubsub_client_returns_the_shared_async_client_for_standalone() -> None:
    cache = _isolated_redis_cache("standalone-pubsub")
    standalone = Redis()
    cache.init_async_client = MagicMock(return_value=standalone)

    assert cache.init_pubsub_client() is standalone
