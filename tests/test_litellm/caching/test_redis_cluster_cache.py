import asyncio
from importlib import import_module
import json
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis, RedisCluster
from redis.asyncio.cluster import ClusterNode
from redis.asyncio.connection import SSLConnection


from litellm.caching.redis_cache import RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.caching.llm_caching_handler import LLMClientCache


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


def test_init_pubsub_client_preserves_tls_and_authentication() -> None:
    cache = _isolated_redis_cache("cluster-pubsub-tls")
    cluster = RedisCluster(
        startup_nodes=[ClusterNode(host="redis.example.test", port=7000)],
        ssl=True,
        ssl_cert_reqs="required",
        ssl_check_hostname=True,
        username="pubsub-user",
        password="test-password",
        socket_connect_timeout=3.0,
        socket_keepalive=True,
    )
    cache.init_async_client = MagicMock(return_value=cluster)

    client = cache.init_pubsub_client()
    connection = client.connection_pool.make_connection()

    assert isinstance(connection, SSLConnection)
    assert connection.ssl_context.cert_reqs == ssl.CERT_REQUIRED
    assert connection.ssl_context.check_hostname is True
    assert connection.username == "pubsub-user"
    assert connection.password == "test-password"
    assert connection.socket_connect_timeout == 3.0
    assert connection.socket_keepalive is True


def test_init_pubsub_client_rejects_missing_nodes_and_can_retry() -> None:
    cache = _isolated_redis_cache("cluster-pubsub-no-nodes")
    cluster = _cluster_for_pubsub()
    cluster.nodes_manager.startup_nodes = {}
    cache.init_async_client = MagicMock(return_value=cluster)

    with pytest.raises(ValueError, match="no default node and no startup nodes"):
        cache.init_pubsub_client()

    cluster.nodes_manager.default_node = ClusterNode(host="recovered.example.test", port=7001)
    client = cache.init_pubsub_client()

    assert client.connection_pool.connection_kwargs["host"] == "recovered.example.test"


@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("has_shared_pool", [False, True])
def test_disconnect_closes_derived_pubsub_connections_even_when_pool_close_fails(
    close_fails: bool, has_shared_pool: bool
) -> None:
    cache = _isolated_redis_cache(f"cluster-pubsub-close-{close_fails}-{has_shared_pool}")
    cache.async_redis_conn_pool = AsyncMock() if has_shared_pool else None
    cache.init_async_client = MagicMock(return_value=_cluster_for_pubsub())

    async def exercise() -> None:
        client = cache.init_pubsub_client()
        connection = AsyncMock()
        connection.disconnect.side_effect = ConnectionError("connection close failed") if close_fails else None
        client.connection_pool._available_connections.append(connection)

        await cache.disconnect()

        connection.disconnect.assert_awaited_once()
        if has_shared_pool:
            cache.async_redis_conn_pool.disconnect.assert_awaited_once_with(inuse_connections=True)
        cache.redis_client.close.assert_called_once()

    asyncio.run(exercise())


def test_expired_pubsub_client_closes_connections_after_eviction() -> None:
    cache = _isolated_redis_cache("cluster-pubsub-expired")
    cache.init_async_client = MagicMock(return_value=_cluster_for_pubsub())
    clients = LLMClientCache(evicted_client_closer=EvictedClientCloser(grace_seconds=0))

    async def exercise() -> None:
        client = cache.init_pubsub_client()
        closed = asyncio.Event()
        connection = AsyncMock()
        connection.disconnect.side_effect = closed.set
        client.connection_pool._available_connections.append(connection)
        cache_key = clients.update_cache_key_with_event_loop(f"{cache._get_async_client_cache_key()}-pubsub")
        clients.ttl_dict[cache_key] = 0

        replacement = cache.init_pubsub_client()
        await asyncio.wait_for(closed.wait(), timeout=1)

        assert replacement is not client
        connection.disconnect.assert_awaited_once()

    with patch("litellm.in_memory_llm_clients_cache", clients):
        asyncio.run(exercise())
