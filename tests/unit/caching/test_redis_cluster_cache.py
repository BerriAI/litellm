import asyncio
import gc
import json
import ssl
import weakref
from collections.abc import Awaitable, Callable
from importlib import import_module
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis, RedisCluster
from redis.asyncio.cluster import ClusterNode
from redis.asyncio.connection import SSLConnection
from redis.crc import key_slot

from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.caching.redis_cache import RedisCache
from litellm.caching.redis_cluster_cache import ClusterConnectionGate, RedisClusterCache


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


@pytest.mark.parametrize(
    ("namespace", "expected_source_key", "expected_processing_prefix"),
    [
        (None, "buffer", "{buffer}:"),
        ("tenant", "tenant:buffer", "tenant:{tenant:buffer}:"),
    ],
)
def test_redis_cluster_processing_list_key_preserves_namespace_and_hash_slot(
    namespace: str | None,
    expected_source_key: str,
    expected_processing_prefix: str,
) -> None:
    with patch("litellm._redis.init_redis_cluster", return_value=MagicMock()):
        cache: Final = RedisClusterCache(
            startup_nodes=[{"host": "localhost", "port": 6379}],
            namespace=namespace,
        )

    source_namespaced_key: Final = cache.check_and_fix_namespace("buffer")
    processing_key: Final = cache._processing_list_key(source_namespaced_key, "worker-1")
    assert source_namespaced_key == expected_source_key
    assert processing_key.startswith(expected_processing_prefix)
    assert key_slot(processing_key.encode()) == key_slot(source_namespaced_key.encode())


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
def test_cache_init_creates_cluster_cache_from_env_var(mock_health, mock_get_client, mock_get_pool, monkeypatch):
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
def test_router_create_redis_cache_cluster_detection(startup_nodes, env_var, expected_cache_type, monkeypatch):
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


def test_init_async_client_returns_one_cluster_client_per_event_loop() -> None:
    clients = LLMClientCache()
    with (
        patch("litellm.in_memory_llm_clients_cache", clients),
        patch("redis.RedisCluster", return_value=MagicMock()),
        patch("redis.Redis", return_value=MagicMock()),
    ):
        cache = RedisClusterCache(startup_nodes=[{"host": "127.0.0.1", "port": 7000}])
        loop_a = asyncio.new_event_loop()
        loop_b = asyncio.new_event_loop()

        async def get_client() -> RedisCluster:
            return cache.init_async_client()

        async def get_client_pair() -> tuple[RedisCluster, RedisCluster]:
            return cache.init_async_client(), cache.init_async_client()

        try:
            client_a, client_a_again = loop_a.run_until_complete(get_client_pair())
            client_b, client_b_again = loop_b.run_until_complete(get_client_pair())

            assert isinstance(client_a, RedisCluster)
            assert isinstance(client_b, RedisCluster)
            assert client_a is client_a_again
            assert client_b is client_b_again
            assert client_a is not client_b

            loop_a.close()
            assert loop_b.run_until_complete(get_client()) is client_b
        finally:
            loop_a.close()
            loop_b.close()


@pytest.mark.parametrize("cluster", [True, False])
def test_evicted_async_client_is_closed_only_for_cluster(cluster: bool) -> None:
    clients = LLMClientCache(evicted_client_closer=EvictedClientCloser(grace_seconds=0))
    with (
        patch("litellm.in_memory_llm_clients_cache", clients),
        patch("redis.RedisCluster", return_value=MagicMock()),
        patch("redis.Redis", return_value=MagicMock()),
    ):
        cache = (
            RedisClusterCache(startup_nodes=[{"host": "127.0.0.1", "port": 7000}])
            if cluster
            else RedisCache(host="standalone-evicted-client", port=6379)
        )

        async def exercise() -> None:
            client = cache.init_async_client()
            closed = asyncio.Event()
            client.aclose = AsyncMock(side_effect=closed.set)
            cache_key = clients.update_cache_key_with_event_loop(cache._get_async_client_cache_key())
            clients.ttl_dict[cache_key] = 0

            replacement = cache.init_async_client()

            assert replacement is not client
            if cluster:
                await asyncio.wait_for(closed.wait(), timeout=1)
                client.aclose.assert_awaited_once()
            else:
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                client.aclose.assert_not_awaited()

        asyncio.run(exercise())


class _InFlightCountingClient:
    """Async cluster client double that records how many commands were in flight at once."""

    def __init__(self, command_seconds: float) -> None:
        self.command_seconds = command_seconds
        self.in_flight = 0
        self.peak_in_flight = 0
        self.completed = 0

    async def set(self, *args: object, **kwargs: object) -> bool:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(self.command_seconds)
        self.in_flight -= 1
        self.completed += 1
        return True


def _cluster_set_cache_with_client(
    client: _InFlightCountingClient,
    max_connections: int | None = None,
    connection_gate: ClusterConnectionGate | None = None,
) -> Callable[[str, str], Awaitable[object]]:
    with patch("litellm._redis.init_redis_cluster"):
        cache = RedisClusterCache(
            startup_nodes=[{"host": "localhost", "port": 7000}],
            max_connections=max_connections,
            connection_gate=connection_gate,
        )
    setattr(cache, "init_async_client", MagicMock(return_value=client))
    return cast(
        "Callable[[str, str], Awaitable[object]]", cache.async_set_cache
    )  # cast-ok: wrapped **kwargs are untyped


@pytest.mark.asyncio
async def test_cluster_cache_queues_commands_beyond_max_connections_instead_of_failing() -> None:
    """A burst larger than max_connections used to hand every command to the node at once, so redis-py raised
    MaxConnectionsError on the overflow and the circuit breaker opened. The overflow now waits for a slot."""
    client = _InFlightCountingClient(command_seconds=0.01)
    set_cache = _cluster_set_cache_with_client(client, max_connections=3)

    await asyncio.gather(*(set_cache(f"key-{i}", "value") for i in range(20)))

    assert client.completed == 20
    assert client.peak_in_flight <= 3


@pytest.mark.asyncio
async def test_cluster_cache_without_max_connections_does_not_throttle() -> None:
    client = _InFlightCountingClient(command_seconds=0.01)
    set_cache = _cluster_set_cache_with_client(client)

    await asyncio.gather(*(set_cache(f"key-{i}", "value") for i in range(20)))

    assert client.completed == 20
    assert client.peak_in_flight == 20


@pytest.mark.asyncio
async def test_cluster_cache_sends_after_the_connection_wait_timeout() -> None:
    """A command that cannot get a slot within the pool timeout is still sent, so a saturated pool keeps
    surfacing through the client's own errors and the breaker rather than stalling forever."""
    client = _InFlightCountingClient(command_seconds=0.2)
    set_cache = _cluster_set_cache_with_client(
        client, connection_gate=ClusterConnectionGate(limit=1, wait_timeout=0.02)
    )

    await asyncio.wait_for(asyncio.gather(*(set_cache(f"key-{i}", "value") for i in range(3))), timeout=1.0)

    assert client.completed == 3
    assert client.peak_in_flight == 3


@pytest.mark.asyncio
async def test_cluster_caches_sharing_a_client_share_the_connection_cap() -> None:
    """init_async_client hands caches with identical settings the same client, so two caches must not each admit
    a full max_connections worth of commands against that one pool."""
    client = _InFlightCountingClient(command_seconds=0.01)
    first = _cluster_set_cache_with_client(client, max_connections=3)
    second = _cluster_set_cache_with_client(client, max_connections=3)

    await asyncio.gather(
        *(first(f"first-{i}", "value") for i in range(10)), *(second(f"second-{i}", "value") for i in range(10))
    )

    assert client.completed == 20
    assert client.peak_in_flight <= 3


@pytest.mark.asyncio
async def test_guarded_command_nested_in_a_guarded_command_reuses_the_slot() -> None:
    """batch_cache_write flushes through async_set_cache_pipeline while it is itself admitted. With one slot the
    inner command must run on the outer slot instead of waiting for the pool timeout."""
    gate = ClusterConnectionGate(limit=1, wait_timeout=5.0)

    async def inner() -> str:
        return "flushed"

    async def outer() -> str:
        return await gate.run(inner, "client-a")

    assert await asyncio.wait_for(gate.run(outer, "client-a"), timeout=1.0) == "flushed"


@pytest.mark.asyncio
async def test_nested_guarded_command_on_another_client_waits_for_that_clients_slot() -> None:
    gate = ClusterConnectionGate(limit=1, wait_timeout=5.0)
    client_b_busy = asyncio.Event()
    release_b = asyncio.Event()
    peak_b = 0
    in_flight_b = 0

    async def hold_b() -> None:
        nonlocal peak_b, in_flight_b
        in_flight_b += 1
        peak_b = max(peak_b, in_flight_b)
        client_b_busy.set()
        await release_b.wait()
        in_flight_b -= 1

    async def touch_b() -> None:
        nonlocal peak_b, in_flight_b
        in_flight_b += 1
        peak_b = max(peak_b, in_flight_b)
        await asyncio.sleep(0)
        in_flight_b -= 1

    async def outer_on_a() -> None:
        await client_b_busy.wait()
        await gate.run(touch_b, "client-b")

    holder = asyncio.create_task(gate.run(hold_b, "client-b"))
    nested = asyncio.create_task(gate.run(outer_on_a, "client-a"))
    await client_b_busy.wait()
    await asyncio.sleep(0.05)
    release_b.set()
    await asyncio.gather(holder, nested)

    assert peak_b == 1


@pytest.mark.asyncio
async def test_child_task_spawned_inside_a_guarded_command_does_not_inherit_the_slot() -> None:
    gate = ClusterConnectionGate(limit=1, wait_timeout=5.0)
    peak = 0
    in_flight = 0

    async def command() -> None:
        nonlocal peak, in_flight
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1

    spawned: list[asyncio.Task[None]] = []

    async def parent() -> None:
        spawned.append(asyncio.create_task(gate.run(command, "client-a")))
        await command()

    await gate.run(parent, "client-a")
    await asyncio.gather(*spawned)

    assert peak == 1


def test_closed_event_loop_is_collected_after_contended_gate_use() -> None:
    gate = ClusterConnectionGate(limit=1, wait_timeout=5.0)

    async def command() -> None:
        await asyncio.sleep(0.01)

    async def contend() -> "weakref.ref[asyncio.AbstractEventLoop]":
        await asyncio.gather(gate.run(command, "client-a"), gate.run(command, "client-a"))
        return weakref.ref(asyncio.get_running_loop())

    loop_ref = asyncio.run(contend())
    gc.collect()

    assert loop_ref() is None
