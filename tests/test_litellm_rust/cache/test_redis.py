import json
import os
import time
from types import SimpleNamespace
from typing import Final
from urllib.parse import urlparse

import pytest
import redis

import litellm
from litellm.caching.caching import Cache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import (
    CacheTestHandle,
    CacheTestResolver,
    assert_native_runtime,
    completion_kwargs,
    request,
    require_rust,
)
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


@pytest.fixture
def cluster_nodes() -> tuple[tuple[str, int], ...]:
    configured: Final = os.environ.get("LITELLM_TEST_REDIS_CLUSTER_NODES")
    if not configured:
        pytest.skip("LITELLM_TEST_REDIS_CLUSTER_NODES is not set")
    return tuple((host, int(port)) for host, _, port in (node.partition(":") for node in configured.split(",")))


async def test_redis_reads_python_sync_and_async_entries_and_writes_without_hidden_prefix(redis_url: str) -> None:
    client: Final = redis.Redis.from_url(redis_url)
    namespace: Final = SimpleNamespace(cache=CacheTestHandle.redis(redis_url, namespace="team"))
    binding: Final = CacheTestResolver(namespace).resolve()
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}, "flag": True, "empty": None}
    envelope: Final = {"timestamp": time.time(), "response": json.dumps(response)}
    client.set("team:sync", str(envelope))
    client.set("team:async", json.dumps({"timestamp": time.time(), "response": response}))
    client.set("team:raw", json.dumps(response))
    client.set("team:invalid", "not a cache entry")
    assert binding.lookup(request("sync")) == response
    assert await binding.async_lookup(request("team:async")) == response
    assert binding.lookup(request("raw")) == response
    assert await binding.async_lookup(request("invalid")) is None
    await binding.async_store({**request("native"), "ttl_seconds": 12.0}, response)
    stored: Final = client.get("team:native")
    assert isinstance(stored, bytes)
    assert json.loads(stored)["response"] == response
    assert 0 < client.ttl("team:native") <= 12
    assert client.get("litellm-cache:team:native") is None
    assert client.get("team:team:async") is None
    client.close()


async def test_redis_facade_buffers_native_async_writes(redis_url: str) -> None:
    parsed: Final = urlparse(redis_url)
    with rebound(litellm, "default_redis_ttl", 60):
        facade: Final = Cache(
            type=LiteLLMCacheType.REDIS,
            host=parsed.hostname,
            port=str(parsed.port),
            redis_flush_size=2,
        )
        with pytest.raises(TypeError, match="default TTLs must match"):
            CacheTestHandle.redis(redis_url, ttl_seconds=61)._bind_facade(facade)
        with pytest.raises(TypeError, match="namespaces must match"):
            CacheTestHandle.redis(redis_url, namespace="other")._bind_facade(facade)
        CacheTestHandle.redis(redis_url, ttl_seconds=60)._bind_facade(facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(redis_url)

    with rebound(facade.cache, "redis_kwargs", {**facade.cache.redis_kwargs, "ssl": True}):
        assert CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"

    pool: Final = facade.cache.redis_client.connection_pool
    with rebound(pool, "connection_kwargs", {**pool.connection_kwargs, "db": 1}):
        assert CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"

    await binding.async_store(request("first"), {"value": 1})
    assert client.get("first") is None
    await binding.async_store(request("second"), {"value": 2})

    assert client.get("first") is not None
    assert client.get("second") is not None
    await facade.cache.disconnect()
    client.close()


async def test_redis_cluster_facade_serves_multi_slot_batches_and_scoped_flush_natively(
    cluster_nodes: tuple[tuple[str, int], ...],
) -> None:
    startup_nodes: Final = [{"host": host, "port": port} for host, port in cluster_nodes]
    url: Final = f"redis://{cluster_nodes[0][0]}:{cluster_nodes[0][1]}"
    with rebound(litellm, "default_redis_ttl", 60):
        facade: Final = Cache(type=LiteLLMCacheType.REDIS, redis_startup_nodes=startup_nodes, namespace="parity")
        assert type(facade.cache) is RedisClusterCache
        with pytest.raises(TypeError, match="types must match"):
            CacheTestHandle.redis(url, namespace="parity")._bind_facade(facade)
        CacheTestHandle.redis(url, namespace="parity", startup_nodes=list(cluster_nodes))._bind_facade(facade)
    resolver: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    assert resolver.resolve().kind == "native"

    manager: Final = facade.cache.redis_client.nodes_manager
    with rebound(manager, "connection_kwargs", {**manager.connection_kwargs, "db": 1}):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "redis_kwargs", {**facade.cache.redis_kwargs, "startup_nodes": startup_nodes[:1]}):
        assert resolver.resolve().kind == "python_callback"
    binding: Final = resolver.resolve()
    assert binding.kind == "native"

    client: Final = redis.RedisCluster(startup_nodes=[redis.cluster.ClusterNode(*node) for node in cluster_nodes])
    keys: Final = tuple(f"slot-{index}" for index in range(12))
    slots: Final = {client.keyslot(f"parity:{key}") for key in keys}
    assert len(slots) > 1, slots
    requests: Final = [request(key) for key in keys]
    values: Final = [{"index": index} for index in range(len(keys))]
    await binding.async_store_batch(requests, values)
    client.set("parity:slot-3", "not a cache entry")
    client.set("parity:slot-7", json.dumps({"timestamp": time.time(), "response": {"index": 7, "python": True}}))

    batch: Final = await binding.async_lookup_batch(requests)
    assert batch == {
        "values": [
            None if index == 3 else {"index": 7, "python": True} if index == 7 else value
            for index, value in enumerate(values)
        ],
        "missing_indices": [3],
    }
    assert facade.cache.get_cache("parity:slot-0")["response"] == {"index": 0}
    assert (await facade.cache.async_get_cache("parity:slot-11"))["response"] == {"index": 11}
    assert facade.cache.redis_client.mget_nonatomic([f"parity:{key}" for key in keys[:2]]) == [
        client.get("parity:slot-0"),
        client.get("parity:slot-1"),
    ]

    await binding.async_store({**request("pinned"), "ttl_seconds": 12.0}, {"pinned": True})
    assert 0 < client.ttl("parity:pinned") <= 12
    client.set("unscoped", "stays")

    await binding.async_flush()

    remaining: Final = tuple(
        sorted(key for node in client.get_primaries() for key in client.keys("parity:*", target_nodes=node))
    )
    assert remaining == (), remaining
    assert client.get("unscoped") == b"stays"
    client.delete("unscoped")
    client.close()
    facade.cache.redis_client.close()


def redis_facade(redis_url: str, **settings: object) -> Cache:
    parsed: Final = urlparse(redis_url)
    return Cache(type=LiteLLMCacheType.REDIS, host=parsed.hostname, port=str(parsed.port), **settings)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        pytest.param({"max_connections": 10}, "max_connections requires Python", id="pool-size"),
        pytest.param({"socket_timeout": 1.0}, "socket_timeout and socket_connect_timeout", id="socket-timeout"),
        pytest.param(
            {"socket_connect_timeout": 1.0}, "socket_timeout and socket_connect_timeout", id="connect-timeout"
        ),
        pytest.param({"socket_keepalive": True}, "does not support socket_keepalive", id="keepalive"),
        pytest.param({"health_check_interval": 5}, "does not support health_check_interval", id="health-check"),
        pytest.param({"client_name": "litellm"}, "does not support client_name", id="client-name"),
        pytest.param({"ssl": True}, "ssl_check_hostname=false require Python", id="tls-default-hostname-check"),
        pytest.param({"ssl": True, "ssl_cert_reqs": "none"}, "ssl_cert_reqs=none", id="tls-without-verification"),
        pytest.param(
            {"ssl": True, "ssl_check_hostname": True, "ssl_ca_certs": "/ca.pem"},
            "does not support ssl_ca_certs",
            id="tls-custom-ca",
        ),
        pytest.param(
            {"ssl": True, "ssl_check_hostname": True, "ssl_certfile": "/client.pem", "ssl_keyfile": "/client.key"},
            "does not support ssl_ca_certs, ssl_ca_data, ssl_certfile or ssl_keyfile",
            id="tls-client-certificate",
        ),
    ],
)
def test_redis_settings_the_native_client_cannot_honor_decline(
    redis_url: str, monkeypatch: pytest.MonkeyPatch, settings: dict[str, object], message: str
) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.REDIS)
    with pytest.raises(RuntimeError, match=f"declined the cache: native Redis.*{message}"):
        redis_facade(redis_url, **settings)


def test_redis_verified_tls_activates_natively(redis_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.REDIS)
    assert_native_runtime(redis_facade(redis_url, ssl=True, ssl_check_hostname=True))


async def test_redis_flush_size_buffers_native_facade_writes(redis_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.REDIS)
    facade: Final = redis_facade(redis_url, redis_flush_size=2, namespace="team")
    assert_native_runtime(facade)
    client: Final = redis.Redis.from_url(redis_url)
    first: Final = completion_kwargs("first")
    await facade.async_add_cache({"value": 1}, **first)
    first_key: Final = facade.get_cache_key(**first)
    assert first_key.startswith("team:")
    assert client.get(first_key) is None
    second: Final = completion_kwargs("second")
    await facade.async_add_cache({"value": 2}, **second)
    assert client.get(first_key) is not None
    assert client.get(facade.get_cache_key(**second)) is not None
    client.close()


def test_rust_with_fallback_keeps_python_when_the_native_client_declines(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        catalog,
        "RULES",
        (CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.REDIS})),),
    )
    assert redis_facade(redis_url, socket_timeout=1.0)._native_cache is None  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor
