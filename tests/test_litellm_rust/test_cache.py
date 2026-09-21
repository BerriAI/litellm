import asyncio
import contextvars
import gc
import json
import os
import threading
import time
import weakref
from collections.abc import Generator
from types import SimpleNamespace
from typing import Final, Protocol, cast
from urllib.parse import urlparse

import fakeredis
import pytest
import redis

import litellm
from litellm.caching.caching import Cache, disable_cache, enable_cache, update_cache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


class CacheLookup(Protocol):
    def get_cache(self, **kwargs: object) -> object: ...


def request(key: str = "key") -> dict[str, object]:
    return {"key": {"preset": key}}


@pytest.fixture
def redis_url() -> Generator[str]:
    server: Final = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"redis://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.fixture
def cluster_nodes() -> tuple[tuple[str, int], ...]:
    configured: Final = os.environ.get("LITELLM_TEST_REDIS_CLUSTER_NODES")
    if not configured:
        pytest.skip("LITELLM_TEST_REDIS_CLUSTER_NODES is not set")
    return tuple((host, int(port)) for host, _, port in (node.partition(":") for node in configured.split(",")))


def test_existing_constructor_and_global_are_unchanged() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    assert type(facade.cache) is InMemoryCache
    assert "_native_cache_handle" not in vars(facade)
    with rebound(litellm, "cache", facade):
        resolver: Final = _native._CacheTestResolver(litellm)
        assert resolver.resolve().kind == "python_callback"
        resolver.resolve().store(None, {"answer": 7}, callback_kwargs={"cache_key": "key"})
        assert cast(CacheLookup, facade).get_cache(cache_key="key") == {"answer": 7}


def test_existing_global_lifecycle_remains_the_resolver_source_of_truth() -> None:
    resolver: Final = _native._CacheTestResolver(litellm)

    enable_cache(type=LiteLLMCacheType.LOCAL, ttl=30)
    enabled: Final = litellm.cache
    assert isinstance(enabled, Cache)
    assert enabled.ttl == 30
    assert resolver.resolve().kind == "python_callback"

    enable_cache(type=LiteLLMCacheType.LOCAL, ttl=60)
    assert litellm.cache is enabled

    update_cache(type=LiteLLMCacheType.LOCAL, ttl=60)
    updated: Final = litellm.cache
    assert isinstance(updated, Cache)
    assert updated is not enabled
    assert updated.ttl == 60

    disable_cache()
    assert litellm.cache is None
    assert resolver.resolve().kind == "disabled"


async def test_native_bindings_survive_replacement_and_capture_writes_before_dispatch() -> None:
    namespace: Final = SimpleNamespace(cache=_native._CacheTestHandle.memory())
    resolver: Final = _native._CacheTestResolver(namespace)
    selected: Final = resolver.resolve()
    assert selected.kind == "native"
    selected.store(request(), {"answer": 1})
    assert await selected.async_lookup(request()) == {"answer": 1}
    with rebound(namespace, "cache", _native._CacheTestHandle.memory()):
        replacement: Final = resolver.resolve()
        await selected.async_store(request(), {"answer": 2})
        assert replacement.lookup(request()) is None
        assert selected.lookup(request()) == {"answer": 2}
    with rebound(namespace, "cache", None):
        disabled: Final = resolver.resolve()
        assert disabled.kind == "disabled"
        assert disabled.lookup(None) is None
        await disabled.async_store(None, object())
        assert await disabled.async_lookup(None) is None
        assert selected.lookup(request()) == {"answer": 2}


async def test_python_callback_preserves_identity_caller_task_context_and_errors() -> None:
    context: Final = contextvars.ContextVar("cache_context", default="caller")
    caller: Final = asyncio.current_task()
    sentinel: Final = object()
    failure: Final = RuntimeError("callback failed")

    class CustomCache:
        async def async_get_cache(self, *, marker: object) -> object:
            assert marker is sentinel
            assert asyncio.current_task() is caller
            context.set("callback")
            return marker

        async def async_add_cache(self, response: object, *, marker: object) -> None:
            assert response is sentinel
            assert marker is sentinel
            raise failure

    namespace: Final = SimpleNamespace(cache=CustomCache())
    binding: Final = _native._CacheTestResolver(namespace).resolve()
    assert binding.kind == "python_callback"
    assert await binding.async_lookup(None, callback_kwargs={"marker": sentinel}) is sentinel
    assert context.get() == "callback"
    with pytest.raises(RuntimeError) as caught:
        await binding.async_store(None, sentinel, callback_kwargs={"marker": sentinel})
    assert caught.value is failure


async def test_callback_cancellation_stays_in_the_callers_task() -> None:
    entered: Final = asyncio.Event()
    finished: Final = asyncio.Event()

    class CustomCache:
        async def async_get_cache(self) -> None:
            entered.set()
            try:
                await asyncio.Future()
            finally:
                finished.set()

    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=CustomCache())).resolve()

    async def lookup() -> object:
        return await binding.async_lookup(None, callback_kwargs={})

    task: Final = asyncio.create_task(lookup())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


def test_registered_facade_uses_native_and_instance_overrides_fall_back() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    handle: Final = _native._CacheTestHandle.memory()
    handle._bind_facade(facade)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
    native: Final = resolver.resolve()
    assert native.kind == "native"
    native.store(request(), {"source": "native"})
    assert native.lookup(request()) == {"source": "native"}
    assert cast(CacheLookup, facade).get_cache(cache_key="key") is None
    sentinel: Final = object()

    def outer_override(**_kwargs: object) -> object:
        return sentinel

    def backend_override(*_args: object, **_kwargs: object) -> dict[str, str]:
        return {"source": "override"}

    with rebound(facade, "get_cache", outer_override):
        fallback: Final = resolver.resolve()
        assert fallback.kind == "python_callback"
        assert fallback.lookup(None, callback_kwargs={"cache_key": "key"}) is sentinel
    assert resolver.resolve().kind == "python_callback"
    delattr(facade, "get_cache")
    assert resolver.resolve().kind == "native"
    with rebound(facade.cache, "get_cache", backend_override):
        backend_fallback: Final = resolver.resolve()
        assert backend_fallback.kind == "python_callback"
        assert backend_fallback.lookup(None, callback_kwargs={"cache_key": "key"}) == {"source": "override"}


def test_facade_subclasses_backend_replacement_and_configuration_changes_are_not_bypassed() -> None:
    class CustomCache(Cache):
        pass

    handle: Final = _native._CacheTestHandle.memory()
    with pytest.raises(TypeError):
        handle._bind_facade(CustomCache(type=LiteLLMCacheType.LOCAL))
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    handle._bind_facade(facade)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
    with rebound(facade, "cache", InMemoryCache()):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade, "ttl", 12):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade, "semantic_cache_scope", "end_user"):
        assert resolver.resolve().kind == "python_callback"

    def custom_key(**_kwargs: object) -> str:
        return "custom"

    with rebound(facade, "get_cache_key", custom_key):
        assert resolver.resolve().kind == "python_callback"
    assert resolver.resolve().kind == "python_callback"
    delattr(facade, "get_cache_key")
    assert resolver.resolve().kind == "native"


def test_resolver_and_callback_cycles_can_be_collected() -> None:
    class CustomCache:
        pass

    def cyclic_reference() -> weakref.ReferenceType[CustomCache]:
        callback: Final = CustomCache()
        namespace: Final = SimpleNamespace(cache=callback)
        binding: Final = _native._CacheTestResolver(namespace).resolve()
        setattr(callback, "binding", binding)
        return weakref.ref(callback)

    reference: Final = cyclic_reference()
    gc.collect()
    assert reference() is None


async def test_redis_reads_python_sync_and_async_entries_and_writes_without_hidden_prefix(redis_url: str) -> None:
    client: Final = redis.Redis.from_url(redis_url)
    namespace: Final = SimpleNamespace(cache=_native._CacheTestHandle.redis(redis_url, namespace="team"))
    binding: Final = _native._CacheTestResolver(namespace).resolve()
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


def test_invalid_duration_and_request_shape_fail_before_storage() -> None:
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=_native._CacheTestHandle.memory())).resolve()
    for seconds in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cache durations must be finite and nonnegative"):
            binding.store({**request(), "ttl_seconds": seconds}, {"answer": 1})
    assert binding.lookup(request()) is None
    with pytest.raises(ValueError, match="cache durations must be finite and nonnegative"):
        _native._CacheTestHandle.memory(ttl_seconds=-1)


async def test_memory_size_policy_is_applied_by_the_native_host() -> None:
    handle: Final = _native._CacheTestHandle.memory(capacity=2, max_entry_bytes=128)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    small: Final = {"answer": "ok"}
    binding.store(request("small"), small)
    assert await binding.async_lookup(request("small")) == small
    await binding.async_store(request("large"), {"answer": "x" * 256})
    assert binding.lookup(request("large")) is None
    assert binding.lookup(request("small")) == small
    disabled: Final = _native._CacheTestResolver(
        SimpleNamespace(cache=_native._CacheTestHandle.memory(capacity=0))
    ).resolve()
    await disabled.async_store(request(), small)
    assert await disabled.async_lookup(request()) is None


async def test_native_batch_lookup_and_store_report_partial_hits() -> None:
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=_native._CacheTestHandle.memory())).resolve()
    requests: Final = [request("hit"), request("miss"), request("disabled")]
    requests[2]["controls"] = {
        "supported_call_type": True,
        "configured": True,
        "native_backend": True,
        "default_on": True,
        "caching": False,
        "no_cache": False,
        "no_store": False,
        "use_cache": False,
    }
    await binding.async_store_batch(requests, [{"value": 1}, {"value": 2}, {"value": 3}])

    partial: Final = await binding.async_lookup_batch(requests)

    assert partial == {
        "values": [{"value": 1}, {"value": 2}, None],
        "missing_indices": [2],
    }


async def test_python_batch_callbacks_use_the_builtin_cache_api() -> None:
    result: Final = object()
    marker: Final = object()

    class CustomCache(Cache):
        def get_cache(self, dynamic_cache_object: object = None, **kwargs: object) -> object:
            return ("sync", kwargs)

        async def async_get_cache(self, dynamic_cache_object: object = None, **kwargs: object) -> object:
            return ("async", kwargs)

        async def async_add_cache_pipeline(
            self, result: object, dynamic_cache_object: object = None, **kwargs: object
        ) -> object:
            return result, kwargs

    binding: Final = _native._CacheTestResolver(
        SimpleNamespace(cache=CustomCache(type=LiteLLMCacheType.LOCAL))
    ).resolve()
    assert binding.kind == "python_callback"
    requests: Final = [request("first"), request("second")]
    kwargs: Final = [{"cache_key": "first"}, {"cache_key": "second"}]

    assert binding.lookup_batch(requests, callback_kwargs=kwargs) == [("sync", kwargs[0]), ("sync", kwargs[1])]
    assert await binding.async_lookup_batch(requests, callback_kwargs=kwargs) == [
        ("async", kwargs[0]),
        ("async", kwargs[1]),
    ]
    with pytest.raises(ValueError, match="equal lengths"):
        binding.lookup_batch(requests, callback_kwargs=kwargs[:1])
    with pytest.raises(TypeError, match="callback_result"):
        await binding.async_store_batch(requests, [1, 2], callback_kwargs={"marker": marker})
    stored: Final = cast(
        tuple[object, dict[str, object]],
        await binding.async_store_batch(requests, [1, 2], callback_result=result, callback_kwargs={"marker": marker}),
    )
    assert stored[0] is result
    assert stored[1] == {"marker": marker}


async def test_unmodified_builtin_cache_callbacks_can_ping_and_flush() -> None:
    async def ping() -> str:
        return "pong"

    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache.cache.set_cache("key", "value")
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=cache)).resolve()
    assert binding.kind == "python_callback"

    setattr(cache.cache, "ping", ping)
    assert await binding.ping() == "pong"
    await binding.async_flush()
    assert cache.cache.get_cache("key") is None


def test_facade_registration_rejects_mismatched_capacity() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    with pytest.raises(TypeError, match="capacities must match"):
        _native._CacheTestHandle.memory(capacity=7)._bind_facade(facade)


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
            _native._CacheTestHandle.redis(redis_url, ttl_seconds=61)._bind_facade(facade)
        with pytest.raises(TypeError, match="namespaces must match"):
            _native._CacheTestHandle.redis(redis_url, namespace="other")._bind_facade(facade)
        _native._CacheTestHandle.redis(redis_url, ttl_seconds=60)._bind_facade(facade)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(redis_url)

    with rebound(facade.cache, "redis_kwargs", {**facade.cache.redis_kwargs, "ssl": True}):
        assert _native._CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"

    pool: Final = facade.cache.redis_client.connection_pool
    with rebound(pool, "connection_kwargs", {**pool.connection_kwargs, "db": 1}):
        assert _native._CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"

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
            _native._CacheTestHandle.redis(url, namespace="parity")._bind_facade(facade)
        _native._CacheTestHandle.redis(url, namespace="parity", startup_nodes=list(cluster_nodes))._bind_facade(facade)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
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

    remaining: Final = tuple(sorted(key for node in client.get_primaries() for key in client.keys("parity:*", target_nodes=node)))
    assert remaining == (), remaining
    assert client.get("unscoped") == b"stays"
    client.delete("unscoped")
    client.close()
    facade.cache.redis_client.close()
