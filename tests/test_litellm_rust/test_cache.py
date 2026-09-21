import asyncio
import contextvars
import gc
import json
import os
import threading
import time
import uuid
import weakref
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Final, Protocol, cast
from urllib.parse import urlparse

import diskcache
import fakeredis
import pytest
import redis
from azure.storage.blob import ContainerClient

import litellm
from litellm.caching.azure_blob_cache import AzureBlobCache
from litellm.caching.caching import Cache, disable_cache, enable_cache, update_cache
from litellm.caching.gcs_cache import GCSCache
from litellm.caching.disk_cache import DiskCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.fake_gcs import FakeGcs
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


class CacheLookup(Protocol):
    def get_cache(self, **kwargs: object) -> object: ...
    def flush_cache(self) -> object: ...


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
def fake_gcs() -> Generator[FakeGcs]:
    server: Final = FakeGcs()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def azure_blob_facade() -> Generator[Cache]:
    account_url: Final = os.environ.get("AZURE_BLOB_CACHE_ACCOUNT_URL")
    if account_url is None:
        pytest.skip(
            "live Azure Blob parity needs AZURE_BLOB_CACHE_ACCOUNT_URL plus DefaultAzureCredential inputs in the environment"
        )
    facade: Final = Cache(
        type=LiteLLMCacheType.AZURE_BLOB,
        azure_account_url=account_url,
        azure_blob_container=f"litellm-parity-{uuid.uuid4().hex[:12]}",
    )
    backend: Final = facade.cache
    assert isinstance(backend, AzureBlobCache)
    try:
        yield facade
    finally:
        backend.container_client.delete_container()
        asyncio.run(backend.disconnect())


def azure_blob_handle(facade: Cache) -> _native._CacheTestHandle:
    backend: Final = facade.cache
    assert isinstance(backend, AzureBlobCache)
    return _native._CacheTestHandle.azure_blob(
        backend.container_client.url.removesuffix(f"/{backend.container_client.container_name}"),
        backend.container_client.container_name,
    )


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


def test_azure_blob_facade_serves_natively_and_python_reads_the_same_blobs(azure_blob_facade: Cache) -> None:
    backend: Final = azure_blob_facade.cache
    assert isinstance(backend, AzureBlobCache)
    handle: Final = azure_blob_handle(azure_blob_facade)
    assert handle.backend == "azure-blob"
    account_url: Final = backend.container_client.url.removesuffix(f"/{backend.container_client.container_name}")
    with pytest.raises(TypeError, match="containers must match"):
        _native._CacheTestHandle.azure_blob(account_url, f"{backend.container_client.container_name}-other")._bind_facade(
            azure_blob_facade
        )
    handle._bind_facade(azure_blob_facade)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=azure_blob_facade))
    native: Final = resolver.resolve()
    assert native.kind == "native"

    response: Final = {"choices": [{"text": "caf\u00e9 \u2603"}], "usage": {"total_tokens": 3}, "flag": True, "empty": None}
    native.store({**request("sync"), "ttl_seconds": 0.001}, response)
    native.store(request("sync"), {"choices": [{"text": "second"}]})
    time.sleep(0.01)
    stored: Final = json.loads(backend.container_client.download_blob("sync").readall())
    assert stored["response"] == response
    assert isinstance(stored["timestamp"], float)
    assert native.lookup(request("sync")) == response
    assert cast(CacheLookup, azure_blob_facade).get_cache(cache_key="sync") == response

    backend.set_cache("python", {"timestamp": time.time(), "response": response})
    backend.set_cache("legacy", "bare legacy value")
    backend.container_client.upload_blob("invalid", b"{not json", overwrite=True)
    assert native.lookup(request("python")) == response
    assert native.lookup(request("legacy")) == cast(CacheLookup, azure_blob_facade).get_cache(cache_key="legacy")
    assert native.lookup_batch([request("python"), request("missing"), request("invalid"), request("sync")]) == {
        "values": [response, None, None, response],
        "missing_indices": [1, 2],
    }

    with rebound(azure_blob_facade, "ttl", 12):
        assert resolver.resolve().kind == "python_callback"
    with rebound(backend, "container_client", ContainerClient.from_container_url(backend.container_client.url)):
        assert resolver.resolve().kind == "python_callback"

    def custom_get(*_args: object, **_kwargs: object) -> None:
        return None

    with rebound(backend, "get_cache", custom_get):
        assert resolver.resolve().kind == "python_callback"
    assert resolver.resolve().kind == "python_callback"
    assert cast(CacheLookup, azure_blob_facade).get_cache(cache_key="sync") == response

    class CustomBlobCache(AzureBlobCache):
        pass

    with rebound(azure_blob_facade, "cache", CustomBlobCache(account_url, backend.container_client.container_name)):
        assert resolver.resolve().kind == "python_callback"
        with pytest.raises(TypeError):
            azure_blob_handle(azure_blob_facade)._bind_facade(azure_blob_facade)


async def test_azure_blob_native_async_writes_overwrite_batch_and_flush_like_python(azure_blob_facade: Cache) -> None:
    backend: Final = azure_blob_facade.cache
    assert isinstance(backend, AzureBlobCache)
    azure_blob_handle(azure_blob_facade)._bind_facade(azure_blob_facade)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=azure_blob_facade)).resolve()
    assert binding.kind == "native"
    ping: Final = cast(dict[str, object], await binding.ping())
    assert ping["status"] == "success", ping

    await binding.async_store(request("async"), {"value": 1})
    await binding.async_store({**request("async"), "ttl_seconds": 0.001}, {"value": 2})
    time.sleep(0.01)
    assert await binding.async_lookup(request("async")) == {"value": 2}
    assert await backend.async_get_cache("async") == json.loads(backend.container_client.download_blob("async").readall())
    assert cast(CacheLookup, azure_blob_facade).get_cache(cache_key="async") == {"value": 2}

    await binding.async_store_batch([request("first"), request("second")], [{"value": 3}, {"value": 4}])
    assert await binding.async_lookup_batch([request("second"), request("missing"), request("first")]) == {
        "values": [{"value": 4}, None, {"value": 3}],
        "missing_indices": [1],
    }
    await binding.async_flush()
    assert [blob.name for blob in backend.container_client.list_blobs()] == []
    assert await binding.async_lookup(request("async")) is None


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
async def test_disk_reads_python_entries_and_python_reads_native_entries(tmp_path: Path) -> None:
    disk_cache: Final = DiskCache(disk_cache_dir=str(tmp_path))
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}}
    disk_cache.disk_cache.set(
        "sync",
        {"timestamp": time.time(), "response": json.dumps(response)},
    )
    disk_cache.disk_cache.set("async", json.dumps({"timestamp": time.time(), "response": response}))
    disk_cache.disk_cache.set("raw", json.dumps(response))
    disk_cache.disk_cache.set("invalid", "not a cache entry")
    disk_cache.disk_cache.set(
        "large",
        {"timestamp": time.time(), "response": {"text": "x" * 70_000}},
    )
    binding: Final = _native._CacheTestResolver(
        SimpleNamespace(cache=_native._CacheTestHandle.disk(str(tmp_path)))
    ).resolve()

    assert binding.lookup(request("sync")) == response
    assert await binding.async_lookup(request("async")) == response
    assert binding.lookup(request("raw")) == response
    assert await binding.async_lookup(request("invalid")) is None
    assert binding.lookup(request("large")) == {"text": "x" * 70_000}

    await binding.async_store({**request("native"), "ttl_seconds": 12.0}, response)
    stored_response: Final = disk_cache.get_cache("native")
    assert isinstance(stored_response, dict)
    assert stored_response["response"] == response
    stored, expire_time = disk_cache.disk_cache.get("native", expire_time=True)
    assert stored is not None
    assert time.time() < expire_time <= time.time() + 12.0
    await binding.async_store(request("no-ttl"), response)
    _, no_expiry = disk_cache.disk_cache.get("no-ttl", expire_time=True)
    assert no_expiry is None


async def test_disk_entries_survive_a_fresh_handle_and_expire_on_time(tmp_path: Path) -> None:
    first: Final = _native._CacheTestResolver(
        SimpleNamespace(cache=_native._CacheTestHandle.disk(str(tmp_path)))
    ).resolve()
    await first.async_store(request("persistent"), {"value": "persistent"})
    await first.async_store({**request("expiring"), "ttl_seconds": 0.3}, {"value": "expiring"})
    fresh: Final = _native._CacheTestResolver(
        SimpleNamespace(cache=_native._CacheTestHandle.disk(str(tmp_path)))
    ).resolve()
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}
    assert fresh.lookup(request("expiring")) == {"value": "expiring"}
    await asyncio.sleep(0.4)
    assert fresh.lookup(request("expiring")) is None
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}


def test_disk_facade_registers_and_store_changes_fall_back(tmp_path: Path) -> None:
    facade: Final = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    with pytest.raises(TypeError, match="directories must match"):
        _native._CacheTestHandle.disk(str(tmp_path / "other"))._bind_facade(facade)
    handle: Final = _native._CacheTestHandle.disk(str(tmp_path))
    handle._bind_facade(facade)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
    binding: Final = resolver.resolve()
    assert binding.kind == "native"
    binding.store(request("native"), {"value": "native"})
    assert facade.get_cache(cache_key="native") == {"value": "native"}

    with rebound(facade.cache, "disk_cache", diskcache.Cache(str(tmp_path))):
        assert resolver.resolve().kind == "python_callback"
    assert resolver.resolve().kind == "native"

    class CustomDiskCache(DiskCache):
        pass

    with rebound(facade, "cache", CustomDiskCache(disk_cache_dir=str(tmp_path))):
        assert resolver.resolve().kind == "python_callback"

    class CustomStore(diskcache.Cache):
        pass

    custom_facade: Final = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    custom_facade.cache.disk_cache = CustomStore(str(tmp_path))
    with pytest.raises(TypeError, match="built-in diskcache store"):
        _native._CacheTestHandle.disk(str(tmp_path))._bind_facade(custom_facade)


async def test_disk_native_batch_lookup_and_store_report_partial_hits(tmp_path: Path) -> None:
    binding: Final = _native._CacheTestResolver(
        SimpleNamespace(cache=_native._CacheTestHandle.disk(str(tmp_path)))
    ).resolve()
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


async def test_gcs_reads_python_entries_and_writes_python_compatible_objects(
    fake_gcs: FakeGcs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}, "flag": True, "empty": None}
    fake_gcs.put(
        "bucket",
        "cache/sync",
        json.dumps({"timestamp": time.time(), "response": json.dumps(response)}).encode(),
    )
    fake_gcs.put("bucket", "cache/async", json.dumps({"timestamp": time.time(), "response": response}).encode())
    fake_gcs.put("bucket", "cache/raw", json.dumps(response).encode())
    fake_gcs.put("bucket", "cache/invalid", b"not a cache entry")
    binding: Final = _native._CacheTestResolver(
        SimpleNamespace(
            cache=_native._CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()

    assert binding.lookup(request("sync")) == response
    assert await binding.async_lookup(request("async")) == response
    assert binding.lookup(request("raw")) == response
    assert await binding.async_lookup(request("invalid")) is None
    assert binding.lookup(request("missing")) is None

    await binding.async_store({**request("native"), "ttl_seconds": 12.0}, response)
    stored: Final = fake_gcs.objects[("bucket", "cache/native")]
    stored_value: Final = cast(dict[str, object], json.loads(stored))
    assert stored_value["response"] == response
    assert isinstance(stored_value["timestamp"], float)
    upload: Final = next(item for item in fake_gcs.requests if item.method == "POST")
    assert upload.path == "/upload/storage/v1/b/bucket/o"
    assert upload.query == "uploadType=media&name=cache%2Fnative"
    assert upload.headers["Authorization"] == f"Bearer {fake_gcs.token}"
    assert upload.headers["Content-Type"] == "application/json"
    upload_text: Final = f"{upload.path}?{upload.query}{upload.headers}"
    assert "ttl" not in upload_text.lower()
    assert "expiry" not in upload_text.lower()
    download: Final = next(item for item in fake_gcs.requests if item.path.endswith("/cache%2Fsync"))
    assert download.path == "/storage/v1/b/bucket/o/cache%2Fsync"
    assert download.query == "alt=media"

    binding.store(request("sync2"), response)
    assert binding.lookup(request("sync2")) == response
    assert GCSCache(bucket_name="bucket", gcs_path="cache").key_prefix == "cache/"
    assert GCSCache(bucket_name="bucket", gcs_path="cache/").key_prefix == "cache/"
    assert GCSCache(bucket_name="bucket").key_prefix == ""


async def test_gcs_batch_lookup_preserves_order_and_treats_malformed_entries_as_misses(fake_gcs: FakeGcs) -> None:
    fake_gcs.put("bucket", "cache/hit", json.dumps({"timestamp": time.time(), "response": {"value": 1}}).encode())
    fake_gcs.put("bucket", "cache/invalid", b"not a cache entry")
    binding: Final = _native._CacheTestResolver(
        SimpleNamespace(
            cache=_native._CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()
    requests: Final = [request("hit"), request("missing"), request("invalid")]
    expected: Final = {"values": [{"value": 1}, None, None], "missing_indices": [1, 2]}

    assert await binding.async_lookup_batch(requests) == expected
    assert binding.lookup_batch(requests) == expected
    await binding.async_store_batch([request("first"), request("second")], [{"value": 1}, {"value": 2}])
    assert ("bucket", "cache/first") in fake_gcs.objects
    assert ("bucket", "cache/second") in fake_gcs.objects


async def test_gcs_facade_binds_only_exact_matching_configuration(
    fake_gcs: FakeGcs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent")
    facade: Final = Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/")
    assert type(facade.cache) is GCSCache

    mismatched_bucket: Final = _native._CacheTestHandle.gcs(
        "other",
        gcs_path="cache",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    with pytest.raises(TypeError, match="buckets must match"):
        mismatched_bucket._bind_facade(facade)
    mismatched_prefix: Final = _native._CacheTestHandle.gcs(
        "bucket",
        gcs_path="x",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    with pytest.raises(TypeError, match="key prefixes must match"):
        mismatched_prefix._bind_facade(facade)
    mismatched_credentials: Final = _native._CacheTestHandle.gcs(
        "bucket",
        gcs_path="cache",
        path_service_account="sa.json",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    with pytest.raises(TypeError, match="credentials must match"):
        mismatched_credentials._bind_facade(facade)
    with pytest.raises(TypeError, match="types must match"):
        _native._CacheTestHandle.memory()._bind_facade(facade)

    matching: Final = _native._CacheTestHandle.gcs(
        "bucket",
        gcs_path="cache",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    matching._bind_facade(facade)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
    binding: Final = resolver.resolve()
    assert binding.kind == "native"
    await binding.async_store(request("native"), {"value": "native"})
    assert await binding.async_lookup(request("native")) == {"value": "native"}
    assert cast(CacheLookup, facade).get_cache(cache_key="native") is None

    with rebound(facade.cache, "bucket_name", "other"):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "key_prefix", "x/"):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "path_service_account", "sa.json"):
        assert resolver.resolve().kind == "python_callback"
    def no_get_cache(*args: object, **kwargs: object) -> None:
        return None

    with rebound(facade.cache, "get_cache", no_get_cache):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade, "ttl", 12):
        assert resolver.resolve().kind == "python_callback"

    class CustomGcs(GCSCache):
        pass

    with rebound(facade, "cache", CustomGcs(bucket_name="bucket", gcs_path="cache/")):
        assert resolver.resolve().kind == "python_callback"
    custom_facade: Final = Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/")
    with rebound(custom_facade, "cache", CustomGcs(bucket_name="bucket", gcs_path="cache/")):
        with pytest.raises(TypeError, match="types must match"):
            matching._bind_facade(custom_facade)

    missing_bucket: Final = Cache(type=LiteLLMCacheType.GCS)
    with pytest.raises(TypeError, match="requires a configured bucket name"):
        matching._bind_facade(missing_bucket)


async def test_gcs_flush_is_a_no_op_and_ping_is_not_implemented(
    fake_gcs: FakeGcs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    binding: Final = _native._CacheTestResolver(
        SimpleNamespace(
            cache=_native._CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()
    await binding.async_store(request("key"), {"value": "stored"})
    await binding.async_flush()
    assert ("bucket", "cache/key") in fake_gcs.objects
    assert await binding.async_lookup(request("key")) == {"value": "stored"}
    with pytest.raises(NotImplementedError):
        await binding.ping()

    facade: Final = Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/")
    with pytest.raises(AttributeError):
        await facade.ping()
    assert cast(CacheLookup, facade.cache).flush_cache() is None


async def test_gcs_unauthorized_and_server_errors_surface_as_runtime_errors(fake_gcs: FakeGcs) -> None:
    wrong_token: Final = _native._CacheTestResolver(
        SimpleNamespace(
            cache=_native._CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token="wrong-token",
            )
        )
    ).resolve()
    with pytest.raises(RuntimeError):
        wrong_token.lookup(request("missing"))
    assert not fake_gcs.objects

    binding: Final = _native._CacheTestResolver(
        SimpleNamespace(
            cache=_native._CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()
    with pytest.raises(RuntimeError):
        binding.lookup(request("server-error"))
    assert binding.lookup(request("missing")) is None


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
