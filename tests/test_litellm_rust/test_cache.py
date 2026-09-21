import asyncio
import contextvars
import gc
import json
import threading
import time
import weakref
from collections.abc import Generator
from types import SimpleNamespace
from typing import Final, Protocol, cast

import fakeredis
import pytest
import redis

import litellm
from litellm.caching.caching import Cache
from litellm.caching.in_memory_cache import InMemoryCache
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


def test_existing_constructor_and_global_are_unchanged() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    assert type(facade.cache) is InMemoryCache
    assert "_native_cache_handle" not in vars(facade)
    with rebound(litellm, "cache", facade):
        resolver: Final = _native.CacheResolver(litellm)
        assert resolver.resolve().kind == "python_callback"
        resolver.resolve().store(None, {"answer": 7}, callback_kwargs={"cache_key": "key"})
        assert cast(CacheLookup, facade).get_cache(cache_key="key") == {"answer": 7}


async def test_native_bindings_survive_replacement_and_capture_writes_before_dispatch() -> None:
    namespace: Final = SimpleNamespace(cache=_native.NativeCacheHandle.memory())
    resolver: Final = _native.CacheResolver(namespace)
    selected: Final = resolver.resolve()
    assert selected.kind == "native"
    selected.store(request(), {"answer": 1})
    assert await selected.async_lookup(request()) == {"answer": 1}
    with rebound(namespace, "cache", _native.NativeCacheHandle.memory()):
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
    binding: Final = _native.CacheResolver(namespace).resolve()
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

    binding: Final = _native.CacheResolver(SimpleNamespace(cache=CustomCache())).resolve()

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
    handle: Final = _native.NativeCacheHandle.memory()
    handle.bind_facade(facade)
    resolver: Final = _native.CacheResolver(SimpleNamespace(cache=facade))
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

    handle: Final = _native.NativeCacheHandle.memory()
    with pytest.raises(TypeError):
        handle.bind_facade(CustomCache(type=LiteLLMCacheType.LOCAL))
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    handle.bind_facade(facade)
    resolver: Final = _native.CacheResolver(SimpleNamespace(cache=facade))
    with rebound(facade, "cache", InMemoryCache()):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade, "ttl", 12):
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
        binding: Final = _native.CacheResolver(namespace).resolve()
        setattr(callback, "binding", binding)
        return weakref.ref(callback)

    reference: Final = cyclic_reference()
    gc.collect()
    assert reference() is None


async def test_redis_reads_python_sync_and_async_entries_and_writes_without_hidden_prefix(redis_url: str) -> None:
    client: Final = redis.Redis.from_url(redis_url)
    namespace: Final = SimpleNamespace(cache=_native.NativeCacheHandle.redis(redis_url, namespace="team"))
    binding: Final = _native.CacheResolver(namespace).resolve()
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}, "flag": True, "empty": None}
    envelope: Final = {"timestamp": time.time(), "response": json.dumps(response)}
    client.set("team:sync", str(envelope))
    client.set("team:async", json.dumps({"timestamp": time.time(), "response": response}))
    assert binding.lookup(request("sync")) == response
    assert await binding.async_lookup(request("team:async")) == response
    await binding.async_store({**request("native"), "ttl_seconds": 12.0}, response)
    stored: Final = client.get("team:native")
    assert isinstance(stored, bytes)
    assert json.loads(stored)["response"] == response
    assert 0 < client.ttl("team:native") <= 12
    assert client.get("litellm-cache:team:native") is None
    assert client.get("team:team:async") is None
    client.close()


def test_invalid_duration_and_request_shape_fail_before_storage() -> None:
    binding: Final = _native.CacheResolver(SimpleNamespace(cache=_native.NativeCacheHandle.memory())).resolve()
    for seconds in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            binding.store({**request(), "ttl_seconds": seconds}, {"answer": 1})
    assert binding.lookup(request()) is None
    with pytest.raises(ValueError):
        _native.NativeCacheHandle.memory(ttl_seconds=-1)
