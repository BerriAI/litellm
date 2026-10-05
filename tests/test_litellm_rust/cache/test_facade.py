import asyncio
import contextvars
import gc
import weakref
from types import SimpleNamespace
from typing import Final, cast

import pytest

import litellm
from litellm.caching.caching import Cache, disable_cache, enable_cache, update_cache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.rust_bridge import _native
from litellm.rust_bridge.response_cache import ResponseCacheRuntime
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import (
    CacheLookup,
    CacheTestResolver,
    activate_native,
    native_runtime,
    request,
)
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


def test_existing_constructor_and_global_are_unchanged() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    assert type(facade.cache) is InMemoryCache
    with rebound(litellm, "cache", facade):
        resolver: Final = CacheTestResolver(litellm)
        assert resolver.resolve().kind == "python_callback"
        resolver.resolve().store(None, {"answer": 7}, callback_kwargs={"cache_key": "key"})
        assert cast(CacheLookup, facade).get_cache(cache_key="key") == {"answer": 7}


async def test_explicit_selection_constructs_native_runtime_from_public_cache_configuration() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    runtime: Final = ResponseCacheRuntime(_native._ResponseCacheRuntime.from_cache(facade))
    assert isinstance(runtime, ResponseCacheRuntime)
    assert runtime.kind == "native"

    sync_request: Final = runtime.request(facade, {"cache_key": "sync"})
    assert sync_request is not None
    runtime.store(sync_request, {"answer": 1})
    assert runtime.lookup(sync_request) == {"answer": 1}
    assert facade.cache.get_cache("sync") is None

    async_request: Final = runtime.request(facade, {"cache_key": "async"})
    assert async_request is not None
    await runtime.async_store(async_request, {"answer": 2})
    assert await runtime.async_lookup(async_request) == {"answer": 2}
    assert await facade.cache.async_get_cache("async") is None

    requests: Final = (sync_request, async_request)
    expected: Final = {
        "values": [{"answer": 1}, {"answer": 2}],
        "missing_indices": [],
    }
    assert runtime.lookup_batch(requests) == expected
    assert await runtime.async_lookup_batch(requests) == expected

    await runtime.async_flush()
    assert runtime.lookup(sync_request) is None
    assert await runtime.async_lookup(async_request) is None


async def test_inference_resolver_uses_the_configured_native_cache_directly() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    runtime: Final = ResponseCacheRuntime(_native._ResponseCacheRuntime.from_cache(facade))
    assert isinstance(runtime, ResponseCacheRuntime)
    facade._native_cache = runtime

    selected: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    assert selected.kind == "native"
    request: Final = runtime.request(facade, {"cache_key": "inference-native"})
    assert request is not None
    await selected.async_store(request, {"answer": 42})
    assert await selected.async_lookup(request) == {"answer": 42}
    assert await runtime.async_lookup(request) == {"answer": 42}
    assert facade.cache.get_cache("inference-native") is None

    facade._native_cache = None
    fallback: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    assert fallback.kind == "python_callback"
    await fallback.async_store(None, {"answer": 7}, callback_kwargs={"cache_key": "inference-python"})
    assert facade.get_cache(cache_key="inference-python") == {"answer": 7}
    assert facade.cache.get_cache("inference-python") is not None


async def test_inference_resolver_declines_a_native_runtime_whose_facade_changed() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    runtime: Final = ResponseCacheRuntime(_native._ResponseCacheRuntime.from_cache(facade))
    assert isinstance(runtime, ResponseCacheRuntime)
    facade._native_cache = runtime
    stale_request: Final = runtime.request(facade, {"cache_key": "stale-only"})
    assert stale_request is not None
    await runtime.async_store(stale_request, {"answer": "stale"})

    replacement: Final = InMemoryCache()
    facade.cache = replacement
    with pytest.raises(_native.RustBridgeDeclined):
        CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    assert await runtime.async_lookup(stale_request) == {"answer": "stale"}
    assert replacement.get_cache("stale-only") is None
    assert replacement.get_cache("swapped-backend") is None


def test_existing_global_lifecycle_remains_the_resolver_source_of_truth() -> None:
    resolver: Final = CacheTestResolver(litellm)

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
    namespace: Final = SimpleNamespace(cache=activate_native(Cache(type=LiteLLMCacheType.LOCAL)))
    resolver: Final = CacheTestResolver(namespace)
    selected: Final = resolver.resolve()
    assert selected.kind == "native"
    selected.store(request(), {"answer": 1})
    assert await selected.async_lookup(request()) == {"answer": 1}
    with rebound(namespace, "cache", activate_native(Cache(type=LiteLLMCacheType.LOCAL))):
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
    binding: Final = CacheTestResolver(namespace).resolve()
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

    binding: Final = CacheTestResolver(SimpleNamespace(cache=CustomCache())).resolve()

    async def lookup() -> object:
        return await binding.async_lookup(None, callback_kwargs={})

    task: Final = asyncio.create_task(lookup())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.parametrize("method", ("get_cache", "get_cache_key", "async_get_cache"))
def test_selected_native_runtime_declines_instance_overrides(method: str) -> None:
    facade: Final = activate_native(Cache(type=LiteLLMCacheType.LOCAL))
    selected: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    native: Final = selected.resolve()
    native.store(request(), {"source": "native"})

    def override(**_kwargs: object) -> None:
        return None

    with rebound(facade, method, override):
        with pytest.raises(_native.RustBridgeDeclined):
            selected.resolve()
    assert native.lookup(request()) == {"source": "native"}


@pytest.mark.parametrize(("attribute", "value"), (("ttl", 12), ("semantic_cache_scope", "end_user")))
def test_selected_native_runtime_declines_policy_changes(attribute: str, value: object) -> None:
    facade: Final = activate_native(Cache(type=LiteLLMCacheType.LOCAL))
    selected: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    with rebound(facade, attribute, value):
        with pytest.raises(_native.RustBridgeDeclined):
            selected.resolve()
    assert selected.resolve().kind == "native"


def test_resolver_and_callback_cycles_can_be_collected() -> None:
    class CustomCache:
        pass

    def cyclic_reference() -> weakref.ReferenceType[CustomCache]:
        callback: Final = CustomCache()
        namespace: Final = SimpleNamespace(cache=callback)
        binding: Final = CacheTestResolver(namespace).resolve()
        setattr(callback, "binding", binding)
        return weakref.ref(callback)

    reference: Final = cyclic_reference()
    gc.collect()
    assert reference() is None


def test_invalid_duration_and_request_shape_fail_before_storage() -> None:
    binding: Final = CacheTestResolver(
        SimpleNamespace(cache=activate_native(Cache(type=LiteLLMCacheType.LOCAL)))
    ).resolve()
    for seconds in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cache durations must be finite and nonnegative"):
            binding.store({**request(), "ttl_seconds": seconds}, {"answer": 1})
    assert binding.lookup(request()) is None
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    facade.cache = InMemoryCache(default_ttl=-1)
    with pytest.raises(ValueError, match="cache durations must be finite and nonnegative"):
        native_runtime(facade)


async def test_memory_size_policy_is_applied_by_the_native_host() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    facade.cache = InMemoryCache(max_size_in_memory=2, max_size_per_item=1)
    handle: Final = activate_native(facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    small: Final = {"answer": "ok"}
    binding.store(request("small"), small)
    assert await binding.async_lookup(request("small")) == small
    await binding.async_store(request("large"), {"answer": "x" * 2048})
    assert binding.lookup(request("large")) is None
    assert binding.lookup(request("small")) == small
    disabled_facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    disabled_facade.cache = InMemoryCache(max_size_in_memory=0)
    disabled: Final = native_runtime(disabled_facade)
    await disabled.async_store(request(), small)
    assert await disabled.async_lookup(request()) is None


async def test_native_batch_lookup_and_store_report_partial_hits() -> None:
    binding: Final = CacheTestResolver(
        SimpleNamespace(cache=activate_native(Cache(type=LiteLLMCacheType.LOCAL)))
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

    binding: Final = CacheTestResolver(SimpleNamespace(cache=CustomCache(type=LiteLLMCacheType.LOCAL))).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=cache)).resolve()
    assert binding.kind == "python_callback"

    setattr(cache.cache, "ping", ping)
    assert await binding.ping() == "pong"
    await binding.async_flush()
    assert cache.cache.get_cache("key") is None
