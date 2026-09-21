import asyncio
import contextvars
import gc
import hashlib
import json
import math
import os
import threading
import time
import weakref
from collections.abc import Callable, Generator
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Final, Protocol, cast
from urllib.parse import urlparse
from uuid import uuid4

import fakeredis
import pytest
import redis

import litellm
from litellm.caching.caching import Cache, disable_cache, enable_cache, update_cache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.caching.redis_semantic_cache import RedisSemanticCache
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType
from litellm.types.llms.custom_llm import CustomLLMItem
from litellm.types.utils import EmbeddingResponse
from tests.test_litellm_rust.support.isolation import rebound

_CacheTestHandle: Final = _native._CacheTestHandle  # pyright: ignore[reportPrivateUsage]  # test-only handle has no public module name
_CacheTestResolver: Final = _native._CacheTestResolver  # pyright: ignore[reportPrivateUsage]  # test-only resolver has no public module name

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
        resolver: Final = _CacheTestResolver(litellm)
        assert resolver.resolve().kind == "python_callback"
        resolver.resolve().store(None, {"answer": 7}, callback_kwargs={"cache_key": "key"})
        assert cast(CacheLookup, facade).get_cache(cache_key="key") == {"answer": 7}


def test_existing_global_lifecycle_remains_the_resolver_source_of_truth() -> None:
    resolver: Final = _CacheTestResolver(litellm)

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
    namespace: Final = SimpleNamespace(cache=_CacheTestHandle.memory())
    resolver: Final = _CacheTestResolver(namespace)
    selected: Final = resolver.resolve()
    assert selected.kind == "native"
    selected.store(request(), {"answer": 1})
    assert await selected.async_lookup(request()) == {"answer": 1}
    with rebound(namespace, "cache", _CacheTestHandle.memory()):
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
    binding: Final = _CacheTestResolver(namespace).resolve()
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

    binding: Final = _CacheTestResolver(SimpleNamespace(cache=CustomCache())).resolve()

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
    handle: Final = _CacheTestHandle.memory()
    handle._bind_facade(facade)
    resolver: Final = _CacheTestResolver(SimpleNamespace(cache=facade))
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

    handle: Final = _CacheTestHandle.memory()
    with pytest.raises(TypeError):
        handle._bind_facade(CustomCache(type=LiteLLMCacheType.LOCAL))
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    handle._bind_facade(facade)
    resolver: Final = _CacheTestResolver(SimpleNamespace(cache=facade))
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
        binding: Final = _CacheTestResolver(namespace).resolve()
        setattr(callback, "binding", binding)
        return weakref.ref(callback)

    reference: Final = cyclic_reference()
    gc.collect()
    assert reference() is None


async def test_redis_reads_python_sync_and_async_entries_and_writes_without_hidden_prefix(redis_url: str) -> None:
    client: Final = redis.Redis.from_url(redis_url)
    namespace: Final = SimpleNamespace(cache=_CacheTestHandle.redis(redis_url, namespace="team"))
    binding: Final = _CacheTestResolver(namespace).resolve()
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
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=_CacheTestHandle.memory())).resolve()
    for seconds in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cache durations must be finite and nonnegative"):
            binding.store({**request(), "ttl_seconds": seconds}, {"answer": 1})
    assert binding.lookup(request()) is None
    with pytest.raises(ValueError, match="cache durations must be finite and nonnegative"):
        _CacheTestHandle.memory(ttl_seconds=-1)


async def test_memory_size_policy_is_applied_by_the_native_host() -> None:
    handle: Final = _CacheTestHandle.memory(capacity=2, max_entry_bytes=128)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    small: Final = {"answer": "ok"}
    binding.store(request("small"), small)
    assert await binding.async_lookup(request("small")) == small
    await binding.async_store(request("large"), {"answer": "x" * 256})
    assert binding.lookup(request("large")) is None
    assert binding.lookup(request("small")) == small
    disabled: Final = _CacheTestResolver(
        SimpleNamespace(cache=_CacheTestHandle.memory(capacity=0))
    ).resolve()
    await disabled.async_store(request(), small)
    assert await disabled.async_lookup(request()) is None


async def test_native_batch_lookup_and_store_report_partial_hits() -> None:
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=_CacheTestHandle.memory())).resolve()
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

    binding: Final = _CacheTestResolver(
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
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=cache)).resolve()
    assert binding.kind == "python_callback"

    setattr(cache.cache, "ping", ping)
    assert await binding.ping() == "pong"
    await binding.async_flush()
    assert cache.cache.get_cache("key") is None


def test_facade_registration_rejects_mismatched_capacity() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    with pytest.raises(TypeError, match="capacities must match"):
        _CacheTestHandle.memory(capacity=7)._bind_facade(facade)


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
            _CacheTestHandle.redis(redis_url, ttl_seconds=61)._bind_facade(facade)
        with pytest.raises(TypeError, match="namespaces must match"):
            _CacheTestHandle.redis(redis_url, namespace="other")._bind_facade(facade)
        _CacheTestHandle.redis(redis_url, ttl_seconds=60)._bind_facade(facade)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(redis_url)

    with rebound(facade.cache, "redis_kwargs", {**facade.cache.redis_kwargs, "ssl": True}):
        assert _CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"

    pool: Final = facade.cache.redis_client.connection_pool
    with rebound(pool, "connection_kwargs", {**pool.connection_kwargs, "db": 1}):
        assert _CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"

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


PARAPHRASE_MARKER: Final = " (paraphrase)"
SEMANTIC_EMBEDDING_MODEL: Final = "semantic-test/deterministic"
SEMANTIC_INDEX_PREFIX: Final = "litellm_test_semantic_"
SEMANTIC_CONTEXT: Final = contextvars.ContextVar("semantic_test_context", default="unset")


def _normalized(vector: list[float]) -> list[float]:
    norm: Final = math.sqrt(sum(component * component for component in vector))
    return [component / norm for component in vector]


def _base_embedding(prompt: str) -> list[float]:
    digest: Final = hashlib.sha256(prompt.encode("utf-8")).digest()
    return _normalized([float(digest[index] + 1) for index in range(8)])


def _semantic_embedding(prompt: str) -> list[float]:
    if PARAPHRASE_MARKER not in prompt:
        return _base_embedding(prompt)
    base: Final = _base_embedding(prompt.replace(PARAPHRASE_MARKER, "").strip())
    pivot: Final = min(range(8), key=lambda index: abs(base[index]))
    direction: Final = _normalized(
        [
            (1.0 - base[pivot] * base[pivot]) if index == pivot else -base[index] * base[pivot]
            for index in range(8)
        ]
    )
    # Rotating an orthogonal unit direction by 0.329 produces ~0.05 cosine distance
    return _normalized([base[index] + 0.329 * direction[index] for index in range(8)])


class DeterministicEmbedding(litellm.CustomLLM):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.async_calls: list[dict[str, object]] = []

    def _respond(
        self,
        model: str,
        input: object,
        model_response: EmbeddingResponse,
    ) -> EmbeddingResponse:
        texts: Final = cast(list[object], input if isinstance(input, list) else [input])
        self.calls.append({"model": model, "input": texts})
        model_response.model = model
        model_response.data = [
            {"object": "embedding", "index": index, "embedding": _semantic_embedding(str(text))}
            for index, text in enumerate(texts)
        ]
        return model_response

    def embedding(
        self,
        model: str,
        input: list[object],
        model_response: EmbeddingResponse,
        print_verbose: Callable[..., object],
        logging_obj: object,
        optional_params: dict[str, object],
        api_key: object = None,
        api_base: object = None,
        timeout: object = None,
        litellm_params: object = None,
    ) -> EmbeddingResponse:
        return self._respond(model, input, model_response)

    async def aembedding(
        self,
        model: str,
        input: list[object],
        model_response: EmbeddingResponse,
        print_verbose: Callable[..., object],
        logging_obj: object,
        optional_params: dict[str, object],
        api_key: object = None,
        api_base: object = None,
        timeout: object = None,
        litellm_params: object = None,
    ) -> EmbeddingResponse:
        texts: Final = cast(list[object], input if isinstance(input, list) else [input])
        self.async_calls.append(
            {
                "model": model,
                "input": texts,
                "task": asyncio.current_task(),
                "context": SEMANTIC_CONTEXT.get(),
            }
        )
        SEMANTIC_CONTEXT.set("written-in-aembedding")
        return self._respond(model, input, model_response)


@pytest.fixture
def semantic_embedding() -> Generator[DeterministicEmbedding]:
    handler: Final = DeterministicEmbedding()
    with ExitStack() as stack:
        stack.enter_context(
            rebound(
                litellm,
                "custom_provider_map",
                [
                    *litellm.custom_provider_map,
                    cast(
                        CustomLLMItem,
                        {"provider": "semantic-test", "custom_handler": handler},
                    ),
                ],
            )
        )
        stack.enter_context(
            rebound(
                litellm,
                "_custom_providers",  # pyright: ignore[reportPrivateUsage]  # no public provider-registration hook
                [*litellm._custom_providers, "semantic-test"],  # pyright: ignore[reportPrivateUsage]  # no public provider-registration hook
            )
        )
        stack.enter_context(
            rebound(litellm, "provider_list", [*litellm.provider_list, "semantic-test"])
        )
        yield handler


@pytest.fixture
def redis_stack() -> Generator[tuple[str, str]]:
    url: Final = os.environ.get("LITELLM_REDIS_STACK_URL")
    if url is None:
        pytest.skip("LITELLM_REDIS_STACK_URL is not set")
    index: Final = f"{SEMANTIC_INDEX_PREFIX}{uuid4().hex}"
    yield url, index
    client: Final = redis.Redis.from_url(url)
    try:
        client.execute_command("FT.DROPINDEX", index, "DD")  # pyright: ignore[reportUnknownMemberType]  # redis-py leaves execute_command partially unknown
    except redis.RedisError:
        pass
    client.close()


def semantic_request(key: str, prompt: str, **extra: object) -> dict[str, object]:
    return {
        "key": {"preset": key},
        "messages": [{"role": "user", "content": prompt}],
        **extra,
    }


def semantic_messages(prompt: str) -> list[dict[str, object]]:
    return [{"role": "user", "content": prompt}]


def semantic_entry_id(prompt: str, tag: str) -> str:
    return hashlib.sha256(f"{prompt}litellm_cache_key{tag}".encode()).hexdigest()


def semantic_facade(url: str, index: str, *, similarity_threshold: float = 0.8) -> Cache:
    facade: Final = Cache(
        type=LiteLLMCacheType.REDIS_SEMANTIC,
        redis_url=url,
        similarity_threshold=similarity_threshold,
        redis_semantic_cache_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        redis_semantic_cache_index_name=index,
    )
    _CacheTestHandle.redis_semantic(facade.cache)._bind_facade(facade)
    return facade


def test_redis_semantic_constructor_identity_and_provenance(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    backend: Final = cast(RedisSemanticCache, facade.cache)
    assert backend.__class__.__module__ == "litellm.caching.redis_semantic_cache"
    assert type(backend) is RedisSemanticCache
    assert backend._redis_url == url  # pyright: ignore[reportPrivateUsage]  # provenance check needs the projected config
    assert backend._index_name == index  # pyright: ignore[reportPrivateUsage]  # provenance check needs the projected config
    assert backend.similarity_threshold == 0.8
    assert backend.embedding_model == SEMANTIC_EMBEDDING_MODEL
    handle: Final = cast(object, getattr(facade, "_native_cache_handle"))
    assert isinstance(handle, _CacheTestHandle)
    assert handle.backend == "redis_semantic"
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    assert binding.kind == "native"


def test_redis_semantic_native_and_python_sync_entries_share_one_layout(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(url)
    response: Final = {"choices": [{"text": "paris"}], "usage": {"total_tokens": 2}}

    binding.store(semantic_request("geo", "what is the capital of france"), response)

    native_hash_key: Final = f"{index}:{semantic_entry_id('what is the capital of france', 'geo')}"
    stored: Final = client.hgetall(native_hash_key)
    assert set(stored) == {
        b"entry_id",
        b"prompt",
        b"response",
        b"prompt_vector",
        b"inserted_at",
        b"updated_at",
        b"litellm_cache_key",
    }, stored
    assert stored[b"entry_id"].decode() == native_hash_key.split(":", 1)[1]
    assert stored[b"prompt"] == b"what is the capital of france"
    assert stored[b"litellm_cache_key"] == b"geo"
    assert len(stored[b"prompt_vector"]) == 32
    decoded: Final = cast(dict[str, object], json.loads(stored[b"response"]))
    assert decoded["response"] == response
    assert (
        cast(RedisSemanticCache, facade.cache).get_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
            "geo", messages=semantic_messages("what is the capital of france")
        )
        == decoded
    )
    assert semantic_embedding.calls == [
        {"model": "deterministic", "input": ["what is the capital of france"]},
        {"model": "deterministic", "input": ["what is the capital of france"]},
        {"model": "deterministic", "input": ["dimension test"]},
    ]

    cast(RedisSemanticCache, facade.cache).set_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
        "math",
        json.dumps({"timestamp": 1700000000.0, "response": {"answer": 42}}),
        messages=semantic_messages("what is 6 times 7"),
    )
    python_hash_key: Final = f"{index}:{semantic_entry_id('what is 6 times 7', 'math')}"
    assert json.loads(cast(bytes, client.hget(python_hash_key, "response"))) == {
        "timestamp": 1700000000.0,
        "response": {"answer": 42},
    }
    assert binding.lookup(semantic_request("math", "what is 6 times 7")) == {"answer": 42}
    client.close()


async def test_redis_semantic_async_paths_and_store_batch_share_one_layout(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(url)

    await binding.async_store(
        semantic_request("async", "name a primary color"), {"answer": "blue"}
    )
    hash_key: Final = f"{index}:{semantic_entry_id('name a primary color', 'async')}"
    decoded: Final = cast(dict[str, object], json.loads(cast(bytes, client.hget(hash_key, "response"))))
    python_read: Final = await cast(RedisSemanticCache, facade.cache).async_get_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
        "async", messages=semantic_messages("name a primary color")
    )
    assert python_read == decoded

    await binding.async_store_batch(
        [
            semantic_request("batch-one", "first batch prompt"),
            semantic_request("batch-two", "second batch prompt"),
        ],
        [{"answer": 1}, {"answer": 2}],
    )
    expected: Final = {
        key: json.loads(
            cast(bytes, client.hget(f"{index}:{semantic_entry_id(prompt, key)}", "response"))
        )
        for key, prompt in (
            ("batch-one", "first batch prompt"),
            ("batch-two", "second batch prompt"),
        )
    }
    for key, prompt in (
        ("batch-one", "first batch prompt"),
        ("batch-two", "second batch prompt"),
    ):
        assert cast(RedisSemanticCache, facade.cache).get_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
            key, messages=semantic_messages(prompt)
        ) == expected[key], key

    cast(RedisSemanticCache, facade.cache).set_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
        "async-python",
        json.dumps({"timestamp": 1700000000.0, "response": {"answer": "python"}}),
        messages=semantic_messages("python written prompt"),
    )
    assert await binding.async_lookup(
        semantic_request("async-python", "python written prompt")
    ) == {"answer": "python"}
    client.close()


async def test_native_semantic_async_embedding_runs_inline_in_the_callers_task(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    assert binding.kind == "native"
    caller: Final = asyncio.current_task()
    SEMANTIC_CONTEXT.set("caller-sentinel")
    response: Final = {"choices": [{"text": "paris"}]}

    await binding.async_store(
        semantic_request("inline", "what is the capital of france"), response
    )
    assert (
        await binding.async_lookup(
            semantic_request("inline", f"what is the capital of france{PARAPHRASE_MARKER}")
        )
        == response
    )
    assert await binding.async_lookup(semantic_request("inline", "python written prompt")) is None
    assert SEMANTIC_CONTEXT.get() == "written-in-aembedding"
    assert semantic_embedding.async_calls == [
        {
            "model": "deterministic",
            "input": ["what is the capital of france"],
            "task": caller,
            "context": "caller-sentinel",
        },
        {
            "model": "deterministic",
            "input": [f"what is the capital of france{PARAPHRASE_MARKER}"],
            "task": caller,
            "context": "written-in-aembedding",
        },
        {
            "model": "deterministic",
            "input": ["python written prompt"],
            "task": caller,
            "context": "written-in-aembedding",
        },
    ], semantic_embedding.async_calls


def test_redis_semantic_similarity_tag_and_threshold_boundaries(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()

    binding.store(semantic_request("sim", "tell me a joke"), {"answer": "haha"})
    paraphrase: Final = f"tell me a joke{PARAPHRASE_MARKER}"
    assert binding.lookup(semantic_request("sim", paraphrase)) == {"answer": "haha"}
    assert binding.lookup(semantic_request("sim", "an unrelated question about spreadsheets")) is None
    assert binding.lookup(semantic_request("other-key", "tell me a joke")) is None

    strict: Final = semantic_facade(url, index, similarity_threshold=0.99)
    strict_binding: Final = _CacheTestResolver(SimpleNamespace(cache=strict)).resolve()
    assert strict_binding.lookup(semantic_request("sim", paraphrase)) is None
    assert strict_binding.lookup(semantic_request("sim", "tell me a joke")) == {"answer": "haha"}


def test_redis_semantic_ttl_is_written_only_when_requested(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(url)

    binding.store(
        {**semantic_request("ttl", "ttl prompt"), "ttl_seconds": 12.0}, {"answer": 1}
    )
    expiring: Final = f"{index}:{semantic_entry_id('ttl prompt', 'ttl')}"
    assert 0 < client.ttl(expiring) <= 12

    binding.store(semantic_request("ttl-none", "untimed prompt"), {"answer": 2})
    persistent: Final = f"{index}:{semantic_entry_id('untimed prompt', 'ttl-none')}"
    assert client.ttl(persistent) == -1

    binding.store(
        {**semantic_request("ttl-fraction", "fractional prompt"), "ttl_seconds": 1.5},
        {"answer": 3},
    )
    fractional: Final = f"{index}:{semantic_entry_id('fractional prompt', 'ttl-fraction')}"
    assert client.ttl(fractional) == 2
    client.close()


def test_redis_semantic_malformed_response_is_a_miss_for_both_readers(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(url)

    binding.store(semantic_request("bad", "corrupt me"), {"answer": 1})
    hash_key: Final = f"{index}:{semantic_entry_id('corrupt me', 'bad')}"
    client.hset(hash_key, "response", b"{not json")
    assert binding.lookup(semantic_request("bad", "corrupt me")) is None
    assert (
        cast(RedisSemanticCache, facade.cache).get_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
            "bad", messages=semantic_messages("corrupt me")
        )
        is None
    )
    client.close()


async def test_redis_semantic_unsupported_operations_raise_not_implemented(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()

    with pytest.raises(NotImplementedError):
        binding.lookup_batch([semantic_request("batch", "prompt one")])
    with pytest.raises(NotImplementedError):
        await binding.async_lookup_batch([semantic_request("batch", "prompt one")])
    with pytest.raises(NotImplementedError):
        await binding.async_flush()
    with pytest.raises(NotImplementedError):
        await binding.ping()


def test_redis_semantic_requests_without_prompt_are_noops(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(url)

    binding.store(request("plain"), {"answer": 1})
    assert binding.lookup(request("plain")) is None
    assert semantic_embedding.calls == []
    assert client.keys(f"{index}:*") == []
    client.close()


def test_redis_semantic_scope_overrides_the_tag_and_isolates_entries(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = _CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    client: Final = redis.Redis.from_url(url)

    scoped: Final = {**semantic_request("scoped", "scoped prompt"), "scope": "team-a"}
    binding.store(scoped, {"answer": "kept"})
    hash_key: Final = f"{index}:{semantic_entry_id('scoped prompt', 'team-a')}"
    assert client.hget(hash_key, "litellm_cache_key") == b"team-a"
    assert binding.lookup(scoped) == {"answer": "kept"}
    assert binding.lookup(semantic_request("scoped", "scoped prompt")) is None
    assert binding.lookup({**scoped, "scope": "team-b"}) is None
    client.close()


def test_redis_semantic_configuration_drift_falls_back_to_python(
    redis_stack: tuple[str, str],
    semantic_embedding: DeterministicEmbedding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    resolver: Final = _CacheTestResolver(SimpleNamespace(cache=facade))
    assert resolver.resolve().kind == "native"

    with rebound(facade.cache, "similarity_threshold", 0.5):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade, "semantic_cache_scope", "end_user"):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "embedding_model", "other-model"):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "_index_name", "other-index"):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "CACHE_KEY_FIELD_NAME", "other-field"):
        assert resolver.resolve().kind == "python_callback"

    def patched_embedding(self: object, prompt: str, metadata: object = None) -> list[float]:
        return _semantic_embedding(prompt)

    monkeypatch.setattr(RedisSemanticCache, "_get_embedding", patched_embedding)
    assert resolver.resolve().kind == "python_callback"


def test_redis_semantic_handle_rejects_wrong_backends(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack

    class CustomSemanticCache(RedisSemanticCache):
        pass

    with pytest.raises(TypeError, match="built-in RedisSemanticCache"):
        _CacheTestHandle.redis_semantic(object())
    with pytest.raises(TypeError, match="built-in RedisSemanticCache"):
        _CacheTestHandle.redis_semantic(
            CustomSemanticCache(
                redis_url=url,
                similarity_threshold=0.8,
                embedding_model=SEMANTIC_EMBEDDING_MODEL,
                index_name=f"{index}_subclass",
            )
        )

    facade: Final = semantic_facade(url, index)
    with pytest.raises(TypeError, match="backend types must match"):
        _CacheTestHandle.redis(url)._bind_facade(facade)

    subclassed_facade: Final = Cache(
        type=LiteLLMCacheType.REDIS_SEMANTIC,
        redis_url=url,
        similarity_threshold=0.8,
        redis_semantic_cache_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        redis_semantic_cache_index_name=index,
    )
    subclassed_facade.cache = CustomSemanticCache(  # pyright: ignore[reportAttributeAccessIssue]  # facade backend slot is not declared

        redis_url=url,
        similarity_threshold=0.8,
        embedding_model=SEMANTIC_EMBEDDING_MODEL,
        index_name=index,
    )
    with pytest.raises(TypeError):
        _CacheTestHandle.redis_semantic(
            subclassed_facade.cache
        )._bind_facade(subclassed_facade)

    replacement_facade: Final = Cache(
        type=LiteLLMCacheType.REDIS_SEMANTIC,
        redis_url=url,
        similarity_threshold=0.8,
        redis_semantic_cache_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        redis_semantic_cache_index_name=index,
    )
    with pytest.raises(TypeError, match="must be the native embedder"):
        _CacheTestHandle.redis_semantic(facade.cache)._bind_facade(replacement_facade)
