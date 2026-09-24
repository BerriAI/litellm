import asyncio
import contextvars
import gc
import hashlib
import http.server
import json
import math
import os
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Coroutine, Generator
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Final, Protocol, TypeAlias, cast
from unittest.mock import Mock, patch
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import botocore.config
import diskcache
import fakeredis
import pytest
import redis
from azure.storage.blob import ContainerClient

import litellm
from litellm.caching.azure_blob_cache import AzureBlobCache
from litellm.caching.caching import Cache, disable_cache, enable_cache, update_cache
from litellm.caching.disk_cache import DiskCache
from litellm.caching.gcs_cache import GCSCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.caching.redis_semantic_cache import RedisSemanticCache
from litellm.caching.s3_cache import S3Cache
from litellm.rust_bridge import _native, catalog
from litellm.rust_bridge.catalog import CacheFacadeRule, CacheRule, Route, RouteRule, Rules, SecretManagerRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.response_cache import (
    NativeResponseCacheRuntime,
    ResponseCacheRuntime,
    resolve_native_runtime,
    resolve_response_cache,
    select_cache_facade,
)
from litellm.types.caching import LiteLLMCacheType
from litellm.types.llms.custom_llm import CustomLLMItem
from litellm.types.utils import EmbeddingResponse
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter
from tests.test_litellm_rust.support.fake_gcs import FakeGcs
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.s3_stub import S3Stub

_CacheResolver: Final = _native._CacheResolver  # pyright: ignore[reportPrivateUsage]  # the resolver native routes use has no public module name
NativeCache: Final = _native.Cache

pytestmark: Final = pytest.mark.requires_rust_extension


class CacheLookup(Protocol):
    def get_cache(self, **kwargs: object) -> object: ...
    def flush_cache(self) -> object: ...


def request(key: str = "key") -> dict[str, object]:
    return {"key": {"preset": key}}


def rust_rules(backend: LiteLLMCacheType) -> Rules:
    return (CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({backend})),)


def runtime_for(facade: object) -> NativeResponseCacheRuntime:
    """The native runtime the catalog activates for `facade`'s storage object under a Rust rule."""
    backend: Final = LiteLLMCacheType(cast(str, getattr(facade, "type")))
    runtime: Final = resolve_native_runtime(cast(Cache, facade), rust_rules(backend))
    assert runtime is not None
    assert runtime.kind == "native"
    return runtime


def resolved_kind(cache: object) -> str:
    """How a native route sees `cache` when it is `litellm.cache`."""
    return _CacheResolver(SimpleNamespace(cache=cache)).resolve().kind


def qdrant_request(
    key: str,
    messages: list[dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    return {**request(key), "messages": messages, **kwargs}


def embedding_vector(text: str) -> list[float]:
    raw: Final = hashlib.sha256(text.encode()).digest()[:8]
    values: Final = [byte / 127.5 - 1 for byte in raw]
    norm: Final = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


@pytest.fixture
def qdrant_url() -> str:
    value: Final[str | None] = os.environ.get("QDRANT_URL")
    if not value:
        pytest.skip("QDRANT_URL is required for Qdrant semantic cache tests")
    return value.rstrip("/")


@pytest.fixture
def fake_embedding_endpoint(monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    class EmbeddingHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length: Final = int(self.headers["Content-Length"])
            body: Final = json.loads(self.rfile.read(length))
            text: Final = body["input"]
            response: Final = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": embedding_vector(text),
                    }
                ],
                "model": body["model"],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
            encoded: Final = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args: object) -> None:
            return

    server: Final = http.server.ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("OPENAI_API_BASE", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


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


@pytest.fixture
def cluster_nodes() -> tuple[tuple[str, int], ...]:
    configured: Final = os.environ.get("LITELLM_TEST_REDIS_CLUSTER_NODES")
    if not configured:
        pytest.skip("LITELLM_TEST_REDIS_CLUSTER_NODES is not set")
    return tuple((host, int(port)) for host, _, port in (node.partition(":") for node in configured.split(",")))


def test_existing_constructor_and_global_are_unchanged() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    assert type(facade.cache) is InMemoryCache
    assert resolve_response_cache(facade) is None
    with rebound(litellm, "cache", facade):
        resolver: Final = _CacheResolver(litellm)
        assert resolver.resolve().kind == "python_callback"
        resolver.resolve().store(None, {"answer": 7}, callback_kwargs={"cache_key": "key"})
        assert cast(CacheLookup, facade).get_cache(cache_key="key") == {"answer": 7}


async def test_catalog_constructs_native_runtime_from_public_cache_configuration() -> None:
    rules: Final = (
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY),
        SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({"local"})),
        CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({"local"})),
    )
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    runtime: Final = resolve_response_cache(facade, rules)
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
    rules: Final = (
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY),
        SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({"local"})),
        CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({"local"})),
    )
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    runtime: Final = resolve_response_cache(facade, rules)
    assert isinstance(runtime, ResponseCacheRuntime)
    facade._native_cache = runtime

    selected: Final = _CacheResolver(SimpleNamespace(cache=facade)).resolve()
    assert selected.kind == "native"
    request: Final = runtime.request(facade, {"cache_key": "inference-native"})
    assert request is not None
    await selected.async_store(request, {"answer": 42})
    assert await selected.async_lookup(request) == {"answer": 42}
    assert await runtime.async_lookup(request) == {"answer": 42}
    assert facade.cache.get_cache("inference-native") is None

    facade._native_cache = None
    fallback: Final = _CacheResolver(SimpleNamespace(cache=facade)).resolve()
    assert fallback.kind == "python_callback"
    await fallback.async_store(None, {"answer": 7}, callback_kwargs={"cache_key": "inference-python"})
    assert facade.get_cache(cache_key="inference-python") == {"answer": 7}
    assert facade.cache.get_cache("inference-python") is not None


async def test_inference_resolver_declines_a_native_runtime_whose_facade_changed() -> None:
    rules: Final = (
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY),
        SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({"local"})),
        CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({"local"})),
    )
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    runtime: Final = resolve_response_cache(facade, rules)
    assert isinstance(runtime, ResponseCacheRuntime)
    facade._native_cache = runtime
    stale_request: Final = runtime.request(facade, {"cache_key": "stale-only"})
    assert stale_request is not None
    await runtime.async_store(stale_request, {"answer": "stale"})

    replacement: Final = InMemoryCache()
    facade.cache = replacement
    with pytest.raises(_native.RustBridgeDeclined):
        _CacheResolver(SimpleNamespace(cache=facade)).resolve()
    assert await runtime.async_lookup(stale_request) == {"answer": "stale"}
    assert replacement.get_cache("stale-only") is None
    assert replacement.get_cache("swapped-backend") is None


def test_existing_global_lifecycle_remains_the_resolver_source_of_truth() -> None:
    resolver: Final = _CacheResolver(litellm)

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


async def test_native_bindings_survive_replacement_and_capture_writes_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.LOCAL)
    namespace: Final = SimpleNamespace(cache=Cache(type=LiteLLMCacheType.LOCAL))
    resolver: Final = _CacheResolver(namespace)
    selected: Final = resolver.resolve()
    assert selected.kind == "native"
    selected.store(request(), {"answer": 1})
    assert await selected.async_lookup(request()) == {"answer": 1}
    with rebound(namespace, "cache", Cache(type=LiteLLMCacheType.LOCAL)):
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
    binding: Final = _CacheResolver(namespace).resolve()
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

    binding: Final = _CacheResolver(SimpleNamespace(cache=CustomCache())).resolve()

    async def lookup() -> object:
        return await binding.async_lookup(None, callback_kwargs={})

    task: Final = asyncio.create_task(lookup())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


def test_native_facade_serves_rust_callers_until_an_instance_override_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.LOCAL)
    facade: Final = NativeCache(type=LiteLLMCacheType.LOCAL)
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
    native: Final = resolver.resolve()
    assert native.kind == "native"
    native.store(request(), {"source": "native"})
    assert native.lookup(request()) == {"source": "native"}
    assert facade.get_cache(cache_key="key") == {"source": "native"}
    assert facade.cache.get_cache("key") is None
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


def test_facade_subclasses_backend_replacement_and_configuration_changes_are_not_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.LOCAL)

    class CustomCache(NativeCache):
        pass

    subclassed: Final = CustomCache(type=LiteLLMCacheType.LOCAL)
    assert resolved_kind(subclassed) == "python_callback"
    subclassed.add_cache({"answer": "sub"}, cache_key="sub")
    assert subclassed.cache.get_cache("sub") is None
    assert subclassed.get_cache(cache_key="sub") == {"answer": "sub"}

    facade: Final = NativeCache(type=LiteLLMCacheType.LOCAL)
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
    original: Final = resolver.resolve()
    assert original.kind == "native"
    original.store(request("before"), {"answer": "before"})
    with rebound(facade, "cache", InMemoryCache()):
        replaced: Final = resolver.resolve()
        assert replaced.kind == "native"
        assert replaced.lookup(request("before")) is None
        assert facade.get_cache(cache_key="before") is None
    assert original.lookup(request("before")) == {"answer": "before"}
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
        binding: Final = _CacheResolver(namespace).resolve()
        setattr(callback, "binding", binding)
        return weakref.ref(callback)

    reference: Final = cyclic_reference()
    gc.collect()
    assert reference() is None


async def test_redis_reads_python_sync_and_async_entries_and_writes_without_hidden_prefix(redis_url: str) -> None:
    client: Final = redis.Redis.from_url(redis_url)
    binding: Final = runtime_for(redis_facade(redis_url, namespace="team"))
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
    binding: Final = runtime_for(Cache(type=LiteLLMCacheType.LOCAL))
    for seconds in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cache durations must be finite and nonnegative"):
            binding.store({**request(), "ttl_seconds": seconds}, {"answer": 1})
    assert binding.lookup(request()) is None


async def test_memory_size_policy_is_applied_by_the_native_host() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    facade.cache = InMemoryCache(max_size_in_memory=2, max_size_per_item=1)
    binding: Final = runtime_for(facade)
    small: Final = {"answer": "ok"}
    binding.store(request("small"), small)
    assert await binding.async_lookup(request("small")) == small
    await binding.async_store(request("large"), {"answer": "x" * 2048})
    assert binding.lookup(request("large")) is None
    assert binding.lookup(request("small")) == small
    empty: Final = Cache(type=LiteLLMCacheType.LOCAL)
    empty.cache = InMemoryCache(max_size_in_memory=0)
    disabled: Final = runtime_for(empty)
    await disabled.async_store(request(), small)
    assert await disabled.async_lookup(request()) is None


async def test_native_batch_lookup_and_store_report_partial_hits() -> None:
    binding: Final = runtime_for(Cache(type=LiteLLMCacheType.LOCAL))
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

    binding: Final = _CacheResolver(SimpleNamespace(cache=CustomCache(type=LiteLLMCacheType.LOCAL))).resolve()
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
    binding: Final = _CacheResolver(SimpleNamespace(cache=cache)).resolve()
    assert binding.kind == "python_callback"

    setattr(cache.cache, "ping", ping)
    assert await binding.ping() == "pong"
    await binding.async_flush()
    assert cache.cache.get_cache("key") is None


def test_azure_blob_facade_serves_natively_and_python_reads_the_same_blobs(
    azure_blob_facade: Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = azure_blob_facade.cache
    assert isinstance(backend, AzureBlobCache)
    native: Final = runtime_for(azure_blob_facade)

    response: Final = {
        "choices": [{"text": "caf\u00e9 \u2603"}],
        "usage": {"total_tokens": 3},
        "flag": True,
        "empty": None,
    }
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

    require_rust(monkeypatch, LiteLLMCacheType.AZURE_BLOB)
    account_url: Final = backend.container_client.url.removesuffix(f"/{backend.container_client.container_name}")
    facade: Final = NativeCache(
        type=LiteLLMCacheType.AZURE_BLOB,
        azure_account_url=account_url,
        azure_blob_container=backend.container_client.container_name,
    )
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
    assert resolver.resolve().kind == "native"
    assert facade.get_cache(cache_key="sync") == response
    with rebound(facade, "ttl", 12):
        assert resolver.resolve().kind == "python_callback"
    with rebound(
        facade.cache, "container_client", ContainerClient.from_container_url(facade.cache.container_client.url)
    ):
        assert resolver.resolve().kind == "python_callback"

    def custom_get(*_args: object, **_kwargs: object) -> None:
        return None

    with rebound(facade.cache, "get_cache", custom_get):
        assert resolver.resolve().kind == "python_callback"

    class CustomBlobCache(AzureBlobCache):
        pass

    with rebound(facade, "cache", CustomBlobCache(account_url, backend.container_client.container_name)):
        assert resolver.resolve().kind == "python_callback"


async def test_azure_blob_native_async_writes_overwrite_batch_and_flush_like_python(azure_blob_facade: Cache) -> None:
    backend: Final = azure_blob_facade.cache
    assert isinstance(backend, AzureBlobCache)
    binding: Final = runtime_for(azure_blob_facade)
    ping: Final = cast(dict[str, object], await binding.ping())
    assert ping["status"] == "success", ping

    await binding.async_store(request("async"), {"value": 1})
    await binding.async_store({**request("async"), "ttl_seconds": 0.001}, {"value": 2})
    time.sleep(0.01)
    assert await binding.async_lookup(request("async")) == {"value": 2}
    assert await backend.async_get_cache("async") == json.loads(
        backend.container_client.download_blob("async").readall()
    )
    assert cast(CacheLookup, azure_blob_facade).get_cache(cache_key="async") == {"value": 2}

    await binding.async_store_batch([request("first"), request("second")], [{"value": 3}, {"value": 4}])
    assert await binding.async_lookup_batch([request("second"), request("missing"), request("first")]) == {
        "values": [{"value": 4}, None, {"value": 3}],
        "missing_indices": [1],
    }
    await binding.async_flush()
    assert [blob.name for blob in backend.container_client.list_blobs()] == []
    assert await binding.async_lookup(request("async")) is None


async def test_redis_facade_buffers_native_async_writes(redis_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.REDIS)
    parsed: Final = urlparse(redis_url)
    facade: Final = NativeCache(
        type=LiteLLMCacheType.REDIS,
        host=parsed.hostname,
        port=str(parsed.port),
        redis_flush_size=2,
    )
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
    binding: Final = resolver.resolve()
    assert binding.kind == "native"
    client: Final = redis.Redis.from_url(redis_url)

    with rebound(facade.cache, "redis_kwargs", {**facade.cache.redis_kwargs, "ssl": True}):
        assert resolver.resolve().kind == "python_callback"

    pool: Final = facade.cache.redis_client.connection_pool
    with rebound(pool, "connection_kwargs", {**pool.connection_kwargs, "db": 1}):
        assert resolver.resolve().kind == "python_callback"

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
    binding: Final = runtime_for(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))

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


async def test_disk_entries_survive_a_fresh_runtime_and_expire_on_time(tmp_path: Path) -> None:
    first: Final = runtime_for(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
    await first.async_store(request("persistent"), {"value": "persistent"})
    await first.async_store({**request("expiring"), "ttl_seconds": 0.3}, {"value": "expiring"})
    fresh: Final = runtime_for(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}
    assert fresh.lookup(request("expiring")) == {"value": "expiring"}
    await asyncio.sleep(0.4)
    assert fresh.lookup(request("expiring")) is None
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}


def test_disk_facade_activates_natively_and_store_changes_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.DISK)
    facade: Final = NativeCache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
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
        assert facade.get_cache(cache_key="native") == {"value": "native"}

    class CustomStore(diskcache.Cache):
        pass

    custom_facade: Final = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    custom_facade.cache.disk_cache = CustomStore(str(tmp_path))
    with pytest.raises(RuntimeError, match="built-in diskcache store"):
        runtime_for(custom_facade)


async def test_disk_native_batch_lookup_and_store_report_partial_hits(tmp_path: Path) -> None:
    binding: Final = runtime_for(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
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


@pytest.fixture
def s3_stub() -> Generator[S3Stub]:
    stub: Final = S3Stub()
    try:
        yield stub
    finally:
        stub.close()


def python_s3(url: str) -> S3Cache:
    return S3Cache(
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
    )


async def test_s3_reads_python_entries_and_writes_with_python_metadata(s3_stub: S3Stub) -> None:
    python_cache: Final = python_s3(s3_stub.url)
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}}
    python_cache.set_cache("sync:key", {"timestamp": time.time(), "response": response}, ttl=90)
    python_cache.set_cache("plain", {"timestamp": time.time(), "response": response})
    s3_stub.put_object("team/malformed", b"not a cache entry")
    s3_stub.put_object(
        "team/expired",
        json.dumps({"timestamp": time.time(), "response": response}).encode(),
        {"expires": "Thu, 01 Jan 1970 00:00:00 GMT"},
    )
    binding: Final = runtime_for(s3_facade(s3_stub.url))

    assert binding.lookup(request("sync:key")) == response
    assert await binding.async_lookup(request("plain")) == response
    assert binding.lookup(request("malformed")) is None
    assert binding.lookup(request("expired")) is None
    assert binding.lookup(request("absent")) is None

    binding.store({**request("native:key"), "ttl_seconds": 90.0}, response)
    await binding.async_store(request("no_ttl"), response)
    stored: Final = s3_stub.objects["team/native/key"]
    assert stored.headers["content-type"] == "application/json"
    assert stored.headers["content-language"] == "en"
    assert stored.headers["content-disposition"] == 'inline; filename="team/native/key.json"'
    assert stored.headers["cache-control"] == "immutable, max-age=90, s-maxage=90"
    expires: Final = cast(datetime, s3_stub.expires("team/native/key"))
    remaining: Final = (expires - datetime.now(expires.tzinfo)).total_seconds()
    assert 60 < remaining <= 91
    no_ttl: Final = s3_stub.objects["team/no_ttl"]
    assert no_ttl.headers["cache-control"] == "immutable, max-age=31536000, s-maxage=31536000"
    assert "expires" not in no_ttl.headers
    assert python_cache.get_cache("native:key")["response"] == response

    partial: Final = await binding.async_lookup_batch([request("native:key"), request("absent"), request("malformed")])
    assert partial == {"values": [response, None, None], "missing_indices": [1, 2]}


def s3_facade(url: str, facade_class: type[Cache] = Cache, **settings: object) -> Cache:
    return facade_class(
        type=LiteLLMCacheType.S3,
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
        **settings,
    )


def test_s3_facade_activates_natively_and_falls_back_on_mutation(
    s3_stub: S3Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.S3)
    facade: Final = s3_facade(s3_stub.url, NativeCache)
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
    binding: Final = resolver.resolve()
    assert binding.kind == "native"

    handler: Final = Mock()
    facade.cache.s3_client.meta.events.register("before-call.s3.*", handler)
    binding.store(request("native"), {"answer": 1})
    assert binding.lookup(request("native")) == {"answer": 1}
    assert handler.call_count == 0
    assert "team/native" in s3_stub.objects

    with rebound(facade.cache, "bucket_name", "other"):
        assert resolver.resolve().kind == "python_callback"
    other_client: Final = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=s3_stub.url,
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )
    with rebound(facade.cache, "s3_client", other_client):
        assert resolver.resolve().kind == "python_callback"

    class CustomS3Cache(S3Cache):
        pass

    facade.cache = CustomS3Cache(
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=s3_stub.url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
    )
    assert resolver.resolve().kind == "python_callback"
    assert facade.get_cache(cache_key="native") == {"answer": 1}


def test_s3_facade_rejects_configurations_that_require_python(s3_stub: S3Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.S3)
    with pytest.raises(RuntimeError, match="requires Python"):
        s3_facade("https://s3.example.test", s3_verify=False)
    with pytest.raises(RuntimeError, match="requires Python"):
        s3_facade(s3_stub.url, s3_config=botocore.config.Config(proxies={"https": "http://proxy.test"}))


def test_gcs_facade_activates_natively_and_falls_back_on_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent")
    require_rust(monkeypatch, LiteLLMCacheType.GCS)
    facade: Final = NativeCache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/")
    assert type(facade.cache) is GCSCache
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
    assert resolver.resolve().kind == "native"

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
    delattr(facade.cache, "get_cache")
    assert resolver.resolve().kind == "native"
    with rebound(facade, "ttl", 12):
        assert resolver.resolve().kind == "python_callback"

    class CustomGcs(GCSCache):
        pass

    with rebound(facade, "cache", CustomGcs(bucket_name="bucket", gcs_path="cache/")):
        assert resolver.resolve().kind == "python_callback"
    assert resolver.resolve().kind == "native"

    with pytest.raises(RuntimeError, match="requires a configured bucket name"):
        Cache(type=LiteLLMCacheType.GCS)


async def test_redis_cluster_facade_serves_multi_slot_batches_and_scoped_flush_natively(
    cluster_nodes: tuple[tuple[str, int], ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    startup_nodes: Final = [{"host": host, "port": port} for host, port in cluster_nodes]
    require_rust(monkeypatch, LiteLLMCacheType.REDIS)
    with rebound(litellm, "default_redis_ttl", 60):
        facade: Final = NativeCache(type=LiteLLMCacheType.REDIS, redis_startup_nodes=startup_nodes, namespace="parity")
        assert type(facade.cache) is RedisClusterCache
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
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
        [(1.0 - base[pivot] * base[pivot]) if index == pivot else -base[index] * base[pivot] for index in range(8)]
    )
    # Rotating an orthogonal unit direction by 0.329 produces ~0.05 cosine distance
    return _normalized([base[index] + 0.329 * direction[index] for index in range(8)])


class DeterministicEmbedding(litellm.CustomLLM):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.async_calls: list[dict[str, object]] = []
        self.entered = asyncio.Event()
        self.gate: asyncio.Event | None = None

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
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
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
        stack.enter_context(rebound(litellm, "provider_list", [*litellm.provider_list, "semantic-test"]))
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


def semantic_facade(
    url: str, index: str, *, similarity_threshold: float = 0.8, facade_class: type[Cache] = Cache
) -> Cache:
    facade: Final = facade_class(
        type=LiteLLMCacheType.REDIS_SEMANTIC,
        redis_url=url,
        similarity_threshold=similarity_threshold,
        redis_semantic_cache_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        redis_semantic_cache_index_name=index,
    )
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
    runtime_for(facade)


def test_redis_semantic_native_and_python_sync_entries_share_one_layout(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = runtime_for(facade)
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
    binding: Final = runtime_for(facade)
    client: Final = redis.Redis.from_url(url)

    await binding.async_store(semantic_request("async", "name a primary color"), {"answer": "blue"})
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
        key: json.loads(cast(bytes, client.hget(f"{index}:{semantic_entry_id(prompt, key)}", "response")))
        for key, prompt in (
            ("batch-one", "first batch prompt"),
            ("batch-two", "second batch prompt"),
        )
    }
    for key, prompt in (
        ("batch-one", "first batch prompt"),
        ("batch-two", "second batch prompt"),
    ):
        assert (
            cast(RedisSemanticCache, facade.cache).get_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
                key, messages=semantic_messages(prompt)
            )
            == expected[key]
        ), key

    cast(RedisSemanticCache, facade.cache).set_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
        "async-python",
        json.dumps({"timestamp": 1700000000.0, "response": {"answer": "python"}}),
        messages=semantic_messages("python written prompt"),
    )
    assert await binding.async_lookup(semantic_request("async-python", "python written prompt")) == {"answer": "python"}
    client.close()


async def test_native_semantic_async_embedding_runs_inline_in_the_callers_task(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = runtime_for(facade)
    assert binding.kind == "native"
    caller: Final = asyncio.current_task()
    SEMANTIC_CONTEXT.set("caller-sentinel")
    response: Final = {"choices": [{"text": "paris"}]}

    await binding.async_store(semantic_request("inline", "what is the capital of france"), response)
    assert (
        await binding.async_lookup(semantic_request("inline", f"what is the capital of france{PARAPHRASE_MARKER}"))
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


async def test_native_semantic_cancellation_during_embedding_skips_the_backend(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = runtime_for(facade)
    assert binding.kind == "native"
    semantic_embedding.gate = asyncio.Event()

    async def lookup() -> object:
        return await binding.async_lookup(semantic_request("cancel", "cancelled prompt"))

    task: Final = asyncio.create_task(lookup())
    await semantic_embedding.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    semantic_embedding.gate.set()

    assert len(semantic_embedding.async_calls) == 1
    assert (
        await cast(RedisSemanticCache, facade.cache).async_get_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs stays unknown on the backend class
            "cancel", messages=semantic_messages("cancelled prompt")
        )
        is None
    )


def test_redis_semantic_similarity_tag_and_threshold_boundaries(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = runtime_for(facade)

    binding.store(semantic_request("sim", "tell me a joke"), {"answer": "haha"})
    paraphrase: Final = f"tell me a joke{PARAPHRASE_MARKER}"
    assert binding.lookup(semantic_request("sim", paraphrase)) == {"answer": "haha"}
    assert binding.lookup(semantic_request("sim", "an unrelated question about spreadsheets")) is None
    assert binding.lookup(semantic_request("other-key", "tell me a joke")) is None

    strict: Final = semantic_facade(url, index, similarity_threshold=0.99)
    strict_binding: Final = runtime_for(strict)
    assert strict_binding.lookup(semantic_request("sim", paraphrase)) is None
    assert strict_binding.lookup(semantic_request("sim", "tell me a joke")) == {"answer": "haha"}


def test_redis_semantic_ttl_is_written_only_when_requested(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = runtime_for(facade)
    client: Final = redis.Redis.from_url(url)

    binding.store({**semantic_request("ttl", "ttl prompt"), "ttl_seconds": 12.0}, {"answer": 1})
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
    binding: Final = runtime_for(facade)
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
    binding: Final = runtime_for(facade)

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
    binding: Final = runtime_for(facade)
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
    binding: Final = runtime_for(facade)
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
    require_rust(monkeypatch, LiteLLMCacheType.REDIS_SEMANTIC)
    facade: Final = semantic_facade(url, index, facade_class=NativeCache)
    resolver: Final = _CacheResolver(SimpleNamespace(cache=facade))
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


def test_redis_semantic_native_facade_keeps_custom_backends_on_python(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, index = redis_stack
    require_rust(monkeypatch, LiteLLMCacheType.REDIS_SEMANTIC)

    class CustomSemanticCache(RedisSemanticCache):
        pass

    facade: Final = semantic_facade(url, index, facade_class=NativeCache)
    assert resolved_kind(facade) == "native"
    facade.cache = CustomSemanticCache(
        redis_url=url,
        similarity_threshold=0.8,
        embedding_model=SEMANTIC_EMBEDDING_MODEL,
        index_name=index,
    )
    assert resolved_kind(facade) == "python_callback"


def qdrant_facade(qdrant_url: str, collection_name: str, facade_class: type[Cache] = Cache) -> Cache:
    return facade_class(
        type=LiteLLMCacheType.QDRANT_SEMANTIC,
        qdrant_api_base=qdrant_url,
        qdrant_collection_name=collection_name,
        similarity_threshold=0.99,
        qdrant_semantic_cache_embedding_model="text-embedding-3-small",
        qdrant_semantic_cache_vector_size=8,
    )


def test_qdrant_semantic_facade_binds_native_and_shares_entries(qdrant_url: str, fake_embedding_endpoint: str) -> None:
    del fake_embedding_endpoint
    messages: Final = [{"role": "user", "content": "shared prompt"}]
    collection: Final = f"cache_{uuid4().hex}"
    facade: Final = qdrant_facade(qdrant_url, collection)
    facade.cache.set_cache(
        "python-key",
        {"timestamp": time.time(), "response": json.dumps({"id": "py"})},
        messages=messages,
    )
    binding: Final = runtime_for(facade)
    assert binding.lookup(qdrant_request("python-key", messages)) == {"id": "py"}
    binding.store(qdrant_request("native-key", messages), {"id": "native"})
    python_value: Final = facade.cache.get_cache("native-key", messages=messages)
    assert isinstance(python_value, dict)
    assert python_value["response"] == {"id": "native"}
    unrelated: Final = [{"role": "user", "content": "unrelated prompt"}]
    assert binding.lookup(qdrant_request("native-key", unrelated)) is None
    assert facade.cache.get_cache("native-key", messages=unrelated) is None
    assert binding.lookup(qdrant_request("different-key", messages)) is None
    assert facade.cache.get_cache("different-key", messages=messages) is None


async def test_qdrant_semantic_async_parity(qdrant_url: str, fake_embedding_endpoint: str) -> None:
    del fake_embedding_endpoint
    messages: Final = [{"role": "user", "content": "async prompt"}]
    collection: Final = f"cache_{uuid4().hex}"
    facade: Final = qdrant_facade(qdrant_url, collection)
    binding: Final = runtime_for(facade)
    await facade.cache.async_set_cache(
        "python-key",
        {"timestamp": time.time(), "response": json.dumps({"id": "py"})},
        messages=messages,
    )
    assert await binding.async_lookup(qdrant_request("python-key", messages)) == {"id": "py"}
    await binding.async_store(qdrant_request("native-key", messages), {"id": "native"})
    python_value: Final = await facade.cache.async_get_cache("native-key", messages=messages)
    assert isinstance(python_value, dict)
    assert python_value["response"] == {"id": "native"}


async def test_qdrant_semantic_async_store_batch_shares_entries(qdrant_url: str, fake_embedding_endpoint: str) -> None:
    del fake_embedding_endpoint
    collection: Final = f"cache_{uuid4().hex}"
    facade: Final = qdrant_facade(qdrant_url, collection)
    binding: Final = runtime_for(facade)
    entries: Final = [
        qdrant_request("batch-one", [{"role": "user", "content": "first batch prompt"}]),
        qdrant_request("batch-two", [{"role": "user", "content": "second batch prompt"}]),
    ]
    await binding.async_store_batch(entries, [{"id": "one"}, {"id": "two"}])

    assert binding.lookup(entries[0]) == {"id": "one"}
    assert binding.lookup(entries[1]) == {"id": "two"}
    assert (await facade.cache.async_get_cache("batch-one", messages=entries[0]["messages"]))["response"] == {
        "id": "one"
    }
    assert (await facade.cache.async_get_cache("batch-two", messages=entries[1]["messages"]))["response"] == {
        "id": "two"
    }


async def test_qdrant_semantic_malformed_entries_and_unsupported_operations(
    qdrant_url: str, fake_embedding_endpoint: str
) -> None:
    del fake_embedding_endpoint
    messages: Final = [{"role": "user", "content": "malformed prompt"}]
    collection: Final = f"cache_{uuid4().hex}"
    facade: Final = qdrant_facade(qdrant_url, collection)
    binding: Final = runtime_for(facade)
    key: Final = "malformed-key"
    response: Final = {
        "points": [
            {
                "id": str(uuid4()),
                "vector": embedding_vector("malformed prompt"),
                "payload": {
                    "litellm_cache_key": key,
                    "text": "malformed prompt",
                    "response": "not json",
                },
            }
        ]
    }
    facade.cache.sync_client.put(
        url=f"{qdrant_url}/collections/{collection}/points",
        headers=facade.cache.headers,
        json=response,
    )
    assert binding.lookup(qdrant_request(key, messages)) is None
    with pytest.raises(RuntimeError, match="operation is not supported"):
        binding.lookup_batch([qdrant_request(key, messages)])
    with pytest.raises(RuntimeError, match="operation is not supported"):
        await binding.async_flush()
    with pytest.raises(RuntimeError, match="operation is not supported"):
        await binding.ping()


def test_qdrant_semantic_ignores_request_expiry(qdrant_url: str, fake_embedding_endpoint: str) -> None:
    del fake_embedding_endpoint
    messages: Final = [{"role": "user", "content": "persistent prompt"}]
    collection: Final = f"cache_{uuid4().hex}"
    facade: Final = qdrant_facade(qdrant_url, collection)
    binding: Final = runtime_for(facade)
    binding.store(qdrant_request("persistent-key", messages, ttl_seconds=1.0), {"id": "persistent"})
    time.sleep(1.2)
    assert binding.lookup(qdrant_request("persistent-key", messages)) == {"id": "persistent"}
    python_value: Final = facade.cache.get_cache("persistent-key", messages=messages)
    assert isinstance(python_value, dict)
    assert python_value["response"] == {"id": "persistent"}


def test_qdrant_semantic_mutation_and_projection_fallback(
    qdrant_url: str, fake_embedding_endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_embedding_endpoint
    require_rust(monkeypatch, LiteLLMCacheType.QDRANT_SEMANTIC)
    facade: Final = qdrant_facade(qdrant_url, f"cache_{uuid4().hex}", NativeCache)
    assert resolved_kind(facade) == "native"
    facade.cache.qdrant_api_key = "rotated"
    assert resolved_kind(facade) == "python_callback"
    facade.cache.similarity_threshold = 0.5
    assert resolved_kind(facade) == "python_callback"

    unsupported: Final = qdrant_facade(qdrant_url, f"cache_{uuid4().hex}")
    unsupported.cache.embedding_max_input_tokens = 100
    with pytest.raises(RuntimeError, match="requires Python"):
        runtime_for(unsupported)
    unsupported.cache.embedding_max_input_tokens = None
    unsupported.cache.qdrant_api_base = "http://127.0.0.1:7777"
    with pytest.raises(RuntimeError, match="gRPC"):
        runtime_for(unsupported)


CacheFactory: TypeAlias = Callable[[], Cache]


def require_rust(monkeypatch: pytest.MonkeyPatch, backend: LiteLLMCacheType) -> None:
    monkeypatch.setattr(catalog, "RULES", (CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({backend})),))


def native_runtime(facade: Cache) -> ResponseCacheRuntime:
    runtime: Final = facade._native_cache  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor
    assert isinstance(runtime, ResponseCacheRuntime)
    assert runtime.kind == "native"
    return runtime


@pytest.fixture
def cache_factory(request: pytest.FixtureRequest, tmp_path: Path) -> CacheFactory:
    backend: Final = cast(LiteLLMCacheType, request.param)
    match backend:
        case LiteLLMCacheType.LOCAL:
            return lambda: Cache(type=backend)
        case LiteLLMCacheType.DISK:
            return lambda: Cache(type=backend, disk_cache_dir=str(tmp_path))
        case LiteLLMCacheType.REDIS:
            parsed: Final = urlparse(cast(str, request.getfixturevalue("redis_url")))
            return lambda: Cache(type=backend, host=parsed.hostname, port=str(parsed.port))
        case LiteLLMCacheType.S3:
            stub: Final = cast(S3Stub, request.getfixturevalue("s3_stub"))
            return lambda: Cache(
                type=backend,
                s3_bucket_name="cache-bucket",
                s3_region_name="us-east-1",
                s3_endpoint_url=stub.url,
                s3_aws_access_key_id="key",
                s3_aws_secret_access_key="secret",
                s3_path="team",
            )
        case LiteLLMCacheType.GCS:
            return lambda: Cache(type=backend, gcs_bucket_name="bucket", gcs_path="cache/")
        case LiteLLMCacheType.REDIS_SEMANTIC:
            return lambda: Cache(
                type=backend,
                redis_url="redis://127.0.0.1:6379",
                similarity_threshold=0.8,
                redis_semantic_cache_embedding_model="text-embedding-3-small",
            )
        case LiteLLMCacheType.VALKEY_SEMANTIC:
            return lambda: Cache(type=backend, redis_url="redis://127.0.0.1:6390/0", similarity_threshold=0.8)
        case _:
            raise AssertionError(f"no local factory for {backend}")


ROUND_TRIP_BACKENDS: Final = (
    LiteLLMCacheType.LOCAL,
    LiteLLMCacheType.DISK,
    LiteLLMCacheType.REDIS,
    LiteLLMCacheType.S3,
)
SHARED_STORE_BACKENDS: Final = (LiteLLMCacheType.DISK, LiteLLMCacheType.REDIS, LiteLLMCacheType.S3)


def completion_kwargs(label: str) -> dict[str, object]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": f"{label} {uuid4().hex}"}]}


@pytest.mark.parametrize("backend", list(LiteLLMCacheType))
def test_shipped_rules_keep_every_backend_on_python(backend: LiteLLMCacheType) -> None:
    assert resolve_response_cache(cast(Cache, SimpleNamespace(type=backend))) is None


@pytest.mark.parametrize(
    "cache_factory",
    [
        LiteLLMCacheType.LOCAL,
        LiteLLMCacheType.DISK,
        LiteLLMCacheType.REDIS,
        LiteLLMCacheType.S3,
        LiteLLMCacheType.GCS,
        LiteLLMCacheType.REDIS_SEMANTIC,
        LiteLLMCacheType.VALKEY_SEMANTIC,
    ],
    indirect=True,
)
def test_shipped_rules_construct_python_backed_facades(cache_factory: CacheFactory) -> None:
    assert cache_factory()._native_cache is None  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor


@pytest.mark.parametrize(
    "cache_factory",
    [
        LiteLLMCacheType.LOCAL,
        LiteLLMCacheType.DISK,
        LiteLLMCacheType.REDIS,
        LiteLLMCacheType.S3,
        LiteLLMCacheType.GCS,
        LiteLLMCacheType.REDIS_SEMANTIC,
        LiteLLMCacheType.VALKEY_SEMANTIC,
    ],
    indirect=True,
)
def test_rust_required_rule_activates_the_native_backend(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    native_runtime(cache_factory())


@pytest.mark.parametrize("cache_factory", ROUND_TRIP_BACKENDS, indirect=True)
async def test_facade_storage_calls_round_trip_through_the_native_backend(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    facade: Final = cache_factory()
    native_runtime(facade)

    sync_kwargs: Final = completion_kwargs("sync")
    facade.add_cache({"answer": 1}, **sync_kwargs)
    assert facade.get_cache(**sync_kwargs) == {"answer": 1}

    async_kwargs: Final = completion_kwargs("async")
    await facade.async_add_cache({"answer": 2}, **async_kwargs)
    assert await facade.async_get_cache(**async_kwargs) == {"answer": 2}
    assert facade.get_cache(**completion_kwargs("absent")) is None


async def test_memory_facade_writes_bypass_the_python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.LOCAL)
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    native_runtime(facade)
    kwargs: Final = completion_kwargs("memory")
    facade.add_cache({"answer": 1}, **kwargs)
    assert facade.cache.get_cache(facade.get_cache_key(**kwargs)) is None
    assert facade.get_cache(**kwargs) == {"answer": 1}


@pytest.mark.parametrize("cache_factory", SHARED_STORE_BACKENDS, indirect=True)
async def test_native_and_python_facades_share_one_wire_format(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    python_facade: Final = cache_factory()
    assert python_facade._native_cache is None  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    native_facade: Final = cache_factory()
    native_runtime(native_facade)

    native_written: Final = completion_kwargs("native")
    native_facade.add_cache({"writer": "native"}, **native_written)
    assert python_facade.get_cache(**native_written) == {"writer": "native"}

    python_written: Final = completion_kwargs("python")
    python_facade.add_cache({"writer": "python"}, **python_written)
    assert native_facade.get_cache(**python_written) == {"writer": "python"}

    async_native: Final = completion_kwargs("async-native")
    await native_facade.async_add_cache({"writer": "async-native"}, **async_native)
    assert await python_facade.async_get_cache(**async_native) == {"writer": "async-native"}

    async_python: Final = completion_kwargs("async-python")
    await python_facade.async_add_cache({"writer": "async-python"}, **async_python)
    assert await native_facade.async_get_cache(**async_python) == {"writer": "async-python"}


@pytest.mark.parametrize("cache_factory", ROUND_TRIP_BACKENDS, indirect=True)
async def test_embedding_pipeline_stores_one_native_entry_per_input(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    facade: Final = cache_factory()
    native_runtime(facade)
    inputs: Final = [f"alpha {uuid4().hex}", f"beta {uuid4().hex}"]
    result: Final = EmbeddingResponse(
        model="text-embedding-3-small",
        data=[
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
    )
    await facade.async_add_cache_pipeline(result, model="text-embedding-3-small", input=inputs)

    keys: Final = [facade.get_cache_key(model="text-embedding-3-small", input=text) for text in inputs]
    assert len(set(keys)) == len(inputs)
    for text, expected in zip(inputs, ([0.1, 0.2], [0.3, 0.4]), strict=True):
        cached = await facade.async_get_cache(model="text-embedding-3-small", input=text)
        assert isinstance(cached, dict)
        assert cached["embedding"] == expected
    assert await facade.async_get_cache(model="text-embedding-3-small", input=inputs) is None


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
    native_runtime(redis_facade(redis_url, ssl=True, ssl_check_hostname=True))


async def test_redis_flush_size_buffers_native_facade_writes(redis_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.REDIS)
    facade: Final = redis_facade(redis_url, redis_flush_size=2, namespace="team")
    native_runtime(facade)
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


@pytest.mark.parametrize(
    ("backend", "settings", "message"),
    [
        pytest.param(
            LiteLLMCacheType.VALKEY_SEMANTIC,
            {"redis_url": "rediss://127.0.0.1:6390/0", "similarity_threshold": 0.8},
            "native Valkey semantic cache does not support TLS connections",
            id="valkey-tls",
        ),
        pytest.param(
            LiteLLMCacheType.VALKEY_SEMANTIC,
            {"redis_url": "redis://127.0.0.1:6390/0?socket_timeout=1", "similarity_threshold": 0.8},
            "native Redis uses fixed socket timeouts; socket_timeout and socket_connect_timeout require Python",
            id="valkey-socket-timeout",
        ),
        pytest.param(
            LiteLLMCacheType.REDIS_SEMANTIC,
            {"redis_url": "rediss://127.0.0.1:6380", "similarity_threshold": 0.8},
            "native Redis semantic cache does not support TLS or query options in redis_url",
            id="redis-semantic-tls",
        ),
        pytest.param(
            LiteLLMCacheType.REDIS_SEMANTIC,
            {"redis_url": "redis://127.0.0.1:6379?socket_timeout=1", "similarity_threshold": 0.8},
            "native Redis semantic cache does not support TLS or query options in redis_url",
            id="redis-semantic-query",
        ),
    ],
)
def test_semantic_settings_the_native_client_cannot_honor_decline(
    monkeypatch: pytest.MonkeyPatch, backend: LiteLLMCacheType, settings: dict[str, object], message: str
) -> None:
    require_rust(monkeypatch, backend)
    with pytest.raises(RuntimeError, match=f"declined the cache: {message}"):
        Cache(type=backend, **settings)


def test_rust_with_fallback_keeps_python_when_the_native_client_declines(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        catalog,
        "RULES",
        (CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.REDIS})),),
    )
    assert redis_facade(redis_url, socket_timeout=1.0)._native_cache is None  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor


def test_qdrant_semantic_rust_required_rule_activates_natively(
    qdrant_url: str, fake_embedding_endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_embedding_endpoint
    require_rust(monkeypatch, LiteLLMCacheType.QDRANT_SEMANTIC)
    facade: Final = qdrant_facade(qdrant_url, f"cache_{uuid4().hex}")
    native_runtime(facade)
    kwargs: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "qdrant activation"}]}
    facade.add_cache({"answer": "qdrant"}, **kwargs)
    assert facade.get_cache(**kwargs) == {"answer": "qdrant"}


async def test_redis_semantic_rust_required_rule_activates_natively(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding, monkeypatch: pytest.MonkeyPatch
) -> None:
    del semantic_embedding
    url, index = redis_stack
    require_rust(monkeypatch, LiteLLMCacheType.REDIS_SEMANTIC)
    facade: Final = Cache(
        type=LiteLLMCacheType.REDIS_SEMANTIC,
        redis_url=url,
        similarity_threshold=0.8,
        redis_semantic_cache_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        redis_semantic_cache_index_name=index,
    )
    native_runtime(facade)
    kwargs: Final = {"model": "gpt-4o", "messages": semantic_messages("name a primary color")}
    await facade.async_add_cache({"answer": "blue"}, **kwargs)
    assert await facade.async_get_cache(**kwargs) == {"answer": "blue"}


async def test_azure_blob_rust_required_rule_activates_natively(monkeypatch: pytest.MonkeyPatch) -> None:
    account_url: Final = os.environ.get("AZURE_BLOB_CACHE_ACCOUNT_URL")
    if account_url is None:
        pytest.skip(
            "live Azure Blob parity needs AZURE_BLOB_CACHE_ACCOUNT_URL plus DefaultAzureCredential inputs in the environment"
        )
    require_rust(monkeypatch, LiteLLMCacheType.AZURE_BLOB)
    facade: Final = Cache(
        type=LiteLLMCacheType.AZURE_BLOB,
        azure_account_url=account_url,
        azure_blob_container=f"litellm-parity-{uuid.uuid4().hex[:12]}",
    )
    backend: Final = facade.cache
    assert isinstance(backend, AzureBlobCache)
    try:
        native_runtime(facade)
        kwargs: Final = completion_kwargs("azure")
        await facade.async_add_cache({"answer": "azure"}, **kwargs)
        assert await facade.async_get_cache(**kwargs) == {"answer": "azure"}
        assert backend.get_cache(facade.get_cache_key(**kwargs))["response"] == {"answer": "azure"}
    finally:
        backend.container_client.delete_container()
        await backend.disconnect()


class _SemanticHit:
    """A native semantic runtime that answers every lookup with one cached response."""

    kind: Final = "native"

    def lookup_semantic(self, request: object) -> tuple[object, float | None]:
        return {"answer": 42}, 0.97

    async def async_lookup_semantic(self, request: object) -> tuple[object, float | None]:
        return {"answer": 42}, 0.97


@pytest.mark.parametrize("semantic_type", [LiteLLMCacheType.QDRANT_SEMANTIC, LiteLLMCacheType.REDIS_SEMANTIC])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_native_semantic_hit_stamps_similarity_on_request_metadata(
    semantic_type: LiteLLMCacheType, use_async: bool
) -> None:
    """Python semantic backends write `metadata["semantic-similarity"]` on every lookup, and the
    facade copies it to the caller's metadata; the native path must report it the same way."""
    facade: Final = Cache()
    facade.type = semantic_type
    facade._native_cache = ResponseCacheRuntime(cast(NativeResponseCacheRuntime, _SemanticHit()))  # pyright: ignore[reportPrivateUsage]  # the native path under test has no public setter
    metadata: Final[dict[str, object]] = {}
    kwargs: Final = {
        "cache_key": "semantic-key",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": metadata,
    }

    result: Final = asyncio.run(facade.async_get_cache(**kwargs)) if use_async else facade.get_cache(**kwargs)

    assert result == {"answer": 42}
    assert metadata["semantic-similarity"] == 0.97


def test_shipped_rules_keep_the_python_facade_and_a_rust_rule_selects_the_native_one() -> None:
    class PythonFacade:
        pass

    assert select_cache_facade(PythonFacade) is PythonFacade
    assert select_cache_facade(PythonFacade, (CacheFacadeRule(Rollout.RUST_REQUIRED),)) is NativeCache


def test_native_facade_keeps_python_subclass_and_attribute_semantics() -> None:
    class Initialized(NativeCache):
        def __init__(self) -> None:
            super().__init__(type=LiteLLMCacheType.LOCAL, ttl=7)

    class Uninitialized(NativeCache):
        def __init__(self) -> None:
            pass

    initialized: Final = Initialized()
    assert isinstance(initialized, NativeCache)
    assert type(initialized.cache) is InMemoryCache
    assert initialized.ttl == 7
    uninitialized: Final = Uninitialized()
    with pytest.raises(AttributeError):
        _ = uninitialized.cache
    with pytest.raises(AttributeError):
        _ = uninitialized.type

    facade: Final = NativeCache(type=LiteLLMCacheType.LOCAL, namespace="team")
    assert {"type", "mode", "ttl", "namespace", "supported_call_types", "semantic_cache_scope"} <= set(vars(facade))
    facade.ttl = 12
    assert facade.ttl == 12
    blank: Final = NativeCache.__new__(NativeCache)
    blank.type = LiteLLMCacheType.LOCAL
    blank.mode = "default_on"
    blank.ttl = None
    blank.cache = InMemoryCache()
    blank.add_cache({"answer": "blank"}, cache_key="blank")
    assert blank.get_cache(cache_key="blank") == {"answer": "blank"}


@pytest.mark.parametrize("facade_class", [Cache, NativeCache], ids=["python", "native"])
def test_both_facades_derive_the_same_keys_and_envelopes(facade_class: type[Cache]) -> None:
    kwargs: Final = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "metadata": {"model_group": "group", "caching_groups": [("group", "other")]},
        "litellm_params": {},
    }
    python_key: Final = Cache(type=LiteLLMCacheType.LOCAL, namespace="team").get_cache_key(**kwargs)
    facade: Final = facade_class(type=LiteLLMCacheType.LOCAL, namespace="team", ttl=30)
    assert facade.get_cache_key(**kwargs) == python_key
    assert kwargs["litellm_params"] == {"preset_cache_key": python_key}
    facade.add_cache({"answer": 1}, cache_key="envelope")
    stored: Final = cast(dict[str, object], facade.cache.get_cache("envelope"))
    assert set(stored) == {"timestamp", "response"}
    assert stored["response"] == {"answer": 1}
    assert facade.get_cache(cache_key="envelope", cache={"s-maxage": 1e-9}) is None
    assert facade.get_cache(cache_key="envelope", cache={"s-maxage": 0}) == {"answer": 1}


async def test_native_facade_async_methods_return_coroutines_on_every_path() -> None:
    disabled_facade: Final = NativeCache(type=LiteLLMCacheType.LOCAL, mode="default_off")
    disabled: Final = disabled_facade.async_get_cache(cache_key="k")
    assert asyncio.iscoroutine(disabled)
    assert await disabled is None

    facade: Final = NativeCache(type=LiteLLMCacheType.LOCAL)
    write: Final = facade.async_add_cache({"answer": 1}, cache_key="k")
    assert asyncio.iscoroutine(write)
    await asyncio.create_task(write)
    read: Final = facade.async_get_cache(cache_key="k")
    assert asyncio.iscoroutine(read)
    assert await asyncio.create_task(read) == {"answer": 1}


def test_native_facade_overridden_helpers_steer_its_own_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.LOCAL)
    facade: Final = NativeCache(type=LiteLLMCacheType.LOCAL)
    assert resolved_kind(facade) == "native"
    kwargs: Final = completion_kwargs("override")
    facade.add_cache({"answer": "native"}, **kwargs)
    assert facade.get_cache(**kwargs) == {"answer": "native"}

    with patch.object(NativeCache, "get_cache_key", return_value="patched-key"):
        assert resolved_kind(facade) == "python_callback"
        facade.add_cache({"answer": "patched"}, **kwargs)
        assert facade.get_cache(cache_key="patched-key") == {"answer": "patched"}
    assert resolved_kind(facade) == "native"
    assert facade.get_cache(**kwargs) == {"answer": "native"}

    class KeyedCache(NativeCache):
        def get_cache_key(self, **kwargs: object) -> str:
            del kwargs
            return "subclass-key"

    keyed: Final = KeyedCache(type=LiteLLMCacheType.LOCAL)
    keyed.add_cache({"answer": "sub"}, **kwargs)
    assert keyed.get_cache(model="other") == {"answer": "sub"}
    assert resolved_kind(keyed) == "python_callback"


_FINALIZER_READS_THE_FACADE: Final = """
from types import SimpleNamespace

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.rust_bridge import _native, catalog
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.caching import LiteLLMCacheType

if {native}:
    catalog.RULES = (CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({{LiteLLMCacheType.LOCAL}})),)
facade = _native.Cache(type=LiteLLMCacheType.LOCAL)


class ReadsFacadeWhenCollected:
    def __del__(self):
        print("finalizer saw", type(facade.cache).__name__, flush=True)


class Replacement(InMemoryCache):
    pass


backend = InMemoryCache()
backend.finalizer = ReadsFacadeWhenCollected()
facade.cache = backend
del backend
print(_native._CacheResolver(SimpleNamespace(cache=facade)).resolve().kind, flush=True)
facade.cache = Replacement()
print("replaced", flush=True)
"""


@pytest.mark.parametrize(("native", "kind"), [(False, "python_callback"), (True, "native")], ids=["python", "native"])
def test_replacing_the_backend_lets_its_finalizer_read_the_facade(native: bool, kind: str) -> None:
    result: Final = run_child_interpreter(_FINALIZER_READS_THE_FACADE.format(native=native), timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [kind, "finalizer saw Replacement", "replaced"], result.stderr


class RecordingBackend(InMemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def _record(self, name: str) -> Coroutine[object, object, None]:
        self.calls.append(name)
        return asyncio.sleep(0)

    def async_get_cache(self, key: object, **kwargs: object) -> Coroutine[object, object, None]:
        return self._record("async_get_cache")

    def async_set_cache(self, key: object, value: object, **kwargs: object) -> Coroutine[object, object, None]:
        return self._record("async_set_cache")

    def async_set_cache_pipeline(
        self, cache_list: object, ttl: object = None, **kwargs: object
    ) -> Coroutine[object, object, None]:
        return self._record("async_set_cache_pipeline")

    def batch_cache_write(self, key: object, value: object, **kwargs: object) -> Coroutine[object, object, None]:
        return self._record("batch_cache_write")

    def ping(self) -> Coroutine[object, object, None]:
        return self._record("ping")

    def delete_cache_keys(self, keys: object) -> Coroutine[object, object, None]:
        return self._record("delete_cache_keys")

    def disconnect(self) -> Coroutine[object, object, None]:
        return self._record("disconnect")


def embedding_result() -> EmbeddingResponse:
    return EmbeddingResponse(
        model="text-embedding-3-small", data=[{"object": "embedding", "index": 0, "embedding": [0.5]}]
    )


AsyncCall: TypeAlias = Callable[[Cache], Coroutine[object, object, object]]

ASYNC_CALLS: Final[tuple[tuple[str, AsyncCall, str], ...]] = (
    ("async_get_cache", lambda facade: facade.async_get_cache(**completion_kwargs("get")), "async_get_cache"),
    ("async_add_cache", lambda facade: facade.async_add_cache({"a": 1}, **completion_kwargs("add")), "async_set_cache"),
    (
        "async_add_cache_pipeline",
        lambda facade: facade.async_add_cache_pipeline(embedding_result(), model="m", input=["hello"]),
        "async_set_cache_pipeline",
    ),
    (
        "batch_cache_write",
        lambda facade: facade.batch_cache_write({"a": 1}, **completion_kwargs("batch")),
        "batch_cache_write",
    ),
    ("ping", lambda facade: facade.ping(), "ping"),
    ("delete_cache_keys", lambda facade: facade.delete_cache_keys(["key"]), "delete_cache_keys"),
    ("disconnect", lambda facade: facade.disconnect(), "disconnect"),
)


@pytest.mark.parametrize("facade_class", [Cache, NativeCache], ids=["python", "native"])
@pytest.mark.parametrize(
    ("start", "backend_method"),
    [(start, backend_method) for _, start, backend_method in ASYNC_CALLS],
    ids=[name for name, _, _ in ASYNC_CALLS],
)
async def test_both_facades_run_async_methods_only_when_awaited(
    facade_class: type[Cache], start: AsyncCall, backend_method: str
) -> None:
    built_keys: Final[list[dict[str, object]]] = []

    class KeyRecording(facade_class):
        def get_cache_key(self, **kwargs: object) -> str:
            built_keys.append(kwargs)
            return super().get_cache_key(**kwargs)

    first: Final = RecordingBackend()
    second: Final = RecordingBackend()
    facade: Final = KeyRecording(type=LiteLLMCacheType.LOCAL)
    facade.cache = first
    start(facade).close()
    pending: Final = start(facade)
    assert (built_keys, first.calls) == ([], [])

    facade.cache = second
    await pending

    assert (first.calls, second.calls) == ([], [backend_method])


def constructed_with_none(facade_class: type[Cache], parameter: str) -> object:
    try:
        facade: Final = facade_class(**{parameter: None})
    except ValueError as error:
        return type(error)
    return facade.type, facade.supported_call_types, facade.mode, type(getattr(facade, "cache", None))


@pytest.mark.parametrize("parameter", ["type", "supported_call_types", "mode", "semantic_cache_scope"])
def test_native_facade_binds_an_explicit_none_like_the_python_facade(parameter: str) -> None:
    assert constructed_with_none(NativeCache, parameter) == constructed_with_none(Cache, parameter)


def completion_id(prompt: str) -> str:
    response: Final = litellm.completion(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], mock_response="cached"
    )
    assert isinstance(response, litellm.ModelResponse)
    return response.id


@pytest.mark.parametrize("facade_class", [Cache, NativeCache], ids=["python", "native"])
@pytest.mark.parametrize(
    ("settings", "cache_hit"), [({}, True), ({"supported_call_types": None}, False)], ids=["default", "none"]
)
def test_both_facades_skip_completion_caching_when_supported_call_types_is_none(
    facade_class: type[Cache], settings: dict[str, None], cache_hit: bool
) -> None:
    prompt: Final = f"supported call types {uuid4().hex}"
    with rebound(litellm, "cache", facade_class(type=LiteLLMCacheType.LOCAL, **settings)):
        first: Final = completion_id(prompt)
        second: Final = completion_id(prompt)
    assert (first == second) is cache_hit
