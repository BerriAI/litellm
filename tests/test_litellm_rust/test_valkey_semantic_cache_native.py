import asyncio
import contextvars
import hashlib
import os
import struct
import threading
import time
from collections.abc import Generator, Mapping
from types import SimpleNamespace
from typing import Final, cast
from uuid import uuid4

import pytest
import redis

from litellm.caching.caching import Cache
from litellm.caching.valkey_semantic_cache import ValkeySemanticCache
from litellm.rust_bridge import _native, catalog
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.caching import LiteLLMCacheType

pytestmark: Final = pytest.mark.requires_rust_extension
embedding_context: Final = contextvars.ContextVar("embedding_context")


@pytest.fixture
def valkey_url() -> str:
    url: Final = os.environ.get("LITELLM_TEST_VALKEY_URL")
    if url is None:
        pytest.skip("LITELLM_TEST_VALKEY_URL is not set")
    return url


@pytest.fixture
def index_name(valkey_url: str) -> Generator[str]:
    index: Final = f"litellm_test_{uuid4().hex}"
    yield index
    client: Final = redis.Redis.from_url(valkey_url)
    try:
        client.ft(index).dropindex(delete_documents=True)
    except redis.ResponseError:
        pass
    finally:
        client.close()


def _request(prompt: str = "semantic cache prompt") -> dict[str, object]:
    return {
        "key": {"preset": "key"},
        "messages": [{"role": "user", "content": prompt}],
    }


def _field_request(
    prompt: str,
    metadata: Mapping[str, object],
    *,
    namespace: str | None = None,
    litellm_metadata: Mapping[str, object] | None = None,
    litellm_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    request: Final = {
        "key": {
            "fields": [
                {
                    "name": "model",
                    "value": "gpt-4.1",
                    "api_parameter": True,
                    "internal_parameter": False,
                },
                {
                    "name": "messages",
                    "value": prompt,
                    "api_parameter": True,
                    "internal_parameter": False,
                },
            ],
            "namespace": namespace,
        },
        "messages": [{"role": "user", "content": prompt}],
        "metadata": dict(metadata),
    }
    if litellm_metadata is not None:
        request["litellm_metadata"] = dict(litellm_metadata)
    if litellm_params is not None:
        request["litellm_params"] = dict(litellm_params)
    return request


def _facade(
    url: str,
    index_name: str,
    embeddings: Mapping[str, list[float]],
    *,
    namespace: str | None = None,
) -> Cache:
    facade: Final = Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        redis_url=url,
        similarity_threshold=0.8,
        valkey_semantic_cache_index_name=index_name,
        namespace=namespace,
    )
    vectors: Final = embeddings

    def embed(prompt: str, metadata: Mapping[str, object] | None = None) -> list[float]:
        return vectors[prompt]

    async def async_embedding(prompt: str, metadata: dict[str, object] | None = None) -> list[float]:
        return vectors[prompt]

    facade.cache._get_embedding = embed
    facade.cache._get_async_embedding = async_embedding
    return facade


def _backend(
    url: str,
    index_name: str,
    embeddings: Mapping[str, list[float]] | None = None,
) -> ValkeySemanticCache:
    vectors: Final = embeddings or {"semantic cache prompt": [1.0, 0.0]}
    backend: Final = ValkeySemanticCache(
        redis_url=url,
        similarity_threshold=0.8,
        index_name=index_name,
    )

    def embed(prompt: str, metadata: Mapping[str, object] | None = None) -> list[float]:
        return vectors[prompt]

    async def async_embedding(prompt: str, metadata: dict[str, object] | None = None) -> list[float]:
        return vectors[prompt]

    backend._get_embedding = embed
    backend._get_async_embedding = async_embedding
    return backend


def test_python_write_native_read(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    response: Final = {"answer": "python"}
    backend.set_cache("key", response, messages=_request()["messages"])
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        backend,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    assert binding.lookup(_request()) == response


def test_native_write_python_read(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        backend,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    response: Final = {"answer": "native"}
    binding.store({**_request(), "ttl_seconds": 2.0}, response)
    cached: Final = cast(Mapping[str, object], backend.get_cache("key", messages=_request()["messages"]))
    assert cached["response"] == response


async def test_async_lookup_and_store(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        backend,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    request: Final = {**_request(), "ttl_seconds": 2.0}
    await binding.async_store(request, {"answer": "async"})
    assert await binding.async_lookup(request) == {"answer": "async"}


async def test_disabled_cache_controls_skip_async_embedding(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    calls: Final = []

    async def fail_embedding(prompt: str, metadata: dict[str, object] | None = None) -> list[float]:
        calls.append(prompt)
        raise AssertionError("embedding must not run")

    backend._get_async_embedding = fail_embedding
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        backend,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    controls: Final = {
        "supported_call_type": True,
        "configured": True,
        "native_backend": True,
        "default_on": True,
        "caching": True,
        "no_cache": False,
        "no_store": False,
        "use_cache": True,
    }
    no_read_request: Final = {**_request(), "controls": {**controls, "no_cache": True}}
    assert await binding.async_lookup(no_read_request) is None
    no_write_request: Final = {**_request(), "controls": {**controls, "no_store": True}}
    await binding.async_store(no_write_request, {"answer": "blocked"})
    assert calls == []
    client: Final = redis.Redis.from_url(valkey_url)
    assert list(client.scan_iter(f"{index_name}:*")) == []
    client.close()


async def test_async_embedding_runs_inline_in_caller_task(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    observed: dict[str, object] = {}

    async def async_embedding(prompt: str, metadata: dict[str, object] | None = None) -> list[float]:
        observed["context"] = embedding_context.get("missing")
        observed["task"] = asyncio.current_task()
        observed["thread"] = threading.get_ident()
        embedding_context.set("embedder")
        return [1.0, 0.0]

    backend._get_async_embedding = async_embedding
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        backend,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    request: Final = {**_request(), "ttl_seconds": 2.0}
    caller_task: Final = asyncio.current_task()
    caller_thread: Final = threading.get_ident()
    token: Final = embedding_context.set("caller")
    try:
        await binding.async_store(request, {"answer": "inline"})
        assert observed["context"] == "caller"
        assert observed["task"] is caller_task
        assert observed["thread"] == caller_thread
        assert embedding_context.get() == "embedder"
        assert await binding.async_lookup(request) == {"answer": "inline"}
    finally:
        embedding_context.reset(token)


def test_facade_activation_and_mutation_fallback(
    valkey_url: str,
    index_name: str,
) -> None:
    facade: Final = Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        redis_url=valkey_url,
        similarity_threshold=0.8,
        valkey_semantic_cache_index_name=index_name,
    )
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        facade.cache,
    )
    handle._bind_facade(facade)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
    assert resolver.resolve().kind == "native"
    facade.cache.similarity_threshold = 0.7
    assert resolver.resolve().kind == "python_callback"


def test_batch_lookup_is_unsupported(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        backend,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    with pytest.raises(NotImplementedError):
        binding.lookup_batch([_request()])


def test_ttl_expiry(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    binding.store({**_request(), "ttl_seconds": 1.0}, {"answer": "expires"})
    client: Final = redis.Redis.from_url(valkey_url)
    documents: Final = list(client.scan_iter(f"{index_name}:*"))
    assert len(documents) == 1
    assert client.ttl(documents[0]) > 0
    time.sleep(1.5)
    assert binding.lookup(_request()) is None


def test_no_ttl_is_persistent_and_python_reads_native_value(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    response: Final = {"answer": "persistent"}
    binding.store(_request(), response)
    client: Final = redis.Redis.from_url(valkey_url)
    documents: Final = list(client.scan_iter(f"{index_name}:*"))
    assert len(documents) == 1
    assert client.ttl(documents[0]) == -1
    cached: Final = cast(Mapping[str, object], backend.get_cache("key", messages=_request()["messages"]))
    assert cached["response"] == response


def test_below_threshold_misses_on_native_and_python(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(
        valkey_url,
        index_name,
        {"prompt A": [1.0, 0.0], "prompt B": [0.0, 1.0]},
    )
    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    binding.store(_request("prompt A"), {"answer": "A"})
    assert binding.lookup(_request("prompt B")) is None
    assert backend.get_cache("key", messages=_request("prompt B")["messages"]) is None


def test_malformed_entry_is_a_miss_on_native_and_python(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    client: Final = redis.Redis.from_url(valkey_url)
    scope: Final = hashlib.sha256(b"key").hexdigest()
    document: Final = f"{index_name}:{scope}:{uuid4().hex}"
    client.hset(
        document,
        mapping={
            "litellm_cache_key": scope,
            "prompt": "semantic cache prompt",
            "response": "not json",
            "embedding": struct.pack("<2f", 1.0, 0.0),
        },
    )
    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    assert binding.lookup(_request()) is None
    assert backend.get_cache("key", messages=_request()["messages"]) is None


def test_mixed_content_parts_match_python_semantic_behavior(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    messages: Final = [{"role": "user", "content": ["raw", {"text": "hello"}]}]
    backend.set_cache("key", {"answer": "mixed"}, messages=messages)
    assert backend.get_cache("key", messages=messages) is None

    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    request: Final = {**_request(), "messages": messages}
    binding.store(request, {"answer": "mixed"})
    assert binding.lookup(request) is None
    client: Final = redis.Redis.from_url(valkey_url)
    assert list(client.scan_iter(f"{index_name}:*")) == []
    client.close()


async def test_async_store_batch_and_lookup(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(
        valkey_url,
        index_name,
        {"prompt A": [1.0, 0.0], "prompt B": [0.0, 1.0]},
    )
    sync_calls: Final = []
    async_tasks: Final = []

    def sync_embedding(prompt: str, metadata: Mapping[str, object] | None = None) -> list[float]:
        sync_calls.append(prompt)
        return {"prompt A": [1.0, 0.0], "prompt B": [0.0, 1.0]}[prompt]

    async def async_embedding(
        prompt: str,
        metadata: dict[str, object] | None = None,
    ) -> list[float]:
        async_tasks.append(asyncio.current_task())
        return {"prompt A": [1.0, 0.0], "prompt B": [0.0, 1.0]}[prompt]

    backend._get_embedding = sync_embedding
    backend._get_async_embedding = async_embedding
    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    requests: Final = [_request("prompt A"), _request("prompt B")]
    responses: Final = [{"answer": "A"}, {"answer": "B"}]
    caller_task: Final = asyncio.current_task()
    await binding.async_store_batch(requests, responses)
    assert sync_calls == []
    assert async_tasks
    assert all(task is caller_task for task in async_tasks)
    assert await binding.async_lookup(requests[0]) == responses[0]
    assert await binding.async_lookup(requests[1]) == responses[1]


def test_subclass_backend_falls_back_to_python(
    valkey_url: str,
    index_name: str,
) -> None:
    class Custom(ValkeySemanticCache):
        pass

    facade: Final = Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        redis_url=valkey_url,
        similarity_threshold=0.8,
        valkey_semantic_cache_index_name=index_name,
    )
    facade.cache = Custom(redis_url=valkey_url, similarity_threshold=0.8, index_name=index_name)
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
    assert resolver.resolve().kind == "python_callback"


def test_field_key_matches_python_semantic_scope(
    valkey_url: str,
    index_name: str,
) -> None:
    facade: Final = _facade(valkey_url, index_name, {"semantic cache prompt": [1.0, 0.0]})
    metadata: Final = {"user_api_key": "k1"}
    expected: Final = facade.get_cache_key(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "semantic cache prompt"}],
        metadata=metadata,
    )
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        facade.cache,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    binding.store(_field_request("semantic cache prompt", metadata), {"answer": "scoped"})
    client: Final = redis.Redis.from_url(valkey_url)
    documents: Final = list(client.scan_iter(f"{index_name}:*"))
    assert len(documents) == 1
    document_parts: Final = documents[0].decode().split(":")
    assert document_parts[1] == hashlib.sha256(expected.encode()).hexdigest()
    client.close()


def test_field_key_reads_all_python_tenant_metadata_sources(
    valkey_url: str,
    index_name: str,
) -> None:
    facade: Final = _facade(valkey_url, index_name, {"semantic cache prompt": [1.0, 0.0]})
    params_metadata: Final = {"user_api_key_team_id": "team-from-params"}
    expected: Final = facade.get_cache_key(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "semantic cache prompt"}],
        metadata={},
        litellm_params={"metadata": params_metadata},
    )
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        facade.cache,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    binding.store(
        _field_request(
            "semantic cache prompt",
            {},
            litellm_params={"metadata": params_metadata},
        ),
        {"answer": "params"},
    )
    client: Final = redis.Redis.from_url(valkey_url)
    documents: Final = list(client.scan_iter(f"{index_name}:*"))
    assert len(documents) == 1
    document_parts: Final = documents[0].decode().split(":")
    assert document_parts[1] == hashlib.sha256(expected.encode()).hexdigest()
    client.close()

    assert (
        binding.lookup(
            _field_request(
                "semantic cache prompt",
                {},
                litellm_metadata={"user_api_key_team_id": "team-from-litellm"},
            )
        )
        is None
    )


def test_namespace_isolates_semantic_entries(
    valkey_url: str,
    index_name: str,
) -> None:
    facade: Final = _facade(
        valkey_url,
        index_name,
        {"semantic cache prompt": [1.0, 0.0]},
        namespace="team-a",
    )
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        facade.cache,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    team_a: Final = _field_request("semantic cache prompt", {}, namespace="team-a")
    team_b: Final = _field_request("semantic cache prompt", {}, namespace="team-b")
    binding.store(team_a, {"answer": "team-a"})
    assert binding.lookup(team_b) is None
    assert binding.lookup(team_a) == {"answer": "team-a"}
    cached: Final = cast(
        Mapping[str, object],
        facade.get_cache(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "semantic cache prompt"}],
        ),
    )
    assert cached == {"answer": "team-a"}


def test_field_key_isolates_tenant_scope(
    valkey_url: str,
    index_name: str,
) -> None:
    facade: Final = _facade(valkey_url, index_name, {"semantic cache prompt": [1.0, 0.0]})
    handle: Final = _native._CacheTestHandle.valkey_semantic(
        valkey_url,
        0.8,
        index_name,
        facade.cache,
    )
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    binding.store(
        _field_request("semantic cache prompt", {"user_api_key": "k1"}),
        {"answer": "tenant one"},
    )
    assert binding.lookup(_field_request("semantic cache prompt", {"user_api_key": "k2"})) is None
    assert binding.lookup(_field_request("semantic cache prompt", {"user_api_key": "k1"})) == {"answer": "tenant one"}


def test_tls_valkey_facade_falls_back_to_python(
    index_name: str,
) -> None:
    facade: Final = Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        redis_url="rediss://127.0.0.1:6390/0",
        similarity_threshold=0.8,
        valkey_semantic_cache_index_name=index_name,
    )
    resolver: Final = _native._CacheTestResolver(SimpleNamespace(cache=facade))
    assert resolver.resolve().kind == "python_callback"


async def test_ping_maps_unsupported_native_operation_to_not_implemented(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    with pytest.raises(NotImplementedError):
        await binding.ping()


async def test_rust_required_rule_activates_the_facade_natively(
    valkey_url: str,
    index_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        catalog,
        "RULES",
        (CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({LiteLLMCacheType.VALKEY_SEMANTIC})),),
    )
    facade: Final = _facade(valkey_url, index_name, {"semantic cache prompt": [1.0, 0.0]})
    assert _native._CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "native"
    kwargs: Final = {"model": "gpt-4o", "messages": _request()["messages"]}
    await facade.async_add_cache({"answer": "valkey"}, **kwargs)
    assert await facade.async_get_cache(**kwargs) == {"answer": "valkey"}
