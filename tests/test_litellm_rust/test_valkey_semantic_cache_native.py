import hashlib
import os
import struct
import time
from collections.abc import Generator, Mapping
from types import SimpleNamespace
from typing import Final, cast
from uuid import uuid4

import pytest
import redis

from litellm.caching.caching import Cache
from litellm.caching.valkey_semantic_cache import ValkeySemanticCache
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType

pytestmark: Final = pytest.mark.requires_rust_extension


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


async def test_async_store_batch_and_lookup(
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
    requests: Final = [_request("prompt A"), _request("prompt B")]
    responses: Final = [{"answer": "A"}, {"answer": "B"}]
    await binding.async_store_batch(requests, responses)
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


async def test_ping_maps_unsupported_native_operation_to_not_implemented(
    valkey_url: str,
    index_name: str,
) -> None:
    backend: Final = _backend(valkey_url, index_name)
    handle: Final = _native._CacheTestHandle.valkey_semantic(valkey_url, 0.8, index_name, backend)
    binding: Final = _native._CacheTestResolver(SimpleNamespace(cache=handle)).resolve()
    with pytest.raises(NotImplementedError):
        await binding.ping()
