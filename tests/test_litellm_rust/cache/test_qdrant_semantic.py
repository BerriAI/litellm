import hashlib
import http.server
import json
import math
import os
import threading
import time
from collections.abc import Generator
from types import SimpleNamespace
from typing import Final
from uuid import uuid4

import pytest

from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import (
    CacheTestHandle,
    CacheTestResolver,
    assert_native_runtime,
    request,
    require_rust,
)

pytestmark: Final = pytest.mark.requires_rust_extension


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


def qdrant_facade(qdrant_url: str, collection_name: str) -> Cache:
    return Cache(
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
    handle: Final = CacheTestHandle.qdrant_semantic(
        qdrant_url,
        collection_name=collection,
        similarity_threshold=0.99,
        vector_size=8,
    )
    handle._bind_facade(facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    assert binding.kind == "native"
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
    handle: Final = CacheTestHandle.qdrant_semantic(
        qdrant_url,
        collection_name=collection,
        similarity_threshold=0.99,
        vector_size=8,
    )
    handle._bind_facade(facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    handle: Final = CacheTestHandle.qdrant_semantic(
        qdrant_url,
        collection_name=collection,
        similarity_threshold=0.99,
        vector_size=8,
    )
    handle._bind_facade(facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    handle: Final = CacheTestHandle.qdrant_semantic(
        qdrant_url,
        collection_name=collection,
        similarity_threshold=0.99,
        vector_size=8,
    )
    handle._bind_facade(facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    handle: Final = CacheTestHandle.qdrant_semantic(
        qdrant_url,
        collection_name=collection,
        similarity_threshold=0.99,
        vector_size=8,
    )
    handle._bind_facade(facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    binding.store(qdrant_request("persistent-key", messages, ttl_seconds=1.0), {"id": "persistent"})
    time.sleep(1.2)
    assert binding.lookup(qdrant_request("persistent-key", messages)) == {"id": "persistent"}
    python_value: Final = facade.cache.get_cache("persistent-key", messages=messages)
    assert isinstance(python_value, dict)
    assert python_value["response"] == {"id": "persistent"}


def test_qdrant_semantic_mutation_and_projection_fallback(qdrant_url: str, fake_embedding_endpoint: str) -> None:
    del fake_embedding_endpoint
    collection: Final = f"cache_{uuid4().hex}"
    facade: Final = qdrant_facade(qdrant_url, collection)
    handle: Final = CacheTestHandle.qdrant_semantic(
        qdrant_url,
        collection_name=collection,
        similarity_threshold=0.99,
        vector_size=8,
    )
    handle._bind_facade(facade)
    facade.cache.qdrant_api_key = "rotated"
    assert CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"
    facade.cache.similarity_threshold = 0.5
    assert CacheTestResolver(SimpleNamespace(cache=facade)).resolve().kind == "python_callback"
    unsupported: Final = qdrant_facade(qdrant_url, f"cache_{uuid4().hex}")
    unsupported.cache.embedding_max_input_tokens = 100
    with pytest.raises(TypeError, match="requires Python"):
        handle._bind_facade(unsupported)
    unsupported.cache.embedding_max_input_tokens = None
    unsupported.cache.qdrant_api_base = "http://127.0.0.1:7777"
    with pytest.raises(TypeError, match="gRPC"):
        handle._bind_facade(unsupported)


def test_qdrant_semantic_rust_required_rule_activates_natively(
    qdrant_url: str, fake_embedding_endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_embedding_endpoint
    require_rust(monkeypatch, LiteLLMCacheType.QDRANT_SEMANTIC)
    facade: Final = qdrant_facade(qdrant_url, f"cache_{uuid4().hex}")
    assert_native_runtime(facade)
    kwargs: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "qdrant activation"}]}
    facade.add_cache({"answer": "qdrant"}, **kwargs)
    assert facade.get_cache(**kwargs) == {"answer": "qdrant"}
