import asyncio
import contextvars
import hashlib
import json
import math
import os
from collections.abc import Callable, Generator
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Final, cast
from uuid import uuid4

import pytest
import redis

import litellm
from litellm.caching.caching import Cache
from litellm.caching.redis_semantic_cache import RedisSemanticCache
from litellm.types.caching import LiteLLMCacheType
from litellm.types.llms.custom_llm import CustomLLMItem
from litellm.types.utils import EmbeddingResponse
from tests.test_litellm_rust.support.cache import (
    CacheTestHandle,
    CacheTestResolver,
    assert_native_runtime,
    request,
    require_rust,
)
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


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


def semantic_facade(url: str, index: str, *, similarity_threshold: float = 0.8) -> Cache:
    facade: Final = Cache(
        type=LiteLLMCacheType.REDIS_SEMANTIC,
        redis_url=url,
        similarity_threshold=similarity_threshold,
        redis_semantic_cache_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        redis_semantic_cache_index_name=index,
    )
    CacheTestHandle.redis_semantic(facade.cache)._bind_facade(facade)
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
    assert isinstance(handle, CacheTestHandle)
    assert handle.backend == "redis_semantic"
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
    assert binding.kind == "native"


def test_redis_semantic_native_and_python_sync_entries_share_one_layout(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()

    binding.store(semantic_request("sim", "tell me a joke"), {"answer": "haha"})
    paraphrase: Final = f"tell me a joke{PARAPHRASE_MARKER}"
    assert binding.lookup(semantic_request("sim", paraphrase)) == {"answer": "haha"}
    assert binding.lookup(semantic_request("sim", "an unrelated question about spreadsheets")) is None
    assert binding.lookup(semantic_request("other-key", "tell me a joke")) is None

    strict: Final = semantic_facade(url, index, similarity_threshold=0.99)
    strict_binding: Final = CacheTestResolver(SimpleNamespace(cache=strict)).resolve()
    assert strict_binding.lookup(semantic_request("sim", paraphrase)) is None
    assert strict_binding.lookup(semantic_request("sim", "tell me a joke")) == {"answer": "haha"}


def test_redis_semantic_ttl_is_written_only_when_requested(
    redis_stack: tuple[str, str], semantic_embedding: DeterministicEmbedding
) -> None:
    url, index = redis_stack
    facade: Final = semantic_facade(url, index)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()

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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    binding: Final = CacheTestResolver(SimpleNamespace(cache=facade)).resolve()
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
    resolver: Final = CacheTestResolver(SimpleNamespace(cache=facade))
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
        CacheTestHandle.redis_semantic(object())
    with pytest.raises(TypeError, match="built-in RedisSemanticCache"):
        CacheTestHandle.redis_semantic(
            CustomSemanticCache(
                redis_url=url,
                similarity_threshold=0.8,
                embedding_model=SEMANTIC_EMBEDDING_MODEL,
                index_name=f"{index}_subclass",
            )
        )

    facade: Final = semantic_facade(url, index)
    with pytest.raises(TypeError, match="backend types must match"):
        CacheTestHandle.redis(url)._bind_facade(facade)

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
        CacheTestHandle.redis_semantic(subclassed_facade.cache)._bind_facade(subclassed_facade)

    replacement_facade: Final = Cache(
        type=LiteLLMCacheType.REDIS_SEMANTIC,
        redis_url=url,
        similarity_threshold=0.8,
        redis_semantic_cache_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        redis_semantic_cache_index_name=index,
    )
    with pytest.raises(TypeError, match="must be the native embedder"):
        CacheTestHandle.redis_semantic(facade.cache)._bind_facade(replacement_facade)


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
    assert_native_runtime(facade)
    kwargs: Final = {"model": "gpt-4o", "messages": semantic_messages("name a primary color")}
    await facade.async_add_cache({"answer": "blue"}, **kwargs)
    assert await facade.async_get_cache(**kwargs) == {"answer": "blue"}
