import asyncio
import copy
import hashlib
import json
import logging
import re
import time
import traceback
import tracemalloc
import uuid
from collections import OrderedDict, UserDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from enum import Enum
from functools import reduce
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
import respx
from opentelemetry.trace import NonRecordingSpan, SpanContext
from pydantic import BaseModel, ConfigDict

import litellm
from litellm import CustomLLM, acompletion, aembedding, completion, embedding
import litellm.caching.redis_cache as redis_cache_module
from litellm._internal_context import current_service_target
from litellm.caching.caching import Cache, CacheMode, enable_cache, response_cache_phase, update_cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.constants import (
    CACHE_WRITE_SHUTDOWN_FLUSH_TIMEOUT_SECONDS,
    CALLER_MESSAGE_COUNT_KEY,
    SEMANTIC_CACHE_POLICY_KEY,
)
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisTimeoutLogThrottle
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.caching import (
    DEFAULT_CACHING_SUPPORTED_CALL_TYPES,
    CachingSupportedCallTypes,
    CallerMessageCount,
    EMBEDDING_CACHE_FORMAT_VERSION,
    LiteLLMCacheType,
    SemanticCacheScope,
)
from litellm.types.semantic_cache import SemanticCacheParams, SemanticCachePolicy
from litellm.types.utils import Embedding, EmbeddingResponse, Usage

_CACHING_TEST_MESSAGES: Final = [{"role": "user", "content": "who is ishaan 5222"}]

messages = [{"role": "user", "content": "who is ishaan 5222"}]


@pytest.fixture
def preserve_litellm_set_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "set_verbose", litellm.set_verbose)


@pytest.fixture
def redis_client_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("redis.Redis.info", lambda *args, **kwargs: {"redis_version": "7.2.0"})
    monkeypatch.setattr("redis.Redis.ping", lambda *args, **kwargs: True)
    monkeypatch.setattr("redis.asyncio.Redis.ping", AsyncMock(return_value=True))


@pytest.fixture
def redis_response_cache_clients(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    sync_client: Final = MagicMock(info=MagicMock(return_value={"redis_version": "7.2.0"}))
    async_client: Final = MagicMock(ping=AsyncMock(return_value=True))
    monkeypatch.setattr("litellm._redis.get_redis_client", lambda **_: sync_client)
    monkeypatch.setattr("litellm._redis.get_redis_async_client", lambda **_: async_client)
    return sync_client, async_client


def test_cache_key_debug_log_does_not_include_prompt_material(caplog):
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    prompt_marker = "secret prompt material "

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        cache_key = cache.get_cache_key(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt_marker * 100},
                {"role": "user", "content": "hello"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "lookup_response",
                    "schema": {"type": "object"},
                },
            },
            stream=True,
        )

    assert re.fullmatch(r"[0-9a-f]{64}", cache_key)

    created_cache_key_logs = [
        record.getMessage() for record in caplog.records if "Created cache key:" in record.getMessage()
    ]
    assert created_cache_key_logs
    assert all(prompt_marker not in message for message in created_cache_key_logs)
    assert any(cache_key in message for message in created_cache_key_logs)


@pytest.mark.parametrize(
    ("backend", "expected_level"),
    [
        pytest.param(MagicMock(spec=RedisCache), logging.DEBUG, id="redis_backend_is_throttled"),
        pytest.param(MagicMock(), logging.ERROR, id="other_backend_logs_every_timeout"),
    ],
)
def test_add_cache_timeout_only_joins_redis_throttle_for_redis_backends(backend, expected_level, caplog, monkeypatch):
    throttle = RedisTimeoutLogThrottle(interval=5.0, clock=MagicMock(return_value=1_000.0))
    assert throttle.admit() == 0
    monkeypatch.setattr(redis_cache_module, "redis_timeout_log_throttle", throttle)

    cache = Cache(type=LiteLLMCacheType.LOCAL)
    backend.set_cache.side_effect = TimeoutError("lit7520 backend timed out")
    cache.cache = backend

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        cache.add_cache("result", model="gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}])

    records = [r for r in caplog.records if "lit7520 backend timed out" in r.getMessage()]
    assert [r.levelno for r in records] == [expected_level]


def _embedding_response(prompt_tokens, num_items):
    return EmbeddingResponse(
        model="amazon.titan-embed-image-v1",
        data=[Embedding(embedding=[0.0], index=i, object="embedding") for i in range(num_items)],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens),
    )


def test_get_per_item_prompt_tokens_single_item_returns_full_value():
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    result = _embedding_response(prompt_tokens=0, num_items=1)
    assert cache._get_per_item_prompt_tokens(result, 0) == 0


def test_get_per_item_prompt_tokens_distributes_with_remainder():
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    result = _embedding_response(prompt_tokens=10, num_items=3)
    per_item = [cache._get_per_item_prompt_tokens(result, i) for i in range(3)]
    assert sum(per_item) == 10  # 4 + 3 + 3
    assert per_item == [4, 3, 3]


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("item_usage", [[{"prompt_tokens": 1}, {"prompt_tokens": 2}], None])
async def test_an_upstream_item_usage_key_still_caches_each_embedding_input(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, item_usage: object
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    route: Final = respx_mock.post("http://upstream.test/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": index, "embedding": [0.1, 0.2]} for index in range(2)],
                "model": "m",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
                "item_usage": item_usage,
            },
        )
    )

    async def embed(input: Sequence[str]) -> EmbeddingResponse:
        return await aembedding(
            model="openai_like/m", input=input, api_base="http://upstream.test/v1", api_key="upstream-test-key"
        )

    await embed(["alpha", "beta"])
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    hit: Final = await embed(["alpha"])

    assert route.call_count == 1
    assert hit.usage.prompt_tokens == 2


def _semantic_cache(**cache_kwargs):
    return Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        host="localhost",
        port="6379",
        similarity_threshold=0.8,
        **cache_kwargs,
    )


@pytest.mark.parametrize(
    "cache_type",
    [LiteLLMCacheType.REDIS_SEMANTIC, LiteLLMCacheType.VALKEY_SEMANTIC],
)
def test_semantic_cache_embedding_max_input_tokens_reaches_backend(cache_type):
    cache = Cache(
        type=cache_type,
        redis_url="redis://localhost:6379",
        similarity_threshold=0.8,
        semantic_cache_embedding_max_input_tokens=2048,
    )
    assert cache.cache.embedding_max_input_tokens == 2048


def test_semantic_cache_key_excludes_prompt_so_paraphrases_share_a_bucket():
    cache = _semantic_cache()
    tenant = {"user_api_key": "hash-abc"}
    key_a = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What color is the sky?"}],
        metadata=dict(tenant),
    )
    key_b = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me the colour of the daytime sky."}],
        metadata=dict(tenant),
    )
    assert key_a == key_b


def test_semantic_cache_key_isolates_tenants():
    messages = [{"role": "user", "content": "What color is the sky?"}]
    cache = _semantic_cache()
    key_a = cache.get_cache_key(model="gpt-4o-mini", messages=messages, metadata={"user_api_key": "hash-A"})
    key_b = cache.get_cache_key(model="gpt-4o-mini", messages=messages, metadata={"user_api_key": "hash-B"})
    key_team = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=messages,
        metadata={"user_api_key": "hash-A", "user_api_key_team_id": "team-1"},
    )
    assert key_a != key_b
    assert key_a != key_team


_SEMANTICALLY_IDENTICAL_PROMPTS = (
    [{"role": "user", "content": "What color is the sky?"}],
    [{"role": "user", "content": "Tell me the colour of the daytime sky."}],
)


def _end_user_keys(cache, metadata_field, *end_user_ids):
    return [
        cache.get_cache_key(
            model="gpt-4o-mini",
            messages=messages,
            **{metadata_field: {"user_api_key": "hash-A", "user_api_key_end_user_id": end_user_id}},
        )
        for messages, end_user_id in zip(_SEMANTICALLY_IDENTICAL_PROMPTS, end_user_ids)
    ]


@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
def test_semantic_cache_key_shares_bucket_across_end_users_by_default(metadata_field):
    key_alice, key_bob = _end_user_keys(_semantic_cache(), metadata_field, "alice", "bob")
    assert key_alice == key_bob


@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
def test_semantic_cache_key_isolates_end_users_under_end_user_scope(metadata_field):
    cache = _semantic_cache(semantic_cache_scope="end_user")
    key_alice, key_bob = _end_user_keys(cache, metadata_field, "alice", "bob")
    key_alice_again, _ = _end_user_keys(cache, metadata_field, "alice", "alice")
    assert key_alice != key_bob
    assert key_alice == key_alice_again


def test_semantic_cache_key_end_user_scope_without_end_user_falls_back_to_key_scope():
    cache = _semantic_cache(semantic_cache_scope=SemanticCacheScope.END_USER)
    messages = [{"role": "user", "content": "What color is the sky?"}]
    key_scope_only = cache.get_cache_key(model="gpt-4o-mini", messages=messages, metadata={"user_api_key": "hash-A"})
    end_user_absent = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=messages,
        metadata={"user_api_key": "hash-A", "user_api_key_end_user_id": None},
    )
    other_key = cache.get_cache_key(model="gpt-4o-mini", messages=messages, metadata={"user_api_key": "hash-B"})
    key_alice, _ = _end_user_keys(cache, "metadata", "alice", "alice")
    default_scope_key = _semantic_cache().get_cache_key(
        model="gpt-4o-mini", messages=messages, metadata={"user_api_key": "hash-A"}
    )
    assert key_scope_only == end_user_absent == default_scope_key
    assert key_scope_only != other_key
    assert key_scope_only != key_alice


def test_semantic_cache_key_reads_tenant_identity_from_litellm_metadata():
    cache = _semantic_cache()
    messages = [{"role": "user", "content": "What color is the sky?"}]
    key_a = cache.get_cache_key(model="gpt-4o-mini", messages=messages, litellm_metadata={"user_api_key": "hash-A"})
    key_b = cache.get_cache_key(model="gpt-4o-mini", messages=messages, litellm_metadata={"user_api_key": "hash-B"})
    key_a_in_litellm_params = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=messages,
        litellm_params={"litellm_metadata": {"user_api_key": "hash-A"}},
    )
    assert key_a != key_b
    assert key_a == key_a_in_litellm_params


def test_cache_rejects_unsupported_type() -> None:
    cache_params: Final = {"type": "not-a-cache"}
    with pytest.raises(
        ValueError, match=re.escape("Unsupported cache type 'not-a-cache'. Supported types: ")
    ) as raised:
        Cache(**cache_params)
    assert str(raised.value).split("Supported types: ")[1].split(", ") == [member.value for member in LiteLLMCacheType]


class _PingableInMemoryCache(InMemoryCache):
    async def ping(self) -> bool:
        return True

    async def delete_cache_keys(self, keys: list[str]) -> None:
        for key in keys:
            self.delete_cache(key)


@pytest.mark.asyncio
async def test_cache_ping_and_delete_cache_keys_reach_a_backend_that_supports_them() -> None:
    backend: Final = _PingableInMemoryCache()
    cache: Final = Cache(type="local", _backend=backend)
    backend.set_cache("kept", "kept value")
    backend.set_cache("dropped", "dropped value")

    assert await cache.ping() is True
    await cache.delete_cache_keys(["dropped"])

    assert backend.get_cache("dropped") is None
    assert backend.get_cache("kept") == "kept value"


def test_semantic_cache_scope_rejects_unknown_value():
    with pytest.raises(ValueError, match="'team' is not a valid SemanticCacheScope"):
        _semantic_cache(semantic_cache_scope="team")


def test_semantic_cache_key_still_separates_models_and_params():
    cache = _semantic_cache()
    messages = [{"role": "user", "content": "hi"}]
    tenant = {"user_api_key": "hash-A"}
    assert cache.get_cache_key(model="gpt-4o-mini", messages=messages, metadata=dict(tenant)) != cache.get_cache_key(
        model="gpt-4o", messages=messages, metadata=dict(tenant)
    )
    assert cache.get_cache_key(
        model="gpt-4o-mini", messages=messages, temperature=0, metadata=dict(tenant)
    ) != cache.get_cache_key(model="gpt-4o-mini", messages=messages, temperature=1, metadata=dict(tenant))


def test_exact_cache_key_still_includes_prompt():
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    key_a = cache.get_cache_key(model="gpt-4o-mini", messages=[{"role": "user", "content": "a"}])
    key_b = cache.get_cache_key(model="gpt-4o-mini", messages=[{"role": "user", "content": "b"}])
    assert key_a != key_b


@pytest.mark.parametrize(
    "anthropic_param",
    [
        {"system": "answer ALPHA"},
        {"top_k": 5},
        {"stop_sequences": ["STOP"]},
    ],
)
def test_exact_cache_key_includes_anthropic_messages_params(anthropic_param):
    """Anthropic /v1/messages params with no OpenAI equivalent must still key the
    cache; without them two requests that differ only by system prompt collide."""
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    messages = [{"role": "user", "content": "which greek letter?"}]
    baseline = cache.get_cache_key(model="claude-sonnet-4-5", messages=messages)
    assert baseline != cache.get_cache_key(model="claude-sonnet-4-5", messages=messages, **anthropic_param)


@pytest.mark.parametrize(
    "summary_ask",
    [
        {"reasoning_summary": "auto"},
        {"reasoningSummary": "auto"},
        {"extra_body": {"reasoning_summary": "auto"}},
        {"extra_body": {"reasoningSummary": "auto"}},
    ],
    ids=["top_level", "top_level_camel", "extra_body", "extra_body_camel"],
)
def test_exact_cache_key_includes_the_reasoning_summary_alias(summary_ask: dict[str, object]) -> None:
    """The AI SDK summary alias decides whether thought summaries come back, wherever the request
    carries it, so a reply cached without it must not answer a request that asks for one."""
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    messages: Final = [{"role": "user", "content": "how much is the ball?"}]
    no_ask: Final = cache.get_cache_key(model="gemini/gemini-3.8-flash", messages=messages, reasoning_effort="medium")
    ask: Final = cache.get_cache_key(
        model="gemini/gemini-3.8-flash", messages=messages, reasoning_effort="medium", **summary_ask
    )
    assert no_ask != ask
    assert ask == cache.get_cache_key(
        model="gemini/gemini-3.8-flash", messages=messages, reasoning_effort="medium", **summary_ask
    )
    assert ask != cache.get_cache_key(
        model="gemini/gemini-3.8-flash",
        messages=messages,
        reasoning_effort="medium",
        reasoning_summary="detailed",
    )


@pytest.mark.asyncio
async def test_embedding_cache_skips_write_when_one_input_yields_many_embeddings(monkeypatch):
    """A cross-encoder behind /embeddings returns one score per document for a single
    input string; caching data[0] per input would make the second call return 1 score."""
    import litellm
    from litellm import CustomLLM

    class ScoreEveryDocument(CustomLLM):
        provider_calls: int = 0

        async def aembedding(self, model, input, model_response, **kwargs) -> EmbeddingResponse:
            self.provider_calls += 1
            return EmbeddingResponse(
                model=model,
                data=[Embedding(embedding=[float(i)], index=i, object="embedding") for i in range(5)],
            )

    scorer = ScoreEveryDocument()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "score-every-doc", "custom_handler": scorer}])
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list, "score-every-doc"])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "score-every-doc"])
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))

    batch = '{"query": "q", "documents": ["a", "b", "c", "d", "e"]}'
    first = await litellm.aembedding(model="score-every-doc/m", input=[batch])
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    second = await litellm.aembedding(model="score-every-doc/m", input=[batch])

    assert scorer.provider_calls == 2
    assert [len(first.data), len(second.data)] == [5, 5]


@pytest.mark.asyncio
async def test_embedding_cache_refetches_entries_written_without_format_version(monkeypatch):
    import litellm
    from litellm import CustomLLM

    class EmbedLength(CustomLLM):
        provider_calls: int = 0

        async def aembedding(self, model, input, model_response, **kwargs) -> EmbeddingResponse:
            self.provider_calls += 1
            return EmbeddingResponse(
                model=model,
                data=[
                    Embedding(embedding=[float(len(text))], index=idx, object="embedding")
                    for idx, text in enumerate(input)
                ],
            )

    embedder = EmbedLength()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "embed-length", "custom_handler": embedder}])
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list, "embed-length"])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "embed-length"])
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))

    await litellm.aembedding(model="embed-length/m", input=["abcd"])
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    store = litellm.cache.cache.cache_dict
    stored = [entry["response"] for entry in store.values()]
    assert [entry["format_version"] for entry in stored] == [EMBEDDING_CACHE_FORMAT_VERSION], stored
    legacy_store = {
        key: {
            **entry,
            "response": {
                field: value
                for field, value in {**entry["response"], "embedding": [-1.0]}.items()
                if field != "format_version"
            },
        }
        for key, entry in store.items()
    }
    monkeypatch.setattr(litellm.cache.cache, "cache_dict", legacy_store)

    refetched = await litellm.aembedding(model="embed-length/m", input=["abcd"])

    assert embedder.provider_calls == 2, "an entry written without format_version must be a cache miss"
    assert [item["embedding"] for item in refetched.data] == [[4.0]]


@pytest.mark.asyncio
async def test_embedding_cache_serves_base64_string_embeddings_on_repeat(monkeypatch):
    import litellm
    from litellm import CustomLLM

    class Base64Embedder(CustomLLM):
        provider_calls: int = 0

        async def aembedding(self, model, input, model_response, **kwargs) -> EmbeddingResponse:
            self.provider_calls += 1
            return EmbeddingResponse(
                model=model,
                data=[
                    Embedding(embedding="AACAPwAAAEA=", index=idx, object="embedding") for idx, _ in enumerate(input)
                ],
            )

    embedder = Base64Embedder()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "embed-b64", "custom_handler": embedder}])
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list, "embed-b64"])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "embed-b64"])
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))

    first = await litellm.aembedding(model="embed-b64/m", input=["abcd"])
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    second = await litellm.aembedding(model="embed-b64/m", input=["abcd"])

    assert embedder.provider_calls == 1, "a string embedding written to the cache must be served on repeat"
    assert [item["embedding"] for item in second.data] == [item["embedding"] for item in first.data] == ["AACAPwAAAEA="]


def test_provider_specific_cache_key_ignores_litellm_owned_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", True)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "hi"}], "top_k": 5}

    base_key: Final = cache.get_cache_key(**request)

    assert cache.get_cache_key(**request, _litellm_control={"stream_chunk_size": 64}) == base_key
    assert cache.get_cache_key(**request, litellm_trace_id="trace-1") == base_key
    assert cache.get_cache_key(**request, _skip_mcp_handler=True) == base_key
    assert cache.get_cache_key(**request, fake_stream=True) == base_key
    assert cache.get_cache_key(**{**request, "top_k": 6}) != base_key


def _request_span(trace_id: int) -> NonRecordingSpan:
    return NonRecordingSpan(SpanContext(trace_id=trace_id, span_id=trace_id, is_remote=False))


@pytest.mark.parametrize("provider_specific_params_keyed", [True, False])
def test_cache_key_ignores_the_per_request_otel_span(
    monkeypatch: pytest.MonkeyPatch, provider_specific_params_keyed: bool
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", provider_specific_params_keyed)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {"model": "openai/mock-model", "messages": [{"role": "user", "content": "hi"}]}
    first_span: Final = _request_span(1)
    second_span: Final = _request_span(2)
    assert repr(first_span) != repr(second_span)

    first_key: Final = cache.get_cache_key(**request, my_provider_param="a", parent_otel_span=first_span)

    assert cache.get_cache_key(**request, my_provider_param="a", parent_otel_span=second_span) == first_key
    assert cache.get_cache_key(**request, my_provider_param="a", parent_otel_span=None) == first_key
    assert (cache.get_cache_key(**request, my_provider_param="b", parent_otel_span=first_span) != first_key) is (
        provider_specific_params_keyed
    )


def test_cache_key_tells_apart_requests_that_differ_only_in_continuation_token() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {
        "model": "gemini/gemini-4-argon",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16000,
    }

    first_segment_key: Final = cache.get_cache_key(**request)
    resumed_key: Final = cache.get_cache_key(**request, continuation_token="dG9rLTE=")

    assert resumed_key != first_segment_key
    assert cache.get_cache_key(**request, continuation_token="dG9rLTI=") not in (first_segment_key, resumed_key)
    assert cache.get_cache_key(**request, continuation_token="dG9rLTE=") == resumed_key
    assert cache.get_cache_key(**request, continuation_token=None) == first_segment_key
    assert cache.get_cache_key(**request, continuation_token="") == first_segment_key


def test_cache_key_tells_apart_requests_that_ask_for_token_ids() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {"model": "hosted_vllm/qwen3", "messages": [{"role": "user", "content": "count"}]}

    plain_key: Final = cache.get_cache_key(**request)

    assert cache.get_cache_key(**request, return_token_ids=True) != plain_key
    assert cache.get_cache_key(**request, return_token_ids=False) == plain_key
    assert cache.get_cache_key(**request, return_token_ids=None) == plain_key


def test_cache_key_tells_apart_requests_that_ask_for_token_ids_through_extra_body() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {"model": "hosted_vllm/qwen3", "messages": [{"role": "user", "content": "count"}]}

    plain_key: Final = cache.get_cache_key(**request)
    other_extra_body_key: Final = cache.get_cache_key(**request, extra_body={"top_k": 5})

    assert cache.get_cache_key(**request, extra_body={"return_token_ids": True}) != plain_key
    assert cache.get_cache_key(**request, extra_body={"top_k": 5, "return_token_ids": True}) != other_extra_body_key
    assert cache.get_cache_key(**request, extra_body={"return_token_ids": False}) == plain_key
    assert cache.get_cache_key(**request, extra_body={"continuation_token": "abc"}) != plain_key


@pytest.mark.parametrize(
    "audio_format_ask",
    [
        {"responseFormat": {"audio": {"mimeType": "AUDIO_WAV"}}},
        {"generationConfig": {"responseFormat": {"audio": {"mimeType": "AUDIO_WAV"}}}},
        {"generation_config": {"response_format": {"audio": {"mime_type": "AUDIO_WAV"}}}},
        {"extra_body": {"generationConfig": {"responseFormat": {"audio": {"mimeType": "AUDIO_WAV"}}}}},
        {"extra_body": {"generation_config": {"response_format": {"audio": {"mime_type": "AUDIO_WAV"}}}}},
    ],
    ids=["responseFormat", "generationConfig", "generation_config", "extra_body-camelCase", "extra_body-snake_case"],
)
def test_cache_key_tells_apart_requests_that_name_a_gemini_audio_format(audio_format_ask: dict[str, object]) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {
        "model": "gemini/gemini-3.8-flash-tts",
        "messages": [{"role": "user", "content": "Say hello"}],
        "modalities": ["audio"],
        "audio": {"voice": "Kore"},
    }

    assert cache.get_cache_key(**request, **audio_format_ask) != cache.get_cache_key(**request)


_EXTRA_BODY_REQUEST: Final = MappingProxyType(
    {"model": "hosted_vllm/qwen3", "messages": [{"role": "user", "content": "extra-body"}]}
)
_LARGE_EXTRA_BODY_TEXT: Final = "x" * 70_000


def _as_pair_list(body: Mapping[str, object]) -> Sequence[tuple[str, object]]:
    return list(body.items())


def _as_pair_tuple(body: Mapping[str, object]) -> Sequence[tuple[str, object]]:
    return tuple(body.items())


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"chat_template_kwargs": {"enable_thinking": True}}, {"chat_template_kwargs": {"enable_thinking": False}}),
        ({"top_k": 0}, None),
        ({"top_k": None}, {}),
        ({"top_k": False}, {"top_k": 0}),
        ({"chat_template_kwargs": {}}, {}),
        ({"stop_token_ids": [1, 2]}, {"stop_token_ids": [2, 1]}),
        ({"stop_token_ids": (1,)}, {"stop_token_ids": [1]}),
        ({"guided_regex": b"a"}, {"guided_regex": b"b"}),
        ({"prompt": _LARGE_EXTRA_BODY_TEXT + "a"}, {"prompt": _LARGE_EXTRA_BODY_TEXT + "b"}),
        ({"top_k": 5, "return_token_ids": True}, {"return_token_ids": True}),
        ({"top_k": 5, "return_token_ids": False}, None),
    ],
    ids=[
        "nested_bool",
        "zero",
        "null",
        "false_vs_zero",
        "empty_nested_dict",
        "list_order",
        "tuple_vs_list",
        "bytes",
        "large_string_tail",
        "beside_token_ids",
        "beside_falsy_token_ids",
    ],
)
def test_default_cache_key_tells_apart_requests_whose_extra_body_differs(
    monkeypatch: pytest.MonkeyPatch, first: object, second: object
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", False)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)

    first_key: Final = cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=first)

    assert first_key != cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=second)
    assert first_key == cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=copy.deepcopy(first))


@pytest.mark.parametrize(
    "extra_body",
    [
        None,
        {},
        "top_k=5",
        [],
        (),
        [("return_token_ids", False), ("continuation_token", "")],
        [("top_k",)],
        [("top_k", 5, 6)],
        [(5, "top_k")],
        ["a"],
        ["abc"],
        [("top_k", 5), ("min_p",)],
        {"return_token_ids": False},
        {"continuation_token": ""},
        {"return_token_ids": None, "continuation_token": ""},
    ],
    ids=[
        "none",
        "empty",
        "string",
        "empty_pair_list",
        "empty_pair_tuple",
        "falsy_token_id_pairs",
        "short_pair",
        "long_pair",
        "non_string_pair_key",
        "short_string_item",
        "long_string_item",
        "one_malformed_pair",
        "falsy_token_ids",
        "empty_continuation",
        "falsy_legacy_fields",
    ],
)
def test_default_cache_key_ignores_extra_body_that_adds_nothing_to_the_request(
    monkeypatch: pytest.MonkeyPatch, extra_body: object
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", False)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)

    assert cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=extra_body) == cache.get_cache_key(
        **_EXTRA_BODY_REQUEST
    )


@pytest.mark.parametrize(
    ("provider_specific_params_keyed", "extra_body", "extra_body_preimage"),
    [
        (False, {"return_token_ids": True}, "extra_body.return_token_ids: True"),
        (False, {"continuation_token": "dG9r", "return_token_ids": False}, "extra_body.continuation_token: dG9r"),
        (False, {"return_token_ids": True, "top_k": 5}, "extra_body: {'top_k': 5}extra_body.return_token_ids: True"),
        (True, {}, "extra_body: {}"),
        (True, {"return_token_ids": False}, "extra_body: {'return_token_ids': False}"),
        (
            True,
            {"return_token_ids": True, "top_k": 5},
            "extra_body: {'return_token_ids': True, 'top_k': 5}extra_body.return_token_ids: True",
        ),
        (True, "top_k=5", "extra_body: top_k=5"),
        (
            True,
            MappingProxyType({"return_token_ids": True, "top_k": 5}),
            "extra_body: {'return_token_ids': True, 'top_k': 5}",
        ),
        (True, UserDict({"return_token_ids": True, "top_k": 5}), "extra_body: {'return_token_ids': True, 'top_k': 5}"),
        (
            False,
            MappingProxyType({"return_token_ids": True, "top_k": 5}),
            "extra_body: {'return_token_ids': True, 'top_k': 5}",
        ),
        (False, [("return_token_ids", True), ("top_k", 5)], "extra_body: {'return_token_ids': True, 'top_k': 5}"),
        (True, [("return_token_ids", True), ("top_k", 5)], "extra_body: [('return_token_ids', True), ('top_k', 5)]"),
    ],
    ids=[
        "default_token_ids",
        "default_continuation",
        "default_beside_token_ids",
        "flag_empty",
        "flag_falsy_token_ids",
        "flag_beside_token_ids",
        "flag_string",
        "flag_mapping_proxy",
        "flag_user_dict",
        "default_mapping_proxy_beside_token_ids",
        "default_pairs_beside_token_ids",
        "flag_pairs",
    ],
)
def test_extra_body_cache_key_keeps_its_preimage(
    monkeypatch: pytest.MonkeyPatch, provider_specific_params_keyed: bool, extra_body: object, extra_body_preimage: str
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", provider_specific_params_keyed)
    expected: Final = hashlib.sha256(f"model: mmessages: hi{extra_body_preimage}".encode()).hexdigest()

    assert Cache(type=LiteLLMCacheType.LOCAL).get_cache_key(model="m", messages="hi", extra_body=extra_body) == expected


@pytest.mark.parametrize("mapping_type", [MappingProxyType, UserDict])
def test_default_cache_key_reads_extra_body_from_any_mapping(
    monkeypatch: pytest.MonkeyPatch, mapping_type: Callable[[Mapping[str, object]], Mapping[str, object]]
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", False)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    thinking: Final = {"chat_template_kwargs": {"enable_thinking": True}}
    no_thinking: Final = {"chat_template_kwargs": {"enable_thinking": False}}

    def key_of(extra_body: Mapping[str, object] | None) -> str:
        return cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=extra_body)

    assert key_of(mapping_type(thinking)) == key_of(thinking)
    assert key_of(mapping_type(no_thinking)) == key_of(no_thinking)
    assert key_of(mapping_type(thinking)) != key_of(mapping_type(no_thinking))
    assert key_of(mapping_type({})) == key_of(None)
    assert key_of(mapping_type({"return_token_ids": False, "continuation_token": ""})) == key_of(None)
    assert key_of(mapping_type({"return_token_ids": True})) != key_of(None)
    assert key_of(mapping_type({"continuation_token": "a"})) != key_of(mapping_type({"continuation_token": "b"}))


@pytest.mark.parametrize("pair_sequence", [list, tuple])
def test_default_cache_key_reads_extra_body_given_as_key_value_pairs(
    monkeypatch: pytest.MonkeyPatch, pair_sequence: Callable[[Iterable[Sequence[object]]], Sequence[Sequence[object]]]
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", False)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    thinking_pairs: Final = pair_sequence([("chat_template_kwargs", {"enable_thinking": True})])
    sent_pairs: Final = copy.deepcopy(thinking_pairs)

    def key_of(extra_body: object) -> str:
        return cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=extra_body)

    assert key_of(thinking_pairs) == key_of({"chat_template_kwargs": {"enable_thinking": True}})
    assert key_of(thinking_pairs) != key_of(pair_sequence([("chat_template_kwargs", {"enable_thinking": False})]))
    assert thinking_pairs == sent_pairs
    assert key_of(pair_sequence([("top_k", 1), ("top_k", 5)])) == key_of({"top_k": 5})
    assert key_of(pair_sequence([("top_k", 1), ("top_k", 5)])) != key_of({"top_k": 1})
    assert key_of(pair_sequence([("return_token_ids", True)])) != key_of(None)
    assert key_of(pair_sequence([["top_k", 5]])) == key_of({"top_k": 5})
    assert key_of(pair_sequence(["ab"])) == key_of({"a": "b"})
    assert key_of(pair_sequence(["ab"])) != key_of(pair_sequence(["ac"]))


def test_default_cache_key_leaves_one_shot_extra_body_iterators_unread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", False)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    pairs: Final = iter([("top_k", 5)])

    assert cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=pairs) == cache.get_cache_key(**_EXTRA_BODY_REQUEST)
    assert next(pairs) == ("top_k", 5)


def test_extra_body_keeps_preset_keys_and_namespaces_in_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", False)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    extra_body: Final = {"chat_template_kwargs": {"enable_thinking": False}}

    unscoped_key: Final = cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=extra_body)

    assert (
        cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=extra_body, litellm_params={"preset_cache_key": "preset"})
        == "preset"
    )
    assert (
        cache.get_cache_key(**_EXTRA_BODY_REQUEST, extra_body=extra_body, cache={"namespace": "team-a"})
        == f"team-a:{unscoped_key}"
    )


def test_semantic_cache_scope_separates_extra_body_but_still_ignores_the_prompt() -> None:
    cache: Final = _semantic_cache()
    sky, colour = _SEMANTICALLY_IDENTICAL_PROMPTS

    def scope_key(messages: object, api_key: str, enable_thinking: bool) -> str:
        return cache.get_cache_key(
            model="gpt-4o-mini",
            messages=messages,
            metadata={"user_api_key": api_key},
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )

    assert scope_key(sky, "hash-A", True) == scope_key(colour, "hash-A", True)
    assert scope_key(sky, "hash-A", True) != scope_key(colour, "hash-A", False)
    assert scope_key(sky, "hash-A", True) != scope_key(colour, "hash-B", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_specific_params_keyed", [False, True])
@pytest.mark.parametrize("extra_body_type", [dict, MappingProxyType, UserDict, _as_pair_list, _as_pair_tuple])
async def test_response_cache_answers_each_extra_body_with_its_own_reply(
    monkeypatch: pytest.MonkeyPatch,
    provider_specific_params_keyed: bool,
    extra_body_type: Callable[[Mapping[str, object]], object],
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", provider_specific_params_keyed)
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))

    async def ask(enable_thinking: bool, mock_response: str) -> tuple[str, object]:
        response: Final = await acompletion(
            model="hosted_vllm/qwen",
            api_base="http://localhost:8000",
            messages=[{"role": "user", "content": "extra-body"}],
            extra_body=extra_body_type({"chat_template_kwargs": {"enable_thinking": enable_thinking}}),
            mock_response=mock_response,
        )
        await asyncio.gather(*_PENDING_CACHE_WRITES)
        assert isinstance(response, litellm.ModelResponse), response
        return _completion_text(response), response._hidden_params.get("cache_hit")

    replies: Final = (
        await ask(True, "WITH-THINKING"),
        await ask(False, "NO-THINKING"),
        await ask(True, "not served, cached reply expected"),
        await ask(False, "not served, cached reply expected"),
    )

    assert replies == (
        ("WITH-THINKING", None),
        ("NO-THINKING", None),
        ("WITH-THINKING", True),
        ("NO-THINKING", True),
    )


def test_cache_key_tells_apart_requests_that_differ_only_in_tool_call_id_shape() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {
        "model": "gemini/gemini-3.5-flash",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    }

    signed_ids_key: Final = cache.get_cache_key(**request)
    short_ids_key: Final = cache.get_cache_key(**request, thought_signature_in_tool_call_id=False)

    assert short_ids_key != signed_ids_key
    assert cache.get_cache_key(**request, thought_signature_in_tool_call_id="false") == short_ids_key
    assert cache.get_cache_key(**request, thought_signature_in_tool_call_id=0) == short_ids_key
    assert cache.get_cache_key(**request, thought_signature_in_tool_call_id=True) == signed_ids_key
    assert cache.get_cache_key(**request, thought_signature_in_tool_call_id=None) == signed_ids_key


class PhaseRecordingCache(InMemoryCache):
    """Records the target and the active span each read / write ran under, as a Redis span would."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str | None, str]] = []

    def _record(self) -> None:
        from opentelemetry import trace

        span = trace.get_current_span()
        self.seen.append((current_service_target(), getattr(span, "name", "")))

    def get_cache(self, key, **kwargs):
        self._record()
        return super().get_cache(key, **kwargs)

    def set_cache(self, key, value, **kwargs):
        self._record()
        super().set_cache(key, value, **kwargs)


@pytest.fixture
def v2_span_exporter(monkeypatch):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from litellm.integrations.otel import OpenTelemetryV2Config
    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.integrations.otel.plumbing import providers
    from litellm.proxy import proxy_server

    config = OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    logger = OpenTelemetryV2(config=config, tracer_provider=providers.build_tracer_provider(config, exporter=exporter))
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", logger)
    return exporter


_REQUEST: Final = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "phase me"}]}


@pytest.mark.asyncio
async def test_facade_lookup_and_store_run_inside_the_response_cache_phases(v2_span_exporter):
    """The native bridge calls ``Cache.async_get_cache`` / ``async_add_cache`` straight, never through
    ``caching_handler``, so the ``cache.get llm_response`` / ``cache.set llm_response`` phase and the
    ``llm_response`` target come from the facade: the store runs under them too, and a hit reads back."""
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    backend = PhaseRecordingCache()
    assert await cache.async_get_cache(dynamic_cache_object=backend, **_REQUEST) is None
    await cache.async_add_cache({"id": "resp-1"}, dynamic_cache_object=backend, **_REQUEST)
    assert await cache.async_get_cache(dynamic_cache_object=backend, **_REQUEST) == {"id": "resp-1"}
    assert backend.seen == [
        ("llm_response", "cache.get llm_response"),
        ("llm_response", "cache.set llm_response"),
        ("llm_response", "cache.get llm_response"),
    ]
    assert [s.name for s in v2_span_exporter.get_finished_spans()] == [
        "cache.get llm_response",
        "cache.set llm_response",
        "cache.get llm_response",
    ]
    assert current_service_target() is None


def test_sync_facade_lookup_and_store_run_inside_the_response_cache_phases(v2_span_exporter):
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    backend = PhaseRecordingCache()
    assert cache.get_cache(dynamic_cache_object=backend, **_REQUEST) is None
    cache.add_cache({"id": "resp-1"}, **_REQUEST)
    assert backend.seen == [("llm_response", "cache.get llm_response")]
    assert [s.name for s in v2_span_exporter.get_finished_spans()] == [
        "cache.get llm_response",
        "cache.set llm_response",
    ]


@pytest.mark.asyncio
async def test_a_lookup_already_inside_the_phase_does_not_open_a_second_one(v2_span_exporter):
    """``caching_handler`` opens the phase around the facade call; the facade joins it."""
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    backend = PhaseRecordingCache()
    with response_cache_phase("get"):
        await cache.async_get_cache(dynamic_cache_object=backend, **_REQUEST)
    assert backend.seen == [("llm_response", "cache.get llm_response")]
    assert [s.name for s in v2_span_exporter.get_finished_spans()] == ["cache.get llm_response"]


def test_cache_override():
    # test if we can override the cache, when `caching=False` but litellm.cache = Cache() is set
    # in this case it should not return cached responses
    litellm.cache = Cache()
    print("Testing cache override")
    litellm.set_verbose = True

    # test embedding
    response1 = embedding(
        model="text-embedding-ada-002",
        input=["hello who are you"],
        caching=False,
        mock_response="0.1,0.2,0.3,0.4,0.5",
    )

    response2 = embedding(
        model="text-embedding-ada-002",
        input=["hello who are you"],
        caching=False,
        mock_response="0.6,0.7,0.8,0.9,1.0",
    )

    # When caching=False, responses should have different IDs
    assert response1.data[0].embedding != response2.data[0].embedding


def test_caching_v2():  # test in memory cache
    try:
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            caching=True,
            mock_response="Hello world from cache test",
        )
        response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        litellm.success_callback = []
        litellm._async_success_callback = []
        if response2["choices"][0]["message"]["content"] != response1["choices"][0]["message"]["content"]:
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            pytest.fail(f"Error occurred:")
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_with_ttl():
    try:
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            caching=True,
            ttl=0,
            mock_response="Hello world from cache test 1",
        )
        response2 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            caching=True,
            mock_response="Hello world from cache test 2",
        )
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        litellm.success_callback = []
        litellm._async_success_callback = []
        assert response2["choices"][0]["message"]["content"] != response1["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_with_default_ttl():
    try:
        litellm.set_verbose = True
        litellm.cache = Cache(ttl=0)
        response1 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            caching=True,
            mock_response="Hello world from cache test",
        )
        response2 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            caching=True,
            mock_response="Hello world from cache test",
        )
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        litellm.success_callback = []
        litellm._async_success_callback = []
        assert response2["id"] != response1["id"]
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_with_models_v2():
    messages = [{"role": "user", "content": "who is ishaan CTO of litellm from litellm 2023"}]
    litellm.cache = Cache()
    print("test2 for caching")
    litellm.set_verbose = True
    response1 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        caching=True,
        mock_response="Hello world from cache test",
    )
    response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response3 = completion(
        model="gpt-4.1-nano",
        messages=messages,
        caching=True,
        mock_response="Different model response",
    )
    print(f"response1: {response1}")
    print(f"response2: {response2}")
    print(f"response3: {response3}")
    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []
    if response3["choices"][0]["message"]["content"] == response2["choices"][0]["message"]["content"]:
        # if models are different, it should not return cached response
        print(f"response2: {response2}")
        print(f"response3: {response3}")
        pytest.fail(f"Error occurred:")
    if response1["choices"][0]["message"]["content"] != response2["choices"][0]["message"]["content"]:
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Error occurred:")


@pytest.mark.asyncio
async def test_dual_cache_caching_batch_get_cache():
    """
    - check redis cache called for initial batch get cache
    - check redis cache not called for consecutive batch get cache with same keys
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.redis_cache import RedisCache

    dc = DualCache(redis_cache=MagicMock(spec=RedisCache))

    with patch.object(
        dc.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(return_value={"test_key1": "test_value1", "test_key2": "test_value2"}),
    ) as mock_async_get_cache:
        await dc.async_batch_get_cache(keys=["test_key1", "test_key2"])

        assert mock_async_get_cache.call_count == 1

        await dc.async_batch_get_cache(keys=["test_key1", "test_key2"])

        assert mock_async_get_cache.call_count == 1


def test_get_cache_key():
    from litellm.caching.caching import Cache

    try:
        print("Testing get_cache_key")
        cache_instance = Cache()
        cache_key = cache_instance.get_cache_key(
            **{
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "write a one sentence poem about: 7510"}],
                "max_tokens": 40,
                "temperature": 0.2,
                "stream": True,
                "litellm_call_id": "ffe75e7e-8a07-431f-9a74-71a5b9f35f0b",
                "litellm_logging_obj": {},
            }
        )
        cache_key_2 = cache_instance.get_cache_key(
            **{
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "write a one sentence poem about: 7510"}],
                "max_tokens": 40,
                "temperature": 0.2,
                "stream": True,
                "litellm_call_id": "ffe75e7e-8a07-431f-9a74-71a5b9f35f0b",
                "litellm_logging_obj": {},
            }
        )
        cache_key_str = "model: gpt-3.5-turbomessages: [{'role': 'user', 'content': 'write a one sentence poem about: 7510'}]max_tokens: 40temperature: 0.2stream: True"
        hash_object = hashlib.sha256(cache_key_str.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        assert cache_key == hash_hex
        assert cache_key_2 == hash_hex, (
            f"{cache_key} != {cache_key_2}. The same kwargs should have the same cache key across runs"
        )

        embedding_cache_key = cache_instance.get_cache_key(
            **{
                "model": "azure/text-embedding-ada-002",
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "api_key": "",
                "api_version": "2023-07-01-preview",
                "timeout": None,
                "max_retries": 0,
                "input": ["hi who is ishaan"],
                "caching": True,
                "client": "<openai.lib.azure.AsyncAzureOpenAI object at 0x12b6a1060>",
            }
        )

        print(embedding_cache_key)

        embedding_cache_key_str = "model: azure/text-embedding-ada-002input: ['hi who is ishaan']"
        hash_object = hashlib.sha256(embedding_cache_key_str.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        assert embedding_cache_key == hash_hex, (
            f"{embedding_cache_key} != 'model: azure/text-embedding-ada-002input: ['hi who is ishaan']'. The same kwargs should have the same cache key across runs"
        )

        # Proxy - embedding cache, test if embedding key, gets model_group and not model
        embedding_cache_key_2 = cache_instance.get_cache_key(
            **{
                "model": "azure/text-embedding-ada-002",
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "api_key": "",
                "api_version": "2023-07-01-preview",
                "timeout": None,
                "max_retries": 0,
                "input": ["hi who is ishaan"],
                "caching": True,
                "client": "<openai.lib.azure.AsyncAzureOpenAI object at 0x12b6a1060>",
                "proxy_server_request": {
                    "url": "http://0.0.0.0:8000/embeddings",
                    "method": "POST",
                    "headers": {
                        "host": "0.0.0.0:8000",
                        "user-agent": "curl/7.88.1",
                        "accept": "*/*",
                        "content-type": "application/json",
                        "content-length": "80",
                    },
                    "body": {
                        "model": "azure-embedding-model",
                        "input": ["hi who is ishaan"],
                    },
                },
                "user": None,
                "metadata": {
                    "user_api_key": None,
                    "headers": {
                        "host": "0.0.0.0:8000",
                        "user-agent": "curl/7.88.1",
                        "accept": "*/*",
                        "content-type": "application/json",
                        "content-length": "80",
                    },
                    "model_group": "EMBEDDING_MODEL_GROUP",
                    "deployment": "azure/text-embedding-ada-002-ModelID-azure/text-embedding-ada-002https://openai-gpt-4-test-v-1.openai.azure.com/2023-07-01-preview",
                },
                "model_info": {
                    "mode": "embedding",
                    "base_model": "text-embedding-ada-002",
                    "id": "20b2b515-f151-4dd5-a74f-2231e2f54e29",
                },
                "litellm_call_id": "2642e009-b3cd-443d-b5dd-bb7d56123b0e",
                "litellm_logging_obj": "<litellm.utils.Logging object at 0x12f1bddb0>",
            }
        )

        print(embedding_cache_key_2)
        embedding_cache_key_str_2 = "model: EMBEDDING_MODEL_GROUPinput: ['hi who is ishaan']"
        hash_object = hashlib.sha256(embedding_cache_key_str_2.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        assert embedding_cache_key_2 == hash_hex
        print("passed!")
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred:", e)


def test_redis_caching_multiple_namespaces():
    """
    Test that redis caching works with multiple namespaces

    If client side request specifies a namespace, it should be used for caching

    The same request with different namespaces should not be cached under the same key
    """
    from unittest.mock import MagicMock, patch

    import litellm
    from litellm import completion
    from litellm._uuid import uuid
    from litellm.caching import Cache

    # Use a fixed uuid to ensure consistent cache keys
    test_uuid = "12345678-1234-1234-1234-123456789abc"
    messages = [{"role": "user", "content": f"what is litellm? {test_uuid}"}]

    # Mock the Redis client creation from the _redis module
    with (
        patch("litellm._redis.get_redis_client") as mock_get_redis_client,
        patch("litellm._redis.get_redis_connection_pool") as mock_get_redis_connection_pool,
    ):
        # Create a mock Redis client that simulates real Redis behavior
        mock_redis_client = MagicMock()
        mock_get_redis_client.return_value = mock_redis_client

        # Mock the connection pool
        mock_connection_pool = MagicMock()
        mock_get_redis_connection_pool.return_value = mock_connection_pool

        # Dictionary to simulate Redis storage with namespace support
        redis_storage = {}

        def mock_redis_get(key):
            print(f"Redis GET: {key}")
            value = redis_storage.get(key, None)
            # Convert to bytes to match real Redis behavior
            if value is not None:
                import json

                return json.dumps(value).encode("utf-8")
            return None

        def mock_redis_set(name, value, ex=None, **kwargs):
            print(f"Redis SET: {name} = {value}")
            redis_storage[name] = value
            return True

        def mock_redis_ping():
            return True

        def mock_redis_info():
            return {"redis_version": "7.0.0"}

        mock_redis_client.get = mock_redis_get
        mock_redis_client.set = mock_redis_set
        mock_redis_client.ping = mock_redis_ping
        mock_redis_client.info = mock_redis_info

        # Initialize the cache
        litellm.cache = Cache(type="redis")

        namespace_1 = "org-id1"
        namespace_2 = "org-id2"

        # Use mock_response to ensure deterministic responses without external API calls
        response_1 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            cache={"namespace": namespace_1},
            mock_response="Response for namespace 1",
        )

        response_2 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            cache={"namespace": namespace_2},
            mock_response="Response for namespace 2",
        )

        response_3 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            cache={"namespace": namespace_1},
            mock_response="This should be cached",
        )

        response_4 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            mock_response="Response without namespace",
        )

        print(f"Response 1 type: {type(response_1)} - ID: {getattr(response_1, 'id', 'N/A')}")
        print(f"Response 2 type: {type(response_2)} - ID: {getattr(response_2, 'id', 'N/A')}")
        print(f"Response 3 type: {type(response_3)} - Cache hit: {isinstance(response_3, str)}")
        print(f"Response 4 type: {type(response_4)} - ID: {getattr(response_4, 'id', 'N/A')}")

        print(f"Redis storage keys: {list(redis_storage.keys())}")

        # Verify that different namespaces created different cache keys
        cache_keys = list(redis_storage.keys())
        namespace_1_keys = [k for k in cache_keys if k.startswith(f"{namespace_1}:")]
        namespace_2_keys = [k for k in cache_keys if k.startswith(f"{namespace_2}:")]
        no_namespace_keys = [
            k for k in cache_keys if not k.startswith(f"{namespace_1}:") and not k.startswith(f"{namespace_2}:")
        ]

        print(f"Namespace 1 keys: {namespace_1_keys}")
        print(f"Namespace 2 keys: {namespace_2_keys}")
        print(f"No namespace keys: {no_namespace_keys}")

        # Should have at least one key for each namespace
        assert len(namespace_1_keys) > 0, "Should have cache keys for namespace 1"
        assert len(namespace_2_keys) > 0, "Should have cache keys for namespace 2"
        assert len(no_namespace_keys) > 0, "Should have cache keys for no namespace"

        # The main test: response 3 should be a cache hit (string) because it uses same namespace as response 1
        assert isinstance(response_3, str), "Response 3 should be a cache hit (string) for same namespace"

        # response 1 & 2 should be ModelResponse objects (cache misses)
        assert hasattr(response_1, "id"), "Response 1 should be a ModelResponse object"
        assert hasattr(response_2, "id"), "Response 2 should be a ModelResponse object"
        assert hasattr(response_4, "id"), "Response 4 should be a ModelResponse object"

        # response 1 & 2 should have different IDs (different namespaces)
        assert response_1.id != response_2.id, (
            f"Expected different response ID for different namespace. Got {response_1.id} and {response_2.id}"
        )

        # response 1 & 4 should have different IDs (different namespaces)
        assert response_1.id != response_4.id, (
            f"Expected different response ID for no namespace vs namespaced. Got {response_1.id} and {response_4.id}"
        )


_TOOL_TURN_ITEM: Final = {"role": "user", "content": "hi"}
_FILE_BLOCK_ITEM: Final = {"type": "file", "file": {"file_data": "data:video/mp4;base64,AAAA", "format": "video/mp4"}}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({"messages": [_TOOL_TURN_ITEM] * 4}, True, id="four-messages-are-cached"),
        pytest.param({"messages": [_TOOL_TURN_ITEM] * 5}, False, id="five-messages-skip-the-cache"),
        pytest.param({"input": [_TOOL_TURN_ITEM] * 4}, True, id="four-responses-items-are-cached"),
        pytest.param({"input": [_TOOL_TURN_ITEM] * 5}, False, id="five-responses-items-skip-the-cache"),
        pytest.param({"input": "one prompt"}, True, id="string-input-is-one-message"),
        pytest.param({"input": ["a", "b", "c", "d", "e"]}, True, id="embedding-strings-are-not-messages"),
        pytest.param({"input": [_FILE_BLOCK_ITEM] * 5}, True, id="embedding-file-blocks-are-not-messages"),
        pytest.param(
            {"input": [{"type": "text", "text": "chunk"}] * 5}, True, id="embedding-text-blocks-are-not-messages"
        ),
        pytest.param(
            {"input": [{"type": ["message"], "content": "hi"}] * 5}, False, id="unhashable-item-types-still-count"
        ),
        pytest.param(
            {"input": [{"type": "message", "role": "user", "content": "hi"}] * 5},
            False,
            id="five-typed-responses-messages-skip-the-cache",
        ),
    ],
)
def test_should_use_cache_stops_past_the_default_max_messages(kwargs: dict[str, object], expected: bool) -> None:
    assert Cache(type=LiteLLMCacheType.LOCAL).should_use_cache(**kwargs) is expected


def test_responses_sdk_items_count_toward_max_messages() -> None:
    from openai.types.responses import ResponseFunctionToolCall

    call: Final = ResponseFunctionToolCall(type="function_call", call_id="c1", name="ls", arguments="{}")

    assert Cache(type=LiteLLMCacheType.LOCAL).should_use_cache(input=[_TOOL_TURN_ITEM, call, call, call, call]) is False


def test_max_messages_is_configurable_and_none_disables_it() -> None:
    three: Final = [_TOOL_TURN_ITEM] * 3

    assert Cache(type=LiteLLMCacheType.LOCAL, max_messages=2).should_use_cache(messages=three) is False
    assert Cache(type=LiteLLMCacheType.LOCAL, max_messages=3).should_use_cache(messages=three) is True
    assert Cache(type=LiteLLMCacheType.LOCAL, max_messages=None).should_use_cache(messages=three * 50) is True


def test_max_messages_beats_an_explicit_use_cache_opt_in() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, mode=CacheMode.default_off)

    assert cache.should_use_cache(messages=[_TOOL_TURN_ITEM] * 4, cache={"use-cache": True}) is True
    assert cache.should_use_cache(messages=[_TOOL_TURN_ITEM] * 5, cache={"use-cache": True}) is False


def test_completion_past_max_messages_is_neither_served_from_nor_written_to_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    tag: Final = uuid.uuid4().hex
    four: Final = [{"role": "user", "content": f"{tag} turn {index}"} for index in range(4)]
    five: Final = [*four, {"role": "user", "content": f"{tag} turn 4"}]

    def answer(messages: list[dict[str, str]], mock_response: str) -> str:
        response: Final = litellm.completion(model="gpt-4o-mini", messages=messages, mock_response=mock_response)
        assert isinstance(response, litellm.ModelResponse), response
        choice: Final = response.choices[0]
        assert isinstance(choice, litellm.Choices), choice
        return str(choice.message.content)

    assert answer(four, "four first") == "four first"
    assert answer(four, "four second") == "four first", "a 4-message repeat missed the cache"
    assert answer(five, "five first") == "five first"
    assert answer(five, "five second") == "five second", "a 5-message repeat was served from the cache"


@pytest.mark.parametrize(
    ("controls", "messages", "expected"),
    [
        pytest.param(
            {CALLER_MESSAGE_COUNT_KEY: CallerMessageCount(count=4)},
            [_TOOL_TURN_ITEM] * 5,
            True,
            id="a-trusted-count-under-the-limit-is-cached",
        ),
        pytest.param(
            {CALLER_MESSAGE_COUNT_KEY: CallerMessageCount(count=5)},
            [_TOOL_TURN_ITEM],
            False,
            id="a-trusted-count-past-the-limit-skips-the-cache",
        ),
        pytest.param({CALLER_MESSAGE_COUNT_KEY: 0}, [_TOOL_TURN_ITEM] * 5, False, id="a-forged-int-is-ignored"),
        pytest.param(
            {CALLER_MESSAGE_COUNT_KEY: {"count": 0}}, [_TOOL_TURN_ITEM] * 5, False, id="a-forged-dict-is-ignored"
        ),
    ],
)
def test_should_use_cache_counts_a_trusted_caller_count_and_ignores_forged_ones(
    controls: dict[str, object], messages: list[dict[str, str]], expected: bool
) -> None:
    assert Cache(type=LiteLLMCacheType.LOCAL).should_use_cache(messages=messages, cache=copy.deepcopy(controls)) is (
        expected
    )


def test_a_stamped_caller_count_leaves_the_cache_key_unchanged() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {"model": "gpt-4o-mini", "messages": [_TOOL_TURN_ITEM]}

    assert cache.get_cache_key(**request, cache={CALLER_MESSAGE_COUNT_KEY: CallerMessageCount(count=1)}) == (
        cache.get_cache_key(**request)
    )


_GEMINI_REPLY: Final = MappingProxyType(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }
)
_FIVE_PLAIN_TURNS: Final = tuple(
    MappingProxyType({"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"})
    for index in range(5)
)


def _tool_turn(tool_results: int) -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "weather in each city"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"call_{index}", "name": "get_weather", "input": {"city": f"c{index}"}}
                for index in range(tool_results)
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"call_{index}", "content": "sunny"}
                for index in range(tool_results)
            ],
        },
    ]


def _five_plain_turns() -> list[dict[str, object]]:
    return [dict(turn) for turn in _FIVE_PLAIN_TURNS]


@pytest.fixture
def gemini_route(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, fake_provider_credentials: None
) -> respx.Route:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/[^/]+/models/gemini-3\.8-flash:generateContent.*"
    ).mock(return_value=httpx.Response(200, json=dict(_GEMINI_REPLY)))


def _send_messages_through_the_chat_bridge(
    messages: list[dict[str, object]], cache: Mapping[str, object] | None = None, system: str | None = None
) -> None:
    litellm.anthropic.messages.create(
        model="gemini/gemini-3.8-flash",
        messages=messages,
        max_tokens=16,
        **({} if cache is None else {"cache": cache}),
        **({} if system is None else {"system": system}),
    )


@pytest.mark.parametrize(
    ("max_messages", "messages", "system", "provider_calls"),
    [
        pytest.param(4, _tool_turn(3), None, 1, id="three-messages-with-three-tool-results-are-cached"),
        pytest.param(4, _tool_turn(2), None, 1, id="three-messages-with-two-tool-results-are-cached"),
        pytest.param(4, _tool_turn(2), "Answer briefly", 1, id="a-separate-system-prompt-is-not-a-caller-message"),
        pytest.param(3, _tool_turn(3), None, 1, id="a-custom-limit-counts-the-caller-messages"),
        pytest.param(2, _tool_turn(3), None, 2, id="a-custom-limit-below-the-caller-messages-skips-the-cache"),
        pytest.param(0, _tool_turn(3), None, 2, id="a-zero-limit-skips-the-cache"),
        pytest.param(None, _tool_turn(3), None, 1, id="no-limit-caches"),
        pytest.param(4, _five_plain_turns(), None, 2, id="five-caller-messages-skip-the-cache"),
    ],
)
def test_sync_messages_chat_bridge_counts_the_caller_messages_against_max_messages(
    monkeypatch: pytest.MonkeyPatch,
    gemini_route: respx.Route,
    max_messages: int | None,
    messages: list[dict[str, object]],
    system: str | None,
    provider_calls: int,
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL, max_messages=max_messages))

    _send_messages_through_the_chat_bridge(messages, system=system)
    _send_messages_through_the_chat_bridge(messages, system=system)

    assert gemini_route.call_count == provider_calls


@pytest.mark.parametrize(
    "controls",
    [
        pytest.param(None, id="no-controls"),
        pytest.param({CALLER_MESSAGE_COUNT_KEY: 0}, id="a-forged-int"),
        pytest.param({CALLER_MESSAGE_COUNT_KEY: {"count": 0}}, id="a-forged-dict"),
    ],
)
def test_sync_messages_chat_bridge_neither_reads_nor_writes_past_max_messages(
    monkeypatch: pytest.MonkeyPatch, gemini_route: respx.Route, controls: dict[str, object] | None
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", cache)
    sent_controls: Final = copy.deepcopy(controls)

    _send_messages_through_the_chat_bridge(_five_plain_turns(), cache=controls)
    _send_messages_through_the_chat_bridge(_five_plain_turns(), cache=controls)
    cache.max_messages = None
    _send_messages_through_the_chat_bridge(_five_plain_turns())

    assert gemini_route.call_count == 3, "a request past max_messages was served from or written to the cache"
    assert controls == sent_controls


def test_sync_messages_chat_bridge_keeps_the_caller_cache_controls(
    monkeypatch: pytest.MonkeyPatch, gemini_route: respx.Route
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    controls: Final = {"no-cache": True}

    _send_messages_through_the_chat_bridge(_tool_turn(3), cache=controls)
    _send_messages_through_the_chat_bridge(_tool_turn(3), cache=controls)
    _send_messages_through_the_chat_bridge(_tool_turn(3))

    assert gemini_route.call_count == 2, "no-cache was dropped, or the response was never written"
    assert controls == {"no-cache": True}


def test_sync_messages_chat_bridge_keeps_a_count_an_outer_bridge_stamped(
    monkeypatch: pytest.MonkeyPatch, gemini_route: respx.Route
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    outer_count: Final = {CALLER_MESSAGE_COUNT_KEY: CallerMessageCount(count=5)}

    _send_messages_through_the_chat_bridge(_tool_turn(2), cache=outer_count)
    _send_messages_through_the_chat_bridge(_tool_turn(2), cache=outer_count)

    assert gemini_route.call_count == 2


_SETTLE_TIMEOUT_SECONDS: Final = 10
_COMPLETION_ONLY_CALL_TYPES: Final[tuple[CachingSupportedCallTypes, ...]] = ("completion", "acompletion")


async def _await_cache_writes() -> None:
    await asyncio.wait_for(asyncio.gather(*tuple(_PENDING_CACHE_WRITES)), timeout=_SETTLE_TIMEOUT_SECONDS)


async def _settle_post_response_work() -> None:
    await _await_cache_writes()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=_SETTLE_TIMEOUT_SECONDS)
    await _await_cache_writes()


@pytest.mark.asyncio
async def test_async_messages_chat_bridge_counts_the_caller_messages_when_only_completions_are_cached(
    monkeypatch: pytest.MonkeyPatch, gemini_route: respx.Route
) -> None:
    monkeypatch.setattr(
        litellm,
        "cache",
        Cache(type=LiteLLMCacheType.LOCAL, supported_call_types=list(_COMPLETION_ONLY_CALL_TYPES)),
    )

    try:
        for _ in range(2):
            await litellm.anthropic.messages.acreate(
                model="gemini/gemini-3.8-flash", messages=_tool_turn(3), max_tokens=16
            )
            await _settle_post_response_work()

        assert gemini_route.call_count == 1, "three caller messages were counted as their five chat messages"
    finally:
        await _settle_post_response_work()


@respx.mock
def test_enable_cache_serves_a_repeated_bedrock_titan_embedding_from_the_local_cache(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_provider_credentials: None
) -> None:
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "set_verbose", True)
    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    route: Final = respx.post(
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v2:0/invoke"
    ).mock(return_value=httpx.Response(200, json={"embedding": [0.25, -0.5, 0.75], "inputTextTokenCount": 6}))
    request: Final = {
        "model": "bedrock/amazon.titan-embed-text-v2:0",
        "input": "good morning from litellm, attempting to embed data",
        "aws_region_name": "us-west-2",
    }

    litellm.enable_cache()
    enabled_cache: Final = litellm.cache
    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    litellm.enable_cache()

    first: Final = embedding(**request)
    second: Final = embedding(**request)

    assert isinstance(enabled_cache, Cache)
    assert enabled_cache.type == LiteLLMCacheType.LOCAL
    assert litellm.cache is enabled_cache
    assert litellm.input_callback == ["cache"]
    assert litellm.success_callback == ["cache"]
    assert litellm._async_success_callback == ["cache"]
    assert "LiteLLM: Enabling Cache" in capsys.readouterr().out
    assert route.call_count == 1
    assert first.data[0]["embedding"] == [0.25, -0.5, 0.75]
    assert second.data[0]["embedding"] == [0.25, -0.5, 0.75]
    assert second._hidden_params["cache_hit"] is True


def test_basic_caching_import() -> None:
    from litellm.caching import Cache as PublicCache

    assert PublicCache is Cache
    cache: Final = PublicCache(type=LiteLLMCacheType.LOCAL)
    assert cache.type == LiteLLMCacheType.LOCAL


def test_cache_context_managers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    monkeypatch.setattr(litellm, "input_callback", list(litellm.input_callback))
    monkeypatch.setattr(litellm, "success_callback", list(litellm.success_callback))
    monkeypatch.setattr(litellm, "_async_success_callback", list(litellm._async_success_callback))

    litellm.disable_cache()
    assert litellm.cache is None
    assert "cache" not in litellm.success_callback
    assert "cache" not in litellm._async_success_callback

    litellm.enable_cache(type=LiteLLMCacheType.LOCAL)
    assert litellm.cache is not None
    assert litellm.cache.type == LiteLLMCacheType.LOCAL


@pytest.mark.asyncio
async def test_cache_default_off_acompletion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL, mode=CacheMode.default_off))
    request: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"cache mode {uuid.uuid4().hex}"}],
    }

    first: Final = await acompletion(**request, mock_response="first")
    second: Final = await acompletion(**request, mock_response="second")
    opted_in_first: Final = await acompletion(**request, cache={"use-cache": True}, mock_response="third")
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    opted_in_second: Final = await acompletion(**request, cache={"use-cache": True}, mock_response="fourth")

    assert first.id != second.id
    assert opted_in_first.id == opted_in_second.id


def test_caching_kwargs_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    request: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"kwargs input {uuid.uuid4().hex}"}],
        "caching": True,
    }

    first: Final = completion(**request, mock_response="stored")
    second: Final = completion(**request, mock_response="not stored")

    assert first.id == second.id


def test_caching_reasoning_args_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    request: Final = {
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [{"role": "user", "content": f"reasoning hit {uuid.uuid4().hex}"}],
        "reasoning_effort": "low",
        "caching": True,
    }

    first: Final = completion(**request, mock_response="cached")
    second: Final = completion(**request, mock_response="uncached")

    assert first.id == second.id


def test_caching_reasoning_args_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    messages: Final = [{"role": "user", "content": f"reasoning miss {uuid.uuid4().hex}"}]

    first: Final = completion(
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        reasoning_effort="low",
        caching=True,
        mock_response="low",
    )
    second: Final = completion(
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        caching=True,
        mock_response="default",
    )

    assert first.id != second.id


def test_caching_thinking_args_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    request: Final = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [{"role": "user", "content": f"thinking hit {uuid.uuid4().hex}"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "caching": True,
    }

    first: Final = completion(**request, mock_response="cached")
    second: Final = completion(**request, mock_response="uncached")

    assert first.id == second.id


def test_caching_thinking_args_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    messages: Final = [{"role": "user", "content": f"thinking miss {uuid.uuid4().hex}"}]

    first: Final = completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=messages,
        thinking={"type": "enabled", "budget_tokens": 1024},
        caching=True,
        mock_response="thinking",
    )
    second: Final = completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=messages,
        caching=True,
        mock_response="not thinking",
    )

    assert first.id != second.id


def test_caching_with_reasoning_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    request: Final = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [{"role": "user", "content": f"reasoning content {uuid.uuid4().hex}"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }

    completion(**request, mock_response="reasoned")
    cached: Final = completion(**request, mock_response="different")

    assert cached._hidden_params["cache_hit"] is True


def test_custom_redis_cache_params(redis_client_without_network: None) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="6379", db=4)

    assert cache.cache.redis_client.connection_pool.connection_kwargs["db"] == 4


def test_redis_cache_ignores_a_leftover_disk_cache_dir(
    redis_client_without_network: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="6379")
    assert "disk_cache_dir" not in caplog.text

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        cache: Final = Cache(
            type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="6379", disk_cache_dir="/tmp/litellm-cache"
        )

    assert isinstance(cache.cache, RedisCache)
    assert "disk_cache_dir" in caplog.text


def test_custom_redis_cache_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", cache)
    request: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"custom key {uuid.uuid4().hex}"}],
        "caching": True,
        "cache_key": f"custom-{uuid.uuid4().hex}",
    }

    first: Final = completion(**request, mock_response="stored")
    second: Final = completion(
        **{**request, "messages": [{"role": "user", "content": "different prompt, same key"}]},
        mock_response="not stored",
    )
    uncached: Final = completion(
        **{**request, "cache_key": f"other-{uuid.uuid4().hex}"},
        mock_response="fresh",
    )

    assert second.id == first.id
    assert second.choices[0].message.content == "stored"
    assert uncached.id != first.id
    assert uncached.choices[0].message.content == "fresh"


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache_returns_memory_values() -> None:
    cache: Final = InMemoryCache()
    cache.set_cache("test-value", "cached")
    dual_cache: Final = DualCache(in_memory_cache=cache)

    result: Final = await dual_cache.async_batch_get_cache(keys=["test-value"])

    assert result == ["cached"]


def test_embedding_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    inputs: Final = [f"embedding {uuid.uuid4().hex}"]

    first: Final = embedding(model="text-embedding-3-small", input=inputs, caching=True, mock_response="0.1,0.2")
    second: Final = embedding(model="text-embedding-3-small", input=inputs, caching=True, mock_response="0.3,0.4")

    assert second.data[0]["embedding"] == first.data[0]["embedding"]


def test_embedding_caching_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    request: Final = {
        "model": "azure/text-embedding-3-small",
        "input": [f"azure embedding {uuid.uuid4().hex}"],
        "api_key": "test-key",
        "api_base": "https://example.test",
        "api_version": "2024-01-01",
        "caching": True,
    }

    first: Final = embedding(**request, mock_response="0.1,0.2")
    second: Final = embedding(**request, mock_response="0.3,0.4")

    assert second.data[0]["embedding"] == first.data[0]["embedding"]


@pytest.mark.asyncio
async def test_embedding_caching_individual_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    item: Final = f"individual embedding {uuid.uuid4().hex}"

    first: Final = await aembedding(model="text-embedding-3-small", input=item, caching=True, mock_response="0.1,0.2")
    second: Final = await aembedding(model="text-embedding-3-small", input=item, caching=True, mock_response="0.3,0.4")

    assert second.data[0].embedding == first.data[0].embedding
    assert second._hidden_params["cache_hit"] is True


@pytest.mark.asyncio
async def test_embedding_caching_azure_individual_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    shared: Final = f"shared azure embedding {uuid.uuid4().hex}"

    await aembedding(
        model="azure/text-embedding-3-small",
        input=[shared],
        api_key="test-key",
        api_base="https://example.test",
        api_version="2024-01-01",
        caching=True,
        mock_response="0.1,0.2",
    )
    result: Final = await aembedding(
        model="azure/text-embedding-3-small",
        input=[shared, f"new {uuid.uuid4().hex}"],
        api_key="test-key",
        api_base="https://example.test",
        api_version="2024-01-01",
        caching=True,
        mock_response="0.3,0.4",
    )

    assert result._hidden_params["cache_hit"] is True


@pytest.mark.asyncio
async def test_embedding_caching_azure_individual_items_reordered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    first_item: Final = f"first reordered embedding {uuid.uuid4().hex}"
    second_item: Final = f"second reordered embedding {uuid.uuid4().hex}"
    first: Final = await aembedding(
        model="azure/text-embedding-3-small",
        input=first_item,
        api_key="test-key",
        api_base="https://example.test",
        api_version="2024-01-01",
        caching=True,
        mock_response="0.1,0.2",
    )
    second: Final = await aembedding(
        model="azure/text-embedding-3-small",
        input=second_item,
        api_key="test-key",
        api_base="https://example.test",
        api_version="2024-01-01",
        caching=True,
        mock_response="0.3,0.4",
    )
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    reordered: Final = await aembedding(
        model="azure/text-embedding-3-small",
        input=[second_item, first_item],
        api_key="test-key",
        api_base="https://example.test",
        api_version="2024-01-01",
        caching=True,
        mock_response="0.5,0.6",
    )

    assert reordered._hidden_params["cache_hit"] is True
    assert [item.index for item in reordered.data] == [0, 1]
    assert reordered.data[0].embedding == second.data[0].embedding
    assert reordered.data[1].embedding == first.data[0].embedding


@pytest.mark.asyncio
async def test_embedding_caching_individual_items_and_then_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    first_item: Final = f"first embedding {uuid.uuid4().hex}"
    second_item: Final = f"second embedding {uuid.uuid4().hex}"

    first: Final = await aembedding(
        model="text-embedding-3-small", input=first_item, caching=True, mock_response="0.1,0.2"
    )
    second: Final = await aembedding(
        model="text-embedding-3-small", input=second_item, caching=True, mock_response="0.3,0.4"
    )
    combined: Final = await aembedding(
        model="text-embedding-3-small",
        input=[first_item, second_item],
        caching=True,
        mock_response="0.5,0.6",
    )

    assert combined.data[0].embedding == first.data[0].embedding
    assert combined.data[1].embedding == second.data[0].embedding
    assert combined._hidden_params["cache_hit"] is True


@pytest.mark.asyncio
async def test_caching_with_cache_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    sync_no_ttl_messages: Final = [{"role": "user", "content": f"sync controls {uuid.uuid4().hex}"}]
    sync_no_ttl_first: Final = completion(
        model="gpt-4o-mini", messages=sync_no_ttl_messages, cache={"ttl": 0}, mock_response="first"
    )
    sync_no_ttl_second: Final = completion(
        model="gpt-4o-mini", messages=sync_no_ttl_messages, cache={"s-maxage": 10}, mock_response="second"
    )
    sync_ttl_messages: Final = [{"role": "user", "content": f"sync ttl {uuid.uuid4().hex}"}]
    sync_ttl_first: Final = completion(
        model="gpt-4o-mini", messages=sync_ttl_messages, cache={"ttl": 25}, mock_response="third"
    )
    sync_ttl_second: Final = completion(
        model="gpt-4o-mini", messages=sync_ttl_messages, cache={"s-maxage": 25}, mock_response="fourth"
    )
    async_no_ttl_messages: Final = [{"role": "user", "content": f"async controls {uuid.uuid4().hex}"}]
    async_no_ttl_first: Final = await acompletion(
        model="gpt-4o-mini", messages=async_no_ttl_messages, cache={"ttl": 0}, mock_response="first"
    )
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    async_no_ttl_second: Final = await acompletion(
        model="gpt-4o-mini", messages=async_no_ttl_messages, cache={"s-maxage": 10}, mock_response="second"
    )
    async_ttl_messages: Final = [{"role": "user", "content": f"async ttl {uuid.uuid4().hex}"}]
    async_ttl_first: Final = await acompletion(
        model="gpt-4o-mini", messages=async_ttl_messages, cache={"ttl": 25}, mock_response="third"
    )
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    async_ttl_second: Final = await acompletion(
        model="gpt-4o-mini", messages=async_ttl_messages, cache={"s-maxage": 25}, mock_response="fourth"
    )

    assert sync_no_ttl_first.id != sync_no_ttl_second.id
    assert sync_ttl_first.id == sync_ttl_second.id
    assert async_no_ttl_first.id != async_no_ttl_second.id
    assert async_ttl_first.id == async_ttl_second.id


@pytest.mark.asyncio
async def test_scoped_ttl_rejects_old_exact_cache_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 110.0)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    result: Final = {"id": "stored-response"}
    cache.cache.set_cache("existing-entry", {"timestamp": 100.0, "response": result})
    scoped: Final = {SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=5))}

    assert cache.get_cache(cache_key="existing-entry", cache=scoped) is None
    assert await cache.async_get_cache(cache_key="existing-entry", cache=scoped) is None
    assert cache.get_cache(cache_key="existing-entry") == result
    assert cache.get_cache(cache_key="existing-entry", cache={"s-maxage": 0}) == result
    large_age: Final = {"s-maxage": 10**400}
    assert cache.get_cache(cache_key="existing-entry", cache=large_age) == result
    assert await cache.async_get_cache(cache_key="existing-entry", cache=large_age) == result
    assert cache.get_cache(cache_key="existing-entry", cache={**scoped, **large_age}) is None
    relaxed: Final = {SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=20))}
    assert cache.get_cache(cache_key="existing-entry", cache=relaxed) == result
    assert cache.get_cache(cache_key="existing-entry", cache={**relaxed, "s-maxage": 0}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_age", "accept_fresh"), [("nan", True), ("inf", True), ("-inf", False)])
async def test_nonfinite_request_max_age_cannot_extend_scoped_ttl(
    monkeypatch: pytest.MonkeyPatch, max_age: str, accept_fresh: bool
) -> None:
    monkeypatch.setattr(time, "time", lambda: 110.0)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    fresh: Final = {"id": "fresh-response"}
    cache.cache.set_cache("fresh", {"timestamp": 109.0, "response": fresh})
    cache.cache.set_cache("stale", {"timestamp": 100.0, "response": {"id": "stale-response"}})
    scoped: Final = {
        SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=5)),
        "s-max-age": max_age,
    }

    expected: Final = fresh if accept_fresh else None
    assert cache.get_cache(cache_key="fresh", cache=scoped) == expected
    assert await cache.async_get_cache(cache_key="fresh", cache=scoped) == expected
    assert cache.get_cache(cache_key="stale", cache=scoped) is None
    assert await cache.async_get_cache(cache_key="stale", cache=scoped) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("async_write", [False, True])
@pytest.mark.parametrize("original_ttl", [2, 120])
async def test_scoped_ttl_expires_replaced_local_entries_for_unscoped_readers(
    monkeypatch: pytest.MonkeyPatch, async_write: bool, original_ttl: int
) -> None:
    monkeypatch.setattr(time, "time", lambda: 100.0)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, ttl=120)
    cache.add_cache({"id": "old-response"}, cache_key="existing-entry", cache={"ttl": original_ttl})
    scoped: Final = {
        "ttl": 120,
        SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=5)),
    }
    replacement: Final = {"id": "new-response"}
    if async_write:
        await cache.async_add_cache(replacement, cache_key="existing-entry", cache=scoped)
    else:
        cache.add_cache(replacement, cache_key="existing-entry", cache=scoped)

    monkeypatch.setattr(time, "time", lambda: 104.0)
    assert cache.get_cache(cache_key="existing-entry") == replacement
    monkeypatch.setattr(time, "time", lambda: 105.0)
    assert cache.get_cache(cache_key="existing-entry") is None
    assert await cache.async_get_cache(cache_key="existing-entry") is None


@pytest.mark.asyncio
async def test_disabled_scope_does_not_overwrite_existing_exact_cache_entry() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    result: Final = {"id": "stored-response"}
    cache.add_cache(result, cache_key="existing-entry")
    scoped: Final = {
        SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(enabled=False))
    }

    cache.add_cache({"id": "sync-write"}, cache_key="existing-entry", cache=scoped)
    await cache.async_add_cache({"id": "async-write"}, cache_key="existing-entry", cache=scoped)

    assert cache.get_cache(cache_key="existing-entry") == result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "ttl", "expected_id"),
    [(None, None, "stored-response"), (False, None, None), (True, 5, "fresh-response"), (True, 15, "stored-response")],
)
async def test_batched_redis_reader_applies_scoped_policy(
    monkeypatch: pytest.MonkeyPatch, enabled: bool | None, ttl: int | None, expected_id: str | None
) -> None:
    from litellm.proxy.hooks.batch_redis_get import PROXY_BatchRedisRequests

    monkeypatch.setattr(time, "time", lambda: 110.0)
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, mode=CacheMode.default_off)
    monkeypatch.setattr(litellm, "cache", cache)
    reader: Final = PROXY_BatchRedisRequests()
    worker_store: Final = InMemoryCache()
    reader.in_memory_cache = worker_store
    worker_store.set_cache("existing-entry", {"timestamp": 100.0, "response": {"id": "stored-response"}})
    cache.cache.set_cache("existing-entry", {"timestamp": 109.0, "response": {"id": "fresh-response"}})
    controls: Final = (
        {}
        if enabled is None
        else {
            SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(
                scope="team", params=SemanticCacheParams(enabled=enabled, ttl=ttl)
            )
        }
    )

    assert await cache.async_get_cache(cache_key="existing-entry", cache=controls) == (
        {"id": expected_id} if expected_id is not None else None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("stored_ttl", [None, 5])
async def test_redis_response_cache_reads_refreshed_entry_after_worker_copy_expires(
    monkeypatch: pytest.MonkeyPatch,
    redis_response_cache_clients: tuple[MagicMock, MagicMock],
    sync: bool,
    stored_ttl: int | None,
) -> None:
    monkeypatch.setattr(time, "time", lambda: 110.0)
    cache: Final = Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="6379")
    assert isinstance(cache.cache, RedisCache)
    worker: Final = InMemoryCache()
    dual: Final = DualCache(redis_cache=cache.cache, in_memory_cache=worker)
    stale: Final = {"timestamp": 100.0, "response": {"id": "worker-response"}}
    worker.set_cache("shared-entry", {**stale, "response_cache_ttl": stored_ttl})
    fresh: Final = {"id": "redis-response"}
    stored: Final = json.dumps({"timestamp": 109.0, "response": fresh})
    redis_get: Final = MagicMock(return_value=stored) if sync else AsyncMock(return_value=stored)
    sync_client, async_client = redis_response_cache_clients
    client: Final = sync_client if sync else async_client
    client.get = redis_get
    controls: Final = (
        {SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=5))}
        if stored_ttl is None
        else {}
    )

    if sync:
        assert cache.get_cache(dynamic_cache_object=dual, cache_key="shared-entry", cache=controls) == fresh
        assert cache.get_cache(dynamic_cache_object=dual, cache_key="shared-entry", cache=controls) == fresh
    else:
        assert await cache.async_get_cache(dynamic_cache_object=dual, cache_key="shared-entry", cache=controls) == fresh
        assert await cache.async_get_cache(dynamic_cache_object=dual, cache_key="shared-entry", cache=controls) == fresh
    assert redis_get.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
async def test_redis_response_cache_keeps_worker_copy_when_shared_entry_is_stale(
    monkeypatch: pytest.MonkeyPatch, redis_response_cache_clients: tuple[MagicMock, MagicMock], sync: bool
) -> None:
    monkeypatch.setattr(time, "time", lambda: 110.0)
    cache: Final = Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="6379")
    assert isinstance(cache.cache, RedisCache)
    worker: Final = InMemoryCache()
    dual: Final = DualCache(redis_cache=cache.cache, in_memory_cache=worker)
    worker_response: Final = {"id": "worker-response"}
    worker.set_cache("shared-entry", {"timestamp": 104.0, "response": worker_response})
    stored: Final = json.dumps({"timestamp": 100.0, "response": {"id": "redis-response"}})
    redis_get: Final = MagicMock(return_value=stored) if sync else AsyncMock(return_value=stored)
    sync_client, async_client = redis_response_cache_clients
    client: Final = sync_client if sync else async_client
    client.get = redis_get
    controls: Final = {SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=5))}

    if sync:
        assert cache.get_cache(dynamic_cache_object=dual, cache_key="shared-entry", cache=controls) is None
        assert cache.get_cache(dynamic_cache_object=dual, cache_key="shared-entry") == worker_response
    else:
        assert await cache.async_get_cache(dynamic_cache_object=dual, cache_key="shared-entry", cache=controls) is None
        assert await cache.async_get_cache(dynamic_cache_object=dual, cache_key="shared-entry") == worker_response
    assert redis_get.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("read_path", ["sync", "async", "batched"])
async def test_redis_response_cache_keeps_worker_write_during_shared_read(
    monkeypatch: pytest.MonkeyPatch, redis_response_cache_clients: tuple[MagicMock, MagicMock], read_path: str
) -> None:
    from litellm.proxy.hooks.batch_redis_get import PROXY_BatchRedisRequests

    monkeypatch.setattr(time, "time", lambda: 110.0)
    cache: Final = Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="6379")
    assert isinstance(cache.cache, RedisCache)
    worker: Final = InMemoryCache()
    dual: Final = DualCache(redis_cache=cache.cache, in_memory_cache=worker)
    worker.set_cache("shared-entry", {"timestamp": 104.0, "response": {"id": "old-worker-response"}})
    worker_response: Final = {"id": "new-worker-response"}
    redis_response: Final = {"id": "redis-response"}
    controls: Final = {SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=5))}

    def read_redis(_: str) -> str:
        worker.set_cache("shared-entry", {"timestamp": 109.0, "response": worker_response})
        return json.dumps({"timestamp": 107.0, "response": redis_response})

    sync_client, async_client = redis_response_cache_clients
    if read_path == "sync":
        sync_client.get = MagicMock(side_effect=read_redis)
        assert cache.get_cache(dynamic_cache_object=dual, cache_key="shared-entry", cache=controls) == redis_response
    else:
        async_client.get = AsyncMock(side_effect=read_redis)
        if read_path == "batched":
            monkeypatch.setattr(litellm, "cache", cache)
            reader: Final = PROXY_BatchRedisRequests()
            reader.in_memory_cache = worker
            assert await reader.async_get_cache(cache_key="shared-entry", cache=controls) == redis_response
        else:
            assert await cache.async_get_cache(
                dynamic_cache_object=dual, cache_key="shared-entry", cache=controls
            ) == redis_response
    assert cache.get_cache(dynamic_cache_object=dual, cache_key="shared-entry") == worker_response


@pytest.mark.asyncio
async def test_scoped_ttl_reaches_redis_when_batching_is_configured(
    monkeypatch: pytest.MonkeyPatch, redis_response_cache_clients: tuple[MagicMock, MagicMock]
) -> None:
    monkeypatch.setattr(litellm, "default_redis_ttl", 60)
    cache: Final = Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="6379", redis_flush_size=100)
    pipeline: Final = MagicMock()
    pipeline.__aenter__.return_value = pipeline
    pipeline.execute = AsyncMock()
    _, async_client = redis_response_cache_clients
    async_client.pipeline.return_value = pipeline
    scoped: Final = {
        "ttl": 120,
        SEMANTIC_CACHE_POLICY_KEY: SemanticCachePolicy(scope="team", params=SemanticCacheParams(ttl=30)),
    }

    await cache.async_add_cache({"id": "buffered-response"}, cache_key="existing-entry")
    pipeline.set.assert_not_called()
    await cache.async_add_cache({"id": "scoped-response"}, cache_key="existing-entry", cache=scoped)

    pipeline.execute.assert_awaited_once()
    assert [
        (command.kwargs["name"], json.loads(command.kwargs["value"])["response"], command.kwargs["ex"])
        for command in pipeline.set.call_args_list
    ] == [
        ("existing-entry", {"id": "buffered-response"}, timedelta(seconds=60)),
        ("existing-entry", {"id": "scoped-response"}, timedelta(seconds=30)),
    ]


@pytest.mark.asyncio
async def test_embedding_caching_redis_ttl(monkeypatch: pytest.MonkeyPatch, redis_client_without_network: None) -> None:
    mock_pipeline: Final = AsyncMock()
    mock_set: Final = AsyncMock()
    mock_pipeline.__aenter__.return_value.set = mock_set
    monkeypatch.setattr("redis.asyncio.Redis.pipeline", lambda *args, **kwargs: mock_pipeline)
    monkeypatch.setattr(
        litellm,
        "cache",
        Cache(type=LiteLLMCacheType.REDIS, host="unused", port="6379", default_in_redis_ttl=2),
    )

    await aembedding(
        model="text-embedding-3-small",
        input=[f"redis ttl embedding {uuid.uuid4().hex}"],
        encoding_format="base64",
        caching=True,
        mock_response="0.1,0.2",
    )
    await asyncio.gather(*_PENDING_CACHE_WRITES)

    assert [call.kwargs["ex"] for call in mock_set.call_args_list] == [timedelta(seconds=2)]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_logging_turn_off_message_logging_streaming(
    sync_mode: bool, monkeypatch: pytest.MonkeyPatch, drained_logging_worker: None
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", cache)
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    messages: Final = [{"role": "user", "content": f"stream {uuid.uuid4().hex}"}]
    if sync_mode:
        stream: Final = completion(
            model="gpt-4o-mini", messages=messages, caching=True, stream=True, mock_response="cached"
        )
        tuple(stream)
    else:
        stream: Final = await acompletion(
            model="gpt-4o-mini", messages=messages, caching=True, stream=True, mock_response="cached"
        )
        tuple([chunk async for chunk in stream])

    for _ in range(50):
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()
        await asyncio.gather(*_PENDING_CACHE_WRITES)
        if cache.cache.cache_dict:
            break

    assert len(cache.cache.cache_dict) == 1
    cached_entry: Final = next(iter(cache.cache.cache_dict.values()))
    cached_response: Final = json.loads(cached_entry["response"])
    assert cached_response["choices"][0]["message"]["content"] == "cached"


def test_redis_caching_default_ttl(monkeypatch: pytest.MonkeyPatch, redis_client_without_network: None) -> None:
    monkeypatch.setattr(litellm, "default_redis_ttl", 120)
    cache: Final = RedisCache(host="127.0.0.1", port="6379")

    assert cache.default_ttl == 120


def test_redis_caching_llm_caching_ttl(monkeypatch: pytest.MonkeyPatch, redis_client_without_network: None) -> None:
    monkeypatch.setattr(litellm, "default_redis_ttl", 120)
    cache: Final = RedisCache(host="127.0.0.1", port="6379")
    set_cache: Final = MagicMock()
    monkeypatch.setattr(cache.redis_client, "set", set_cache)

    cache.set_cache(key="cache-key", value="cache-value")

    assert set_cache.call_args.kwargs["ex"] == 120


@pytest.mark.asyncio
async def test_redis_caching_ttl_pipeline(monkeypatch: pytest.MonkeyPatch, redis_client_without_network: None) -> None:
    monkeypatch.setattr(litellm, "default_redis_ttl", 120)
    cache: Final = RedisCache(host="127.0.0.1", port="6379")
    pipeline: Final = AsyncMock()
    set_pipeline: Final = MagicMock()
    monkeypatch.setattr(pipeline, "set", set_pipeline)

    await cache._pipeline_helper(
        pipe=pipeline,
        cache_list=[("first", "one"), ("second", "two")],
        ttl=None,
    )

    assert set_pipeline.call_args_list == [
        call(name="first", value='"one"', ex=timedelta(seconds=120)),
        call(name="second", value='"two"', ex=timedelta(seconds=120)),
    ]


@pytest.mark.asyncio
async def test_redis_caching_ttl_sadd(monkeypatch: pytest.MonkeyPatch, redis_client_without_network: None) -> None:
    monkeypatch.setattr(litellm, "default_redis_ttl", 120)
    cache: Final = RedisCache(host="127.0.0.1", port="6379")
    redis_client: Final = AsyncMock()

    await cache._set_cache_sadd_helper(
        redis_client=redis_client,
        key="cache-key",
        value=["cache-value"],
        ttl=None,
    )

    redis_client.expire.assert_awaited_once_with("cache-key", timedelta(seconds=120))


def test_cache_key_separates_model_ids_unless_a_model_group_owns_the_key() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    variants: Final[tuple[dict[str, object], ...]] = (
        {"model_id": "component-a"},
        {"model_id": "component-b"},
        {},
        {"model_id": "component-a"},
        {"model_id": "component-a", "metadata": {"model_group": "embed"}},
        {"model_id": "component-b", "metadata": {"model_group": "embed"}},
    )
    key_a, key_b, bare, key_a_again, grouped_a, grouped_b = (
        cache.get_cache_key(model="sagemaker/openai/embed", input=["hello"], **extra) for extra in variants
    )

    assert key_a != key_b
    assert key_a != bare
    assert key_a == key_a_again
    assert grouped_a == grouped_b


class _LoudDict(dict[str, str]):
    def __repr__(self) -> str:
        return "<loud dict>"


class _Role(str, Enum):
    USER = "user"


class _ContentPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    text: str


_LARGE_DATA_URL: Final = "data:image/png;base64," + "QUJD" * 524_288


def _self_referencing_list() -> list[object]:  # mutable-ok: a cycle can only be closed by appending to the list
    looped: Final[list[object]] = [{"role": "user", "content": _LARGE_DATA_URL}]  # mutable-ok: closed below
    looped.append({"role": "user", "content": looped})
    return looped


def _self_referencing_dict() -> dict[str, object]:  # mutable-ok: a cycle can only be closed by assigning a key
    looped: Final[dict[str, object]] = {"role": "user", "content": _LARGE_DATA_URL}  # mutable-ok: closed below
    looped["self"] = (looped, [looped])
    return looped


_KEY_PREIMAGE_CASES: Final = MappingProxyType(
    {
        "one_tuple": ({"role": "user", "content": "hi"},),
        "empty_containers": [{"role": "user", "content": [], "a": {}, "b": (), "c": set(), "d": "", "e": frozenset()}],
        "mixed_quotes": [{"role": "user", "content": "it's \"quoted\" and 'single'"}],
        "emoji_newlines_backslashes": [{"role": "user", "content": "\U0001f600 line\nnext\ttab \\ back\r\x00"}],
        "dict_subclass_custom_repr": [_LoudDict(role="user", content="hi")],
        "ordered_dict": [OrderedDict([("content", "hi"), ("role", "user")])],
        "str_enum_top_level": _Role.USER,
        "str_enum_nested": [{"role": _Role.USER, "content": "hi"}],
        "special_floats": [{"role": "user", "content": [float("nan"), -0.0, 1e300, float("inf")]}],
        "bytes": [{"role": "user", "content": b"\x00\xffraw'\"bytes"}],
        "nested_lone_surrogate": [{"role": "user", "content": "a\ud800b"}],
        "deep_nesting": reduce(lambda inner, _: [inner], range(300), {"role": "user", "content": "leaf"}),
        "pydantic_model": [{"role": "user", "content": [_ContentPart(type="text", text="hi")]}],
        "large_data_url_in_one_tuple": [({"content": _LARGE_DATA_URL},)],
        "large_data_url": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": _LARGE_DATA_URL}}],
            }
        ],
        "self_referencing_list": _self_referencing_list(),
        "self_referencing_dict": [_self_referencing_dict()],
        "large_data_url_nested_past_any_walk_depth": reduce(
            lambda inner, _: [inner, ("x",)], range(600), {"role": "user", "content": _LARGE_DATA_URL}
        ),
    }
)


@pytest.mark.parametrize("case", _KEY_PREIMAGE_CASES.values(), ids=_KEY_PREIMAGE_CASES.keys())
def test_cache_key_stays_the_sha256_of_the_formatted_request(case: object) -> None:
    expected: Final = hashlib.sha256(f"model: mmessages: {case}stream: True".encode()).hexdigest()

    assert Cache(type=LiteLLMCacheType.LOCAL).get_cache_key(model="m", messages=case, stream=True) == expected


@pytest.mark.parametrize("provider_specific_params_keyed", [False, True])
@pytest.mark.parametrize("case", _KEY_PREIMAGE_CASES.values(), ids=_KEY_PREIMAGE_CASES.keys())
def test_extra_body_cache_key_formats_any_accepted_value_like_the_request(
    monkeypatch: pytest.MonkeyPatch, provider_specific_params_keyed: bool, case: object
) -> None:
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", provider_specific_params_keyed)
    extra_body: Final = {"payload": case}
    expected: Final = hashlib.sha256(f"model: mmessages: hiextra_body: {extra_body}".encode()).hexdigest()

    assert Cache(type=LiteLLMCacheType.LOCAL).get_cache_key(model="m", messages="hi", extra_body=extra_body) == expected


def test_cache_key_for_a_top_level_lone_surrogate_still_raises() -> None:
    with pytest.raises(UnicodeEncodeError):
        Cache(type=LiteLLMCacheType.LOCAL).get_cache_key(model="m", messages="\ud800", stream=True)


def test_cache_key_for_large_image_payloads_does_not_copy_the_whole_request() -> None:
    image_bytes: Final = 1 << 20
    image_messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"screenshot {index}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + chr(65 + index) * image_bytes}},
            ],
        }
        for index in range(16)
    ]
    payload_bytes: Final = 16 * image_bytes
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)

    already_tracing: Final = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        traced_before, _ = tracemalloc.get_traced_memory()
        cache.get_cache_key(model="m", messages=image_messages, stream=True)
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()

    added_peak: Final = traced_peak - traced_before
    assert added_peak < payload_bytes / 4, (
        f"peak {added_peak / 2**20:.1f} MiB for a {payload_bytes / 2**20:.0f} MiB payload"
    )


class _CountingEmbedder(CustomLLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls: int = 0

    def _respond(self, model: str, input: list[str]) -> EmbeddingResponse:
        self.calls += 1
        return EmbeddingResponse(
            model=model,
            data=[
                Embedding(embedding=[float(self.calls)], index=idx, object="embedding") for idx, _ in enumerate(input)
            ],
        )

    def embedding(self, model: str, input: list[str], *args: object, **kwargs: object) -> EmbeddingResponse:
        return self._respond(model, input)

    async def aembedding(self, model: str, input: list[str], *args: object, **kwargs: object) -> EmbeddingResponse:
        return self._respond(model, input)


@pytest.fixture
def counting_embedder(monkeypatch: pytest.MonkeyPatch) -> _CountingEmbedder:
    embedder: Final = _CountingEmbedder()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "count-embed", "custom_handler": embedder}])
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list, "count-embed"])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "count-embed"])
    return embedder


_EMBED_REQUEST: Final = MappingProxyType({"model": "count-embed/m", "input": ["alpha", "beta"]})
_COMPLETION_REQUEST: Final = MappingProxyType(
    {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "excluded call types"}]}
)
_EMBEDDING_CALL_TYPES: Final = ("embedding", "aembedding")
_CALL_TYPE_EXCLUSION_CASES: Final = (
    pytest.param({}, True, True, id="default"),
    pytest.param({"excluded_call_types": []}, True, True, id="empty_exclusion"),
    pytest.param({"excluded_call_types": list(_EMBEDDING_CALL_TYPES)}, False, True, id="embeddings_excluded"),
    pytest.param({"excluded_call_types": ["completion", "acompletion"]}, True, False, id="completions_excluded"),
    pytest.param(
        {
            "supported_call_types": [*_EMBEDDING_CALL_TYPES, "completion", "acompletion"],
            "excluded_call_types": list(_EMBEDDING_CALL_TYPES),
        },
        False,
        True,
        id="custom_supported_list",
    ),
    pytest.param(
        {"supported_call_types": None, "excluded_call_types": list(_EMBEDDING_CALL_TYPES)},
        False,
        False,
        id="none_supported",
    ),
    pytest.param(
        {"supported_call_types": [], "excluded_call_types": list(_EMBEDDING_CALL_TYPES)},
        False,
        False,
        id="empty_supported",
    ),
)


@asynccontextmanager
async def _owned_cache_writes() -> AsyncIterator[Callable[[], Awaitable[None]]]:
    before: Final = frozenset(_PENDING_CACHE_WRITES)

    async def drain() -> None:
        owned: Final = frozenset(_PENDING_CACHE_WRITES) - before
        if not owned:
            return
        _, unfinished = await asyncio.wait(owned, timeout=CACHE_WRITE_SHUTDOWN_FLUSH_TIMEOUT_SECONDS)
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)

    try:
        yield drain
    finally:
        await drain()


def _completion_text(response: object) -> str:
    assert isinstance(response, litellm.ModelResponse), response
    choice: Final = response.choices[0]
    assert isinstance(choice, litellm.Choices), choice
    return str(choice.message.content)


@pytest.mark.parametrize(("cache_kwargs", "embeddings_cached", "completions_cached"), _CALL_TYPE_EXCLUSION_CASES)
def test_excluded_call_types_decide_sync_cache_reads_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    counting_embedder: _CountingEmbedder,
    cache_kwargs: dict[str, object],
    embeddings_cached: bool,
    completions_cached: bool,
) -> None:
    backend: Final = InMemoryCache()
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL, _backend=backend, **cache_kwargs))

    embedding(**_EMBED_REQUEST)
    embedding(**_EMBED_REQUEST)
    embedding_entries: Final = len(backend.cache_dict)
    completion(**_COMPLETION_REQUEST, mock_response="first")
    second: Final = completion(**_COMPLETION_REQUEST, mock_response="second")

    assert counting_embedder.calls == (1 if embeddings_cached else 2)
    assert (embedding_entries > 0) is embeddings_cached
    assert _completion_text(second) == ("first" if completions_cached else "second")


@pytest.mark.asyncio
@pytest.mark.parametrize(("cache_kwargs", "embeddings_cached", "completions_cached"), _CALL_TYPE_EXCLUSION_CASES)
async def test_excluded_call_types_decide_async_cache_reads_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    counting_embedder: _CountingEmbedder,
    cache_kwargs: dict[str, object],
    embeddings_cached: bool,
    completions_cached: bool,
) -> None:
    backend: Final = InMemoryCache()
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL, _backend=backend, **cache_kwargs))

    async with _owned_cache_writes() as drain_owned_writes:
        await aembedding(**_EMBED_REQUEST)
        await drain_owned_writes()
        await aembedding(**_EMBED_REQUEST)
        await drain_owned_writes()
        embedding_entries: Final = len(backend.cache_dict)
        await acompletion(**_COMPLETION_REQUEST, mock_response="first")
        await drain_owned_writes()
        second: Final = await acompletion(**_COMPLETION_REQUEST, mock_response="second")

    assert counting_embedder.calls == (1 if embeddings_cached else 2)
    assert (embedding_entries > 0) is embeddings_cached
    assert _completion_text(second) == ("first" if completions_cached else "second")


@pytest.mark.asyncio
async def test_excluded_call_types_bypass_embeddings_cached_before_the_exclusion(
    monkeypatch: pytest.MonkeyPatch, counting_embedder: _CountingEmbedder
) -> None:
    backend: Final = InMemoryCache()
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL, _backend=backend))

    async with _owned_cache_writes() as drain_owned_writes:
        await aembedding(**_EMBED_REQUEST)
        await drain_owned_writes()
        primed: Final = dict(backend.cache_dict)
        monkeypatch.setattr(
            litellm,
            "cache",
            Cache(type=LiteLLMCacheType.LOCAL, _backend=backend, excluded_call_types=list(_EMBEDDING_CALL_TYPES)),
        )
        response: Final = await aembedding(**_EMBED_REQUEST)
        await drain_owned_writes()

    assert len(primed) == len(_EMBED_REQUEST["input"])
    assert counting_embedder.calls == 2
    assert [item["embedding"] for item in response.data] == [[2.0], [2.0]]
    assert backend.cache_dict == primed


@pytest.mark.parametrize("configure_cache", [enable_cache, update_cache], ids=["enable_cache", "update_cache"])
def test_cache_setup_helpers_pass_excluded_call_types_to_the_cache(
    monkeypatch: pytest.MonkeyPatch,
    counting_embedder: _CountingEmbedder,
    configure_cache: Callable[..., None],
) -> None:
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "input_callback", list(litellm.input_callback))
    monkeypatch.setattr(litellm, "success_callback", list(litellm.success_callback))
    monkeypatch.setattr(litellm, "_async_success_callback", list(litellm._async_success_callback))
    backend: Final = InMemoryCache()

    configure_cache(type=LiteLLMCacheType.LOCAL, excluded_call_types=list(_EMBEDDING_CALL_TYPES), _backend=backend)
    embedding(**_EMBED_REQUEST)
    embedding(**_EMBED_REQUEST)
    embedding_entries: Final = len(backend.cache_dict)
    completion(**_COMPLETION_REQUEST, mock_response="first")
    second: Final = completion(**_COMPLETION_REQUEST, mock_response="second")

    assert counting_embedder.calls == 2
    assert embedding_entries == 0
    assert _completion_text(second) == "first"


def test_excluded_call_types_leave_the_callers_lists_and_later_defaults_intact() -> None:
    supported: Final = [*_EMBEDDING_CALL_TYPES, "completion"]
    excluded: Final = list(_EMBEDDING_CALL_TYPES)

    Cache(type=LiteLLMCacheType.LOCAL, supported_call_types=supported, excluded_call_types=excluded)
    Cache(type=LiteLLMCacheType.LOCAL, excluded_call_types=excluded)

    assert supported == ["embedding", "aembedding", "completion"]
    assert excluded == ["embedding", "aembedding"]
    assert Cache(type=LiteLLMCacheType.LOCAL).supported_call_types == list(DEFAULT_CACHING_SUPPORTED_CALL_TYPES)


def test_cache_key_tells_apart_requests_that_differ_only_in_previous_interaction_id() -> None:
    cache: Final = Cache()
    request: Final = {"model": "gemini/gemini-4-neon", "messages": [{"role": "user", "content": "and then?"}]}

    fresh_key: Final = cache.get_cache_key(**request)
    first_chat_key: Final = cache.get_cache_key(**request, previous_interaction_id="v1_alice")

    assert first_chat_key != fresh_key
    assert cache.get_cache_key(**request, previous_interaction_id="v1_bob") not in (fresh_key, first_chat_key)
    assert cache.get_cache_key(**request, extra_body={"previous_interaction_id": "v1_bob"}) != fresh_key


class _FloatConversionFails:
    def __init__(self, error: type[Exception]) -> None:
        self.error = error

    def __float__(self) -> float:
        raise self.error("ttl conversion failed")


class _ManualClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def _refresh_local_entry(
    async_write: bool, default_ttl: int | None, refresh_controls: dict[str, object], **write_kwargs: object
) -> tuple[Cache, _ManualClock, float]:
    clock: Final = _ManualClock(1_000_000.0)
    cache: Final = Cache(type="local", ttl=default_ttl)
    cache.cache = InMemoryCache(clock=clock)
    first_written_at: Final = clock.now
    cache.add_cache({"id": "first"}, cache_key="entry", cache={"ttl": 60})
    clock.now += 5
    refresh_kwargs: Final = {
        "cache_key": "entry",
        "cache": {**refresh_controls, "no-cache": True},
        **write_kwargs,
    }
    if async_write:
        await cache.async_add_cache({"id": "refreshed"}, **refresh_kwargs)
    else:
        cache.add_cache({"id": "refreshed"}, **refresh_kwargs)
    return cache, clock, first_written_at


@pytest.mark.asyncio
@pytest.mark.parametrize("async_write", [False, True])
@pytest.mark.parametrize(("request_ttl", "lifetime"), [(10, 10.0), (600, 600.0), ("5", 5.0), (2.5, 2.5)])
async def test_local_response_cache_refresh_restarts_entry_lifetime_from_request_ttl(
    async_write: bool, request_ttl: object, lifetime: float
) -> None:
    cache, clock, first_written_at = await _refresh_local_entry(async_write, None, {"ttl": request_ttl})
    refreshed_at: Final = first_written_at + 5

    clock.now = refreshed_at + lifetime - 1
    assert cache.get_cache(cache_key="entry") == {"id": "refreshed"}
    clock.now = refreshed_at + lifetime + 1
    assert cache.get_cache(cache_key="entry") is None
    assert await cache.async_get_cache(cache_key="entry") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("async_write", [False, True])
@pytest.mark.parametrize(
    ("default_ttl", "refresh_controls", "write_kwargs"),
    [
        *(
            (None, {"ttl": ttl}, {})
            for ttl in (
                "abc",
                [1],
                None,
                0,
                -1,
                "nan",
                "inf",
                float("inf"),
                10**400,
                True,
                False,
                _FloatConversionFails(TypeError),
                _FloatConversionFails(RuntimeError),
            )
        ),
        (None, {}, {}),
        (10, {}, {}),
        (None, {"ttl": 10}, {"replace_ttl": False}),
    ],
)
async def test_local_response_cache_refresh_without_usable_request_ttl_keeps_live_entry_expiry(
    caplog: pytest.LogCaptureFixture,
    async_write: bool,
    default_ttl: int | None,
    refresh_controls: dict[str, object],
    write_kwargs: dict[str, object],
) -> None:
    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        cache, clock, first_written_at = await _refresh_local_entry(
            async_write, default_ttl, refresh_controls, **write_kwargs
        )

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
    clock.now = first_written_at + 59
    assert cache.get_cache(cache_key="entry") == {"id": "refreshed"}
    clock.now = first_written_at + 61
    assert cache.get_cache(cache_key="entry") is None


@pytest.mark.asyncio
async def test_local_embedding_cache_refresh_restarts_entry_lifetime_from_request_ttl() -> None:
    clock: Final = _ManualClock(1_000_000.0)
    cache: Final = Cache(type="local")
    cache.cache = InMemoryCache(clock=clock)
    request: Final = {"model": "text-embedding-3-small", "input": ["hello"]}
    await cache.async_add_cache_pipeline(_embedding_response(prompt_tokens=1, num_items=1), **request)
    clock.now += 5
    await cache.async_add_cache_pipeline(
        _embedding_response(prompt_tokens=2, num_items=1), **request, cache={"no-cache": True, "ttl": 10}
    )
    entry_key: Final = cache.get_cache_key(model="text-embedding-3-small", input="hello")

    clock.now += 9
    assert cache.cache.get_cache(entry_key) is not None
    clock.now += 2
    assert cache.cache.get_cache(entry_key) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TypeError, RuntimeError])
async def test_local_embedding_cache_refresh_with_unconvertible_request_ttl_keeps_live_entry_expiry(
    caplog: pytest.LogCaptureFixture, error: type[Exception]
) -> None:
    clock: Final = _ManualClock(1_000_000.0)
    cache: Final = Cache(type="local")
    cache.cache = InMemoryCache(clock=clock)
    request: Final = {"model": "text-embedding-3-small", "input": ["hello"]}
    await cache.async_add_cache_pipeline(
        _embedding_response(prompt_tokens=1, num_items=1), **request, cache={"ttl": 60}
    )
    clock.now += 5
    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        await cache.async_add_cache_pipeline(
            _embedding_response(prompt_tokens=2, num_items=1),
            **request,
            cache={"no-cache": True, "ttl": _FloatConversionFails(error)},
        )
    entry_key: Final = cache.get_cache_key(model="text-embedding-3-small", input="hello")

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
    clock.now += 54
    refreshed_entry: Final = cache.cache.get_cache(entry_key)
    assert isinstance(refreshed_entry, dict)
    assert refreshed_entry["response"]["prompt_tokens"] == 2
    clock.now += 2
    assert cache.cache.get_cache(entry_key) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("refresh_controls", "expires_after_first_write"), [({"ttl": 10}, 15.0), ({}, 600.0)])
async def test_redis_response_cache_refresh_sets_worker_copy_lifetime_only_from_request_ttl(
    redis_response_cache_clients: tuple[MagicMock, MagicMock],
    refresh_controls: dict[str, object],
    expires_after_first_write: float,
) -> None:
    _, async_client = redis_response_cache_clients
    async_client.set = AsyncMock(return_value=True)
    async_client.get = AsyncMock(return_value=None)
    clock: Final = _ManualClock(1_000_000.0)
    first_written_at: Final = clock.now
    cache: Final = Cache(type="redis", host="127.0.0.1", port="6379")
    dual: Final = DualCache(redis_cache=cache.cache, in_memory_cache=InMemoryCache(clock=clock))
    await cache.async_add_cache({"id": "first"}, dynamic_cache_object=dual, cache_key="entry")
    clock.now += 5
    refreshed: Final = {"id": "refreshed"}
    await cache.async_add_cache(
        refreshed, dynamic_cache_object=dual, cache_key="entry", cache={**refresh_controls, "no-cache": True}
    )

    clock.now = first_written_at + expires_after_first_write - 1
    assert await cache.async_get_cache(dynamic_cache_object=dual, cache_key="entry") == refreshed
    clock.now = first_written_at + expires_after_first_write + 1
    assert await cache.async_get_cache(dynamic_cache_object=dual, cache_key="entry") is None
    assert async_client.set.await_count == 2
