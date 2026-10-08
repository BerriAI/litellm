import asyncio
import hashlib
import logging
import re
import traceback
import uuid
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import completion, embedding
import litellm.caching.redis_cache as redis_cache_module
from litellm._internal_context import current_service_target
from litellm.caching.caching import Cache, CacheMode, response_cache_phase
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, _RedisTimeoutLogThrottle
from litellm.types.caching import EMBEDDING_CACHE_FORMAT_VERSION, LiteLLMCacheType, SemanticCacheScope
from litellm.types.utils import Embedding, EmbeddingResponse, Usage

_CACHING_TEST_MESSAGES: Final = [{"role": "user", "content": "who is ishaan 5222"}]

messages = [{"role": "user", "content": "who is ishaan 5222"}]


@pytest.fixture
def preserve_litellm_set_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "set_verbose", litellm.set_verbose)


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
    throttle = _RedisTimeoutLogThrottle(interval=5.0, clock=MagicMock(return_value=1_000.0))
    assert throttle.admit() == 0
    monkeypatch.setattr(redis_cache_module, "_redis_timeout_log_throttle", throttle)

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
    assert cache.get_cache_key(**{**request, "top_k": 6}) != base_key


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
        if (
            response2["choices"][0]["message"]["content"]
            != response1["choices"][0]["message"]["content"]
        ):
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
        assert (
            response2["choices"][0]["message"]["content"]
            != response1["choices"][0]["message"]["content"]
        )
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
    messages = [
        {"role": "user", "content": "who is ishaan CTO of litellm from litellm 2023"}
    ]
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
    if (
        response3["choices"][0]["message"]["content"]
        == response2["choices"][0]["message"]["content"]
    ):
        # if models are different, it should not return cached response
        print(f"response2: {response2}")
        print(f"response3: {response3}")
        pytest.fail(f"Error occurred:")
    if (
        response1["choices"][0]["message"]["content"]
        != response2["choices"][0]["message"]["content"]
    ):
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
                "messages": [
                    {"role": "user", "content": "write a one sentence poem about: 7510"}
                ],
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
                "messages": [
                    {"role": "user", "content": "write a one sentence poem about: 7510"}
                ],
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
        assert (
            cache_key_2 == hash_hex
        ), f"{cache_key} != {cache_key_2}. The same kwargs should have the same cache key across runs"

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

        embedding_cache_key_str = (
            "model: azure/text-embedding-ada-002input: ['hi who is ishaan']"
        )
        hash_object = hashlib.sha256(embedding_cache_key_str.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        assert (
            embedding_cache_key == hash_hex
        ), f"{embedding_cache_key} != 'model: azure/text-embedding-ada-002input: ['hi who is ishaan']'. The same kwargs should have the same cache key across runs"

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
        embedding_cache_key_str_2 = (
            "model: EMBEDDING_MODEL_GROUPinput: ['hi who is ishaan']"
        )
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
        patch(
            "litellm._redis.get_redis_connection_pool"
        ) as mock_get_redis_connection_pool,
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

        print(
            f"Response 1 type: {type(response_1)} - ID: {getattr(response_1, 'id', 'N/A')}"
        )
        print(
            f"Response 2 type: {type(response_2)} - ID: {getattr(response_2, 'id', 'N/A')}"
        )
        print(
            f"Response 3 type: {type(response_3)} - Cache hit: {isinstance(response_3, str)}"
        )
        print(
            f"Response 4 type: {type(response_4)} - ID: {getattr(response_4, 'id', 'N/A')}"
        )

        print(f"Redis storage keys: {list(redis_storage.keys())}")

        # Verify that different namespaces created different cache keys
        cache_keys = list(redis_storage.keys())
        namespace_1_keys = [k for k in cache_keys if k.startswith(f"{namespace_1}:")]
        namespace_2_keys = [k for k in cache_keys if k.startswith(f"{namespace_2}:")]
        no_namespace_keys = [
            k
            for k in cache_keys
            if not k.startswith(f"{namespace_1}:")
            and not k.startswith(f"{namespace_2}:")
        ]

        print(f"Namespace 1 keys: {namespace_1_keys}")
        print(f"Namespace 2 keys: {namespace_2_keys}")
        print(f"No namespace keys: {no_namespace_keys}")

        # Should have at least one key for each namespace
        assert len(namespace_1_keys) > 0, "Should have cache keys for namespace 1"
        assert len(namespace_2_keys) > 0, "Should have cache keys for namespace 2"
        assert len(no_namespace_keys) > 0, "Should have cache keys for no namespace"

        # The main test: response 3 should be a cache hit (string) because it uses same namespace as response 1
        assert isinstance(
            response_3, str
        ), "Response 3 should be a cache hit (string) for same namespace"

        # response 1 & 2 should be ModelResponse objects (cache misses)
        assert hasattr(response_1, "id"), "Response 1 should be a ModelResponse object"
        assert hasattr(response_2, "id"), "Response 2 should be a ModelResponse object"
        assert hasattr(response_4, "id"), "Response 4 should be a ModelResponse object"

        # response 1 & 2 should have different IDs (different namespaces)
        assert (
            response_1.id != response_2.id
        ), f"Expected different response ID for different namespace. Got {response_1.id} and {response_2.id}"

        # response 1 & 4 should have different IDs (different namespaces)
        assert (
            response_1.id != response_4.id
        ), f"Expected different response ID for no namespace vs namespaced. Got {response_1.id} and {response_4.id}"


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
