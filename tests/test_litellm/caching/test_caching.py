import logging
import re

import pytest

from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType, SemanticCacheScope
from litellm.types.utils import Embedding, EmbeddingResponse, Usage


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
        record.getMessage()
        for record in caplog.records
        if "Created cache key:" in record.getMessage()
    ]
    assert created_cache_key_logs
    assert all(prompt_marker not in message for message in created_cache_key_logs)
    assert any(cache_key in message for message in created_cache_key_logs)


def _embedding_response(prompt_tokens, num_items):
    return EmbeddingResponse(
        model="amazon.titan-embed-image-v1",
        data=[
            Embedding(embedding=[0.0], index=i, object="embedding")
            for i in range(num_items)
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens
        ),
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
        messages=[
            {"role": "user", "content": "Tell me the colour of the daytime sky."}
        ],
        metadata=dict(tenant),
    )
    assert key_a == key_b


def test_semantic_cache_key_isolates_tenants():
    messages = [{"role": "user", "content": "What color is the sky?"}]
    cache = _semantic_cache()
    key_a = cache.get_cache_key(
        model="gpt-4o-mini", messages=messages, metadata={"user_api_key": "hash-A"}
    )
    key_b = cache.get_cache_key(
        model="gpt-4o-mini", messages=messages, metadata={"user_api_key": "hash-B"}
    )
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
    assert cache.get_cache_key(
        model="gpt-4o-mini", messages=messages, metadata=dict(tenant)
    ) != cache.get_cache_key(model="gpt-4o", messages=messages, metadata=dict(tenant))
    assert cache.get_cache_key(
        model="gpt-4o-mini", messages=messages, temperature=0, metadata=dict(tenant)
    ) != cache.get_cache_key(
        model="gpt-4o-mini", messages=messages, temperature=1, metadata=dict(tenant)
    )


def test_exact_cache_key_still_includes_prompt():
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    key_a = cache.get_cache_key(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "a"}]
    )
    key_b = cache.get_cache_key(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "b"}]
    )
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
    assert baseline != cache.get_cache_key(
        model="claude-sonnet-4-5", messages=messages, **anthropic_param
    )
