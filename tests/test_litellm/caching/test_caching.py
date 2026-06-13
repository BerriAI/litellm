import logging
import re
from unittest.mock import MagicMock, patch

from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType
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


def _make_semantic_cache(cache_type):
    # Semantic backends need a live Redis/Qdrant connection to construct; the
    # scope-key logic under test lives entirely in Cache.get_cache_key, so the
    # backend is stubbed out.
    with patch(
        "litellm.caching.caching.RedisSemanticCache", return_value=MagicMock()
    ), patch("litellm.caching.caching.QdrantSemanticCache", return_value=MagicMock()):
        return Cache(type=cache_type)


def test_semantic_cache_key_excludes_prompt_content():
    """Semantic caches must scope on params, not prompt text, so that
    semantically similar but differently worded prompts share a bucket and can
    match via vector similarity instead of landing in separate exact-prompt
    buckets."""
    for cache_type in (
        LiteLLMCacheType.REDIS_SEMANTIC,
        LiteLLMCacheType.QDRANT_SEMANTIC,
    ):
        cache = _make_semantic_cache(cache_type)

        key_a = cache.get_cache_key(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        key_b = cache.get_cache_key(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Which city is the capital of France?"}
            ],
        )

        assert key_a == key_b, f"{cache_type} scoped distinct prompts apart"

        # Non-prompt params must still scope entries apart.
        key_other_model = cache.get_cache_key(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        assert key_a != key_other_model, f"{cache_type} ignored model in scope key"


def test_non_semantic_cache_key_includes_prompt_content():
    """Exact caches must keep prompt content in the key so distinct prompts
    don't collide."""
    cache = Cache(type=LiteLLMCacheType.LOCAL)

    key_a = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    key_b = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Which city is the capital of France?"}],
    )

    assert key_a != key_b


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
