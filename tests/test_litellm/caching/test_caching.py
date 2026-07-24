import logging
import re

import pytest

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


def _semantic_cache():
    return Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        host="localhost",
        port="6379",
        similarity_threshold=0.8,
    )


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


@pytest.mark.parametrize("max_age_key", ["s-maxage", "s-max-age"])
def test_get_cache_honors_zero_s_maxage(max_age_key):
    import time

    cache = Cache(type=LiteLLMCacheType.LOCAL)
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }
    cache.add_cache({"content": "cached"}, **kwargs)
    time.sleep(0.02)

    assert cache.get_cache(cache={max_age_key: 0}, **kwargs) is None
    assert cache.get_cache(cache={max_age_key: 60}, **kwargs) == {"content": "cached"}


@pytest.mark.asyncio
@pytest.mark.parametrize("max_age_key", ["s-maxage", "s-max-age"])
async def test_async_get_cache_honors_zero_s_maxage(max_age_key):
    import asyncio

    cache = Cache(type=LiteLLMCacheType.LOCAL)
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }
    await cache.async_add_cache({"content": "cached"}, **kwargs)
    await asyncio.sleep(0.02)

    assert await cache.async_get_cache(cache={max_age_key: 0}, **kwargs) is None
    assert (
        await cache.async_get_cache(cache={max_age_key: 60}, **kwargs)
        == {"content": "cached"}
    )
