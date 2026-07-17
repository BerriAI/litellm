import os
import sys
from typing import List, cast

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
    _get_min_token_count_for_deployments,
)
from litellm.router_utils.prompt_caching_cache import PromptCachingCache
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import get_prompt_cache_min_tokens, token_counter

MODEL_GROUP_ALIAS = "my-claude-group"
OPUS_4_6_MIN_TOKENS = 4096


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    """
    The remote cost map does not carry `prompt_cache_min_tokens` yet, so a test that reads the
    default map would pass here and flake in CI. Force the in-repo map.

    `get_model_info` is lru_cached, so swapping `model_cost` is not enough on its own: an earlier
    test that resolved these models against the remote map leaves entries with no
    `prompt_cache_min_tokens`, and the stale hit resolves to the default. Clear on the way out too,
    so the entries these tests warm against the local map do not leak into later tests.
    """
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def _deployments(*models: str) -> List[dict]:
    return [
        {
            "model_name": MODEL_GROUP_ALIAS,
            "litellm_params": {"model": model},
            "model_info": {"id": f"dep-{index}"},
        }
        for index, model in enumerate(models, start=1)
    ]


def _messages(word_count: int) -> List[AllMessageValues]:
    return cast(
        List[AllMessageValues],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "word " * word_count,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )


def test_get_min_token_count_for_deployments_takes_max_across_mixed_group():
    """
    A group may legally mix models whose real minimums differ, and one boolean gate decides for
    every member. The threshold must be the highest minimum in the group: taking the lowest would
    let a 1024-token prompt pin the Opus 4.5 deployment for a prefix Anthropic will never cache.
    """
    assert get_prompt_cache_min_tokens(model="anthropic/claude-opus-4-5") == 4096
    assert get_prompt_cache_min_tokens(model="anthropic/claude-sonnet-4-5") == 1024

    deployments = _deployments("anthropic/claude-opus-4-5", "anthropic/claude-sonnet-4-5")

    assert _get_min_token_count_for_deployments(deployments) == 4096


def test_get_min_token_count_for_deployments_falls_back_to_default_for_empty_group():
    """An empty group has no member minimum to read, so it must fall back rather than crash."""
    assert _get_min_token_count_for_deployments([]) == DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT


@pytest.mark.asyncio
async def test_async_filter_deployments_does_not_narrow_prompt_below_model_minimum():
    """
    The regression. Opus 4.6 will not cache a prefix under 4096 tokens, so a ~1400-token prompt is
    not cacheable and routing must stay free across the whole group. Previously the check resolved
    its threshold from `model`, which is the operator's group alias and matches nothing in the cost
    map, silently fell back to 1024, judged this prompt cacheable, and pinned every request to one
    deployment for a cache hit the provider was never going to serve.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6")
    messages = _messages(word_count=1400)

    token_count = token_counter(messages=messages, model="anthropic/claude-opus-4-6", use_default_image_token_count=True)
    assert DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT < token_count < OPUS_4_6_MIN_TOKENS

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == deployments


@pytest.mark.asyncio
async def test_async_filter_deployments_narrows_prompt_above_model_minimum():
    """
    The positive control for the regression above: once the same group's prompt clears Opus 4.6's
    real 4096-token minimum the prefix is genuinely cacheable, so the check must still pin the
    deployment that served it. Proves the fix tightened the gate rather than disabling the feature.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6")
    messages = _messages(word_count=5000)

    token_count = token_counter(messages=messages, model="anthropic/claude-opus-4-6", use_default_image_token_count=True)
    assert token_count > OPUS_4_6_MIN_TOKENS

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == [deployments[1]]


@pytest.mark.asyncio
async def test_async_filter_deployments_narrows_for_group_whose_model_minimum_is_lower():
    """
    Same ~1400-token prompt that must not pin an Opus 4.6 group, on an Opus 4.8 group whose real
    minimum is 1024. Here the prefix is cacheable and the check must pin. Proves the threshold is
    resolved per-model from the deployments rather than tightened for everyone.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments("anthropic/claude-opus-4-8", "anthropic/claude-opus-4-8")
    messages = _messages(word_count=1400)

    assert get_prompt_cache_min_tokens(model="anthropic/claude-opus-4-8") == DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == [deployments[1]]


@pytest.mark.asyncio
async def test_wildcard_route_resolves_underlying_model_minimum(local_model_cost_map):
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*", "api_key": "sk-fake"},
                "model_info": {"id": "wild-1"},
            }
        ]
    )

    deployments = await router.async_get_healthy_deployments(model="anthropic/claude-opus-4-6", request_kwargs={})

    assert deployments[0]["litellm_params"]["model"] == "anthropic/claude-opus-4-6"
    assert _get_min_token_count_for_deployments(deployments) == 4096
