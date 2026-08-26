"""Test Bedrock cross-region inference profile model mapping"""

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import pytest


import litellm
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.common_utils import BedrockModelInfo
from litellm.utils import _get_model_info_helper
from litellm.cost_calculator import completion_cost
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Resolve models against this checkout's cost map instead of the network-fetched
    ``main`` copy, which lags this branch until merge."""
    original_converse_models = set(litellm.bedrock_converse_models)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    try:
        litellm.bedrock_converse_models.update(
            key
            for key, value in litellm.model_cost.items()
            if isinstance(value, dict)
            and value.get("litellm_provider") == "bedrock_converse"
        )
        yield
    finally:
        litellm.bedrock_converse_models.clear()
        litellm.bedrock_converse_models.update(original_converse_models)
        litellm.get_model_info.cache_clear()


class GptProfile(NamedTuple):
    model_id: str
    input_cost: float
    input_cost_above_272k: float
    cache_write: float
    cache_write_above_272k: float
    cache_read: float
    cache_read_above_272k: float
    output_cost: float
    output_cost_above_272k: float


GPT_5_6_PROFILES = [
    GptProfile(
        model_id="us.openai.gpt-5.6-sol",
        input_cost=5.5e-06, input_cost_above_272k=1.1e-05,
        cache_write=6.875e-06, cache_write_above_272k=1.375e-05,
        cache_read=5.5e-07, cache_read_above_272k=1.1e-06,
        output_cost=3.3e-05, output_cost_above_272k=4.95e-05,
    ),
    GptProfile(
        model_id="global.openai.gpt-5.6-sol",
        input_cost=5e-06, input_cost_above_272k=1e-05,
        cache_write=6.25e-06, cache_write_above_272k=1.25e-05,
        cache_read=5e-07, cache_read_above_272k=1e-06,
        output_cost=3e-05, output_cost_above_272k=4.5e-05,
    ),
    GptProfile(
        model_id="us.openai.gpt-5.6-terra",
        input_cost=2.2e-06, input_cost_above_272k=4.4e-06,
        cache_write=2.75e-06, cache_write_above_272k=5.5e-06,
        cache_read=2.2e-07, cache_read_above_272k=4.4e-07,
        output_cost=1.32e-05, output_cost_above_272k=1.98e-05,
    ),
    GptProfile(
        model_id="global.openai.gpt-5.6-terra",
        input_cost=2e-06, input_cost_above_272k=4e-06,
        cache_write=2.5e-06, cache_write_above_272k=5e-06,
        cache_read=2e-07, cache_read_above_272k=4e-07,
        output_cost=1.2e-05, output_cost_above_272k=1.8e-05,
    ),
    GptProfile(
        model_id="us.openai.gpt-5.6-luna",
        input_cost=2.2e-07, input_cost_above_272k=4.4e-07,
        cache_write=2.75e-07, cache_write_above_272k=5.5e-07,
        cache_read=2.2e-08, cache_read_above_272k=4.4e-08,
        output_cost=1.32e-06, output_cost_above_272k=1.98e-06,
    ),
    GptProfile(
        model_id="global.openai.gpt-5.6-luna",
        input_cost=2e-07, input_cost_above_272k=4e-07,
        cache_write=2.5e-07, cache_write_above_272k=5e-07,
        cache_read=2e-08, cache_read_above_272k=4e-08,
        output_cost=1.2e-06, output_cost_above_272k=1.8e-06,
    ),
]


@lru_cache(maxsize=1)
def _packaged_cost_map():
    """The map litellm actually resolves against, for fields ModelInfoBase drops."""
    path = Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
    return json.loads(path.read_text())


def _bedrock_response(model, usage):
    return ModelResponse(
        id="test",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="OK", role="assistant"),
            )
        ],
        usage=usage,
    )


def test_bedrock_cross_region_inference_profile_mapping():
    """Test that bedrock cross-region inference profile model is mapped"""
    model = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"

    model_info = _get_model_info_helper(model=model, custom_llm_provider="bedrock")

    assert model_info is not None
    assert model_info["litellm_provider"] == "bedrock"
    assert model_info["input_cost_per_token"] == 8e-07


def test_proxy_cost_calculation_scenario():
    """Test exact GitHub issue scenario: proxy cost calculation"""
    model = "litellm_proxy/bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"

    # Test model info lookup works
    model_info = _get_model_info_helper(
        model=model, custom_llm_provider="litellm_proxy"
    )
    assert model_info is not None

    # Test cost calculation works
    response = ModelResponse(
        id="test",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Test", role="assistant"),
            )
        ],
        usage=Usage(total_tokens=150, prompt_tokens=100, completion_tokens=50),
    )

    cost = completion_cost(
        completion_response=response, model=model, custom_llm_provider="litellm_proxy"
    )
    expected_cost = (100 * 8e-07) + (50 * 4e-06)
    assert cost == expected_cost


@pytest.mark.parametrize("profile", GPT_5_6_PROFILES, ids=lambda p: p.model_id)
def test_bedrock_gpt_5_6_profiles_route_to_converse(profile, local_model_cost_map):
    """GPT-5.6 is served by Converse on bedrock-runtime, never by Invoke."""
    assert BedrockModelInfo.get_bedrock_route(f"bedrock/{profile.model_id}") == "converse"


@pytest.mark.parametrize("profile", GPT_5_6_PROFILES, ids=lambda p: p.model_id)
def test_bedrock_gpt_5_6_published_rates(profile, local_model_cost_map):
    """Geo and Global profiles carry their own published rates, per context tier."""
    model_info = _get_model_info_helper(
        model=f"bedrock/{profile.model_id}", custom_llm_provider="bedrock"
    )

    assert model_info["litellm_provider"] == "bedrock_converse"
    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 1000000
    assert model_info["input_cost_per_token"] == profile.input_cost
    assert (
        model_info["input_cost_per_token_above_272k_tokens"]
        == profile.input_cost_above_272k
    )
    assert model_info["output_cost_per_token"] == profile.output_cost
    assert (
        model_info["output_cost_per_token_above_272k_tokens"]
        == profile.output_cost_above_272k
    )
    assert model_info["cache_creation_input_token_cost"] == profile.cache_write
    assert (
        model_info["cache_creation_input_token_cost_above_272k_tokens"]
        == profile.cache_write_above_272k
    )
    assert model_info["cache_read_input_token_cost"] == profile.cache_read
    assert (
        model_info["cache_read_input_token_cost_above_272k_tokens"]
        == profile.cache_read_above_272k
    )


def test_bedrock_gpt_5_6_above_272k_tier_applies_to_cost(local_model_cost_map):
    """A prompt over 272K tokens is billed at the long-context rate, not the base rate."""
    response = _bedrock_response(
        "bedrock/us.openai.gpt-5.6-sol",
        Usage(prompt_tokens=300000, completion_tokens=1000, total_tokens=301000),
    )

    cost = completion_cost(
        completion_response=response,
        model="bedrock/us.openai.gpt-5.6-sol",
        custom_llm_provider="bedrock",
    )

    assert cost == pytest.approx((300000 * 1.1e-05) + (1000 * 4.95e-05), rel=1e-9)


def test_bedrock_gpt_5_6_bills_cache_read_tokens(local_model_cost_map):
    """Bedrock caches long prefixes implicitly and reports them, so a cache-read turn
    must be billed at the cache rate rather than dropped to zero."""
    usage = Usage(
        prompt_tokens=15611,
        completion_tokens=5,
        total_tokens=15616,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=15609),
    )
    response = _bedrock_response("bedrock/us.openai.gpt-5.6-sol", usage)

    cost = completion_cost(
        completion_response=response,
        model="bedrock/us.openai.gpt-5.6-sol",
        custom_llm_provider="bedrock",
    )

    expected = (2 * 5.5e-06) + (15609 * 5.5e-07) + (5 * 3.3e-05)
    assert cost == pytest.approx(expected, rel=1e-9)
    # Without cache_read_input_token_cost the cached prefix bills at zero.
    assert cost > (15611 * 5.5e-06) * 0.1


def test_bedrock_gpt_5_6_bills_cache_write_tokens(local_model_cost_map):
    """The write side of the same cache cycle is billed at the 30m cache-write rate."""
    usage = Usage(
        prompt_tokens=15611,
        completion_tokens=5,
        total_tokens=15616,
        cache_creation_input_tokens=15609,
    )
    response = _bedrock_response("bedrock/us.openai.gpt-5.6-sol", usage)

    cost = completion_cost(
        completion_response=response,
        model="bedrock/us.openai.gpt-5.6-sol",
        custom_llm_provider="bedrock",
    )

    expected = (2 * 5.5e-06) + (15609 * 6.875e-06) + (5 * 3.3e-05)
    assert cost == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("profile", GPT_5_6_PROFILES, ids=lambda p: p.model_id)
def test_bedrock_gpt_5_6_advertises_only_converse_supported_features(
    profile, local_model_cost_map
):
    model_info = _get_model_info_helper(
        model=f"bedrock/{profile.model_id}", custom_llm_provider="bedrock"
    )

    assert model_info["supports_function_calling"] is True
    assert model_info["supports_tool_choice"] is True
    assert model_info["supports_vision"] is True

    # Bedrock rejects an explicit cachePoint block for these models, so the flag that
    # offers caller-driven caching stays off even though the cache rates are declared.
    assert not model_info.get("supports_prompt_caching")

    # ModelInfoBase drops these two, so they are read from the map litellm resolves.
    raw = _packaged_cost_map()[profile.model_id]
    assert raw["supported_modalities"] == ["text", "image"]
    assert raw["supported_output_modalities"] == ["text"]
    # No bedrock_converse entry declares supported_endpoints; these models are reachable
    # on chat completions and on the Responses API without it.
    assert "supported_endpoints" not in raw


@pytest.mark.parametrize("profile", GPT_5_6_PROFILES, ids=lambda p: p.model_id)
def test_bedrock_gpt_5_6_offers_tools_and_reasoning_effort_but_not_thinking(profile, local_model_cost_map):
    """GPT-5.x on Converse maps reasoning_effort to reasoning.effort, so reasoning_effort
    is offered while the Anthropic-only thinking/output_config are not, alongside the tool
    params these models accept."""
    supported = AmazonConverseConfig().get_supported_openai_params(
        model=f"bedrock/{profile.model_id}"
    )

    assert "tools" in supported
    assert "tool_choice" in supported
    assert "reasoning_effort" in supported
    assert "thinking" not in supported
    assert "output_config" not in supported
