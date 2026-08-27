import json

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.llms.gemini.image_generation.cost_calculator import (
    cost_calculator as gemini_image_generation_cost_calculator,
)
from litellm.llms.vertex_ai.image_generation.cost_calculator import (
    cost_calculator as vertex_image_generation_cost_calculator,
)
from litellm.types.llms.openai import FileSearchTool, WebSearchOptions
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    ModelInfo,
    ModelResponse,
    PromptTokensDetailsWrapper,
    StandardBuiltInToolsParams,
)

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    PromptTokensDetailsResult,
    TokenTypeCostBreakdown,
    _calculate_input_cost,
    _get_token_base_cost,
    calculate_cache_writing_cost,
    generic_cost_per_token,
    get_token_type_cost_breakdown,
)
from litellm.types.utils import CacheCreationTokenDetails, Usage


@pytest.fixture
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


def test_reasoning_tokens_no_price_set(_local_model_cost_map):
    # Use o1 - o1-mini was deprecated/renamed; o1 has same reasoning-token semantics
    # (no separate output_cost_per_reasoning_token, so all completion tokens use output_cost_per_token)
    model = "o1"
    custom_llm_provider = "openai"
    model_cost_map = litellm.model_cost[model]
    usage = Usage(
        completion_tokens=1578,
        prompt_tokens=17,
        total_tokens=1595,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=952,
            rejected_prediction_tokens=None,
            text_tokens=626,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=17, image_tokens=None
        ),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    print(f"completion_cost: {completion_cost}")
    expected_completion_cost = (
        model_cost_map["output_cost_per_token"] * usage.completion_tokens
    )
    print(f"expected_completion_cost: {expected_completion_cost}")
    assert round(completion_cost, 10) == round(
        expected_completion_cost,
        10,
    )


def test_reasoning_tokens_gemini(_local_model_cost_map):
    model = "gemini-2.5-flash"
    custom_llm_provider = "gemini"

    usage = Usage(
        completion_tokens=1578,
        prompt_tokens=17,
        total_tokens=1595,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=952,
            rejected_prediction_tokens=None,
            text_tokens=626,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=17, image_tokens=None
        ),
    )
    model_cost_map = litellm.model_cost[model]
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        (
            model_cost_map["output_cost_per_token"]
            * usage.completion_tokens_details.text_tokens
        )
        + (
            model_cost_map["output_cost_per_reasoning_token"]
            * usage.completion_tokens_details.reasoning_tokens
        ),
        10,
    )


def test_reasoning_tokens_gemini_3_1_flash_lite(_local_model_cost_map):
    """Test cost calculation for gemini-3.1-flash-lite-preview with reasoning tokens"""
    model = "gemini-3.1-flash-lite-preview"
    custom_llm_provider = "gemini"

    usage = Usage(
        completion_tokens=1000,
        prompt_tokens=500,
        total_tokens=1500,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=400,
            rejected_prediction_tokens=None,
            text_tokens=600,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=500, image_tokens=None
        ),
    )
    model_cost_map = litellm.model_cost[model]
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        (
            model_cost_map["output_cost_per_token"]
            * usage.completion_tokens_details.text_tokens
        )
        + (
            model_cost_map["output_cost_per_reasoning_token"]
            * usage.completion_tokens_details.reasoning_tokens
        ),
        10,
    )


def test_image_tokens_with_custom_pricing():
    """Test that image_tokens in completion are properly costed with output_cost_per_image_token."""
    from unittest.mock import patch

    # Mock model info with image token pricing
    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_image_token": 5e-6,  # Custom pricing for image tokens in output
    }

    usage = Usage(
        completion_tokens=1720,  # text_tokens (600) + image_tokens (1120)
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=600,
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=14, image_tokens=None
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    # Expected costs:
    # Prompt: 14 * 1e-6
    # Completion: (600 * 2e-6) + (1120 * 5e-6)
    expected_prompt_cost = 14 * 1e-6
    expected_completion_cost = (600 * 2e-6) + (1120 * 5e-6)

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_image_tokens_fallback_to_base_cost():
    """Test that image_tokens fall back to base cost when output_cost_per_image_token is not set."""
    from unittest.mock import patch

    # Mock model info without image token pricing
    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        # No output_cost_per_image_token defined
    }

    usage = Usage(
        completion_tokens=1720,
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=600,
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=14, image_tokens=None
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    # Expected costs:
    # Prompt: 14 * 1e-6
    # Completion: (600 * 2e-6) + (1120 * 2e-6)  # image_tokens use base cost
    expected_prompt_cost = 14 * 1e-6
    expected_completion_cost = (600 * 2e-6) + (1120 * 2e-6)

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_video_output_tokens_gemini_omni_flash_preview(_local_model_cost_map):
    """Video output tokens are billed at output_cost_per_video_token, not the text rate and not zero."""
    model = "gemini-omni-flash-preview"

    text_tokens = 100
    video_tokens = 46336
    usage = Usage(
        completion_tokens=text_tokens + video_tokens,
        prompt_tokens=20,
        total_tokens=20 + text_tokens + video_tokens,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            text_tokens=text_tokens,
            video_tokens=video_tokens,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=20),
    )
    model_cost_map = litellm.model_cost[f"gemini/{model}"]
    assert model_cost_map["input_cost_per_token"] == 1.5e-06
    assert model_cost_map["output_cost_per_token"] == 9e-06
    assert model_cost_map["output_cost_per_video_token"] == 1.75e-05

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="gemini",
    )

    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        (model_cost_map["output_cost_per_token"] * text_tokens)
        + (model_cost_map["output_cost_per_video_token"] * video_tokens),
        10,
    )


def test_video_input_tokens_gemini_omni_flash_preview(_local_model_cost_map):
    """Video input tokens are billed at the standard input rate instead of being dropped."""
    model = "gemini-omni-flash-preview"

    usage = Usage(
        completion_tokens=10,
        prompt_tokens=10050,
        total_tokens=10060,
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=10),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=50, video_tokens=10000),
    )
    model_cost_map = litellm.model_cost[f"gemini/{model}"]

    prompt_cost, _ = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="gemini",
    )

    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )


def test_video_tokens_fallback_to_base_cost():
    """Video output tokens fall back to the base output rate when output_cost_per_video_token is not set."""
    from unittest.mock import patch

    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
    }

    usage = Usage(
        completion_tokens=1720,
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            text_tokens=600,
            video_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=14),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    assert round(prompt_cost, 12) == round(14 * 1e-6, 12)
    assert round(completion_cost, 12) == round((600 + 1120) * 2e-6, 12)


def test_generic_cost_per_token_above_200k_tokens(_local_model_cost_map):
    # gemini-2.5-pro-exp-03-25 was removed; gemini-2.5-pro has same above-200k pricing
    model = "gemini-2.5-pro"
    custom_llm_provider = "vertex_ai"

    model_cost_map = litellm.model_cost[model]
    prompt_tokens = 220 * 1e6
    completion_tokens = 150
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token_above_200k_tokens"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        model_cost_map["output_cost_per_token_above_200k_tokens"]
        * usage.completion_tokens,
        10,
    )


def test_get_token_base_cost_picks_highest_crossed_tier():
    """Regression test for #30345.

    With graduated tiers at 90k and 128k whose keys have different digit lengths, a request
    crossing both must be billed at the highest tier it crosses (128k), not the lower one that
    happens to sort first lexicographically.
    """
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_above_90k_tokens": 5e-6,
        "input_cost_per_token_above_128k_tokens": 9e-6,
    }
    usage = Usage(prompt_tokens=150_000, completion_tokens=10, total_tokens=150_010)

    prompt_base_cost = _get_token_base_cost(model_info, usage)[0]

    assert prompt_base_cost == 9e-6


def test_generic_cost_per_token_gpt54_above_272k_tokens(_local_model_cost_map):
    """GPT-5.4/5.4-pro: prompts >272K input tokens priced at 2x input, 1.5x output."""
    model = "gpt-5.4"
    custom_llm_provider = "openai"

    model_cost_map = litellm.model_cost[model]
    prompt_tokens = 273000  # Above 272K threshold
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    expected_prompt = (
        model_cost_map["input_cost_per_token_above_272k_tokens"] * prompt_tokens
    )
    expected_completion = (
        model_cost_map["output_cost_per_token_above_272k_tokens"] * completion_tokens
    )
    assert round(prompt_cost, 10) == round(expected_prompt, 10)
    assert round(completion_cost, 10) == round(expected_completion, 10)


def test_generic_cost_per_token_minimax_m3_above_512k_tokens(_local_model_cost_map):
    """MiniMax-M3: prompts >512K input tokens priced at 2x input, output, and cache read."""
    model = "minimax/MiniMax-M3"
    custom_llm_provider = "minimax"

    model_cost_map = litellm.model_cost[model]
    prompt_tokens = 600000
    cached_tokens = 100000
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    expected_prompt = (
        model_cost_map["input_cost_per_token_above_512k_tokens"]
        * (prompt_tokens - cached_tokens)
        + model_cost_map["cache_read_input_token_cost_above_512k_tokens"]
        * cached_tokens
    )
    expected_completion = (
        model_cost_map["output_cost_per_token_above_512k_tokens"] * completion_tokens
    )
    assert round(prompt_cost, 10) == round(expected_prompt, 10)
    assert round(completion_cost, 10) == round(expected_completion, 10)


@pytest.mark.parametrize(
    "model",
    [
        "bedrock_mantle/openai.gpt-5.6-sol",
        "bedrock_mantle/openai.gpt-5.6-terra",
        "bedrock_mantle/openai.gpt-5.6-luna",
    ],
)
def test_generic_cost_per_token_bedrock_mantle_gpt56_long_context(_local_model_cost_map, model):
    """Bedrock GPT-5.6 enforces a 1,050,000-token context window, billed at the long-context rates above 272K."""

    model_cost_map = litellm.model_cost[model]
    assert model_cost_map["max_input_tokens"] == 1050000

    cached_tokens = 100000
    completion_tokens = 1000

    short_prompt_tokens = 272000
    short_usage = Usage(
        prompt_tokens=short_prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=short_prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )
    short_prompt_cost, short_completion_cost = generic_cost_per_token(
        model=model,
        usage=short_usage,
        custom_llm_provider="bedrock_mantle",
    )
    assert round(short_prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * (short_prompt_tokens - cached_tokens)
        + model_cost_map["cache_read_input_token_cost"] * cached_tokens,
        10,
    )
    assert round(short_completion_cost, 10) == round(
        model_cost_map["output_cost_per_token"] * completion_tokens, 10
    )

    long_prompt_tokens = 900000
    long_usage = Usage(
        prompt_tokens=long_prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=long_prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )
    long_prompt_cost, long_completion_cost = generic_cost_per_token(
        model=model,
        usage=long_usage,
        custom_llm_provider="bedrock_mantle",
    )
    assert round(long_prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token_above_272k_tokens"]
        * (long_prompt_tokens - cached_tokens)
        + model_cost_map["cache_read_input_token_cost_above_272k_tokens"]
        * cached_tokens,
        10,
    )
    assert round(long_completion_cost, 10) == round(
        model_cost_map["output_cost_per_token_above_272k_tokens"] * completion_tokens, 10
    )


@pytest.mark.parametrize(
    "model",
    [
        "bedrock_mantle/openai.gpt-5.5",
        "bedrock_mantle/openai.gpt-5.4",
    ],
)
def test_generic_cost_per_token_bedrock_mantle_gpt55_gpt54_long_context_flat_rate(_local_model_cost_map, model):
    """Bedrock serves gpt-5.5 and gpt-5.4 up to its enforced 1,050,000-token prompt maximum and documents
    no long-context tier for them, so a prompt past 272K is billed at the flat per-token rates."""

    model_cost_map = litellm.model_cost[model]
    assert model_cost_map["max_input_tokens"] == 1050000
    assert [key for key in model_cost_map if "above_272k" in key] == []

    served_prompt_tokens = 1030590
    cached_tokens = 100000
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=served_prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=served_prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="bedrock_mantle",
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * (served_prompt_tokens - cached_tokens)
        + model_cost_map["cache_read_input_token_cost"] * cached_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(model_cost_map["output_cost_per_token"] * completion_tokens, 10)


def test_generic_cost_per_token_honors_non_standard_above_threshold():
    """Regression for #30344: get_model_info must keep arbitrary
    input/output_cost_per_token_above_<N>_tokens thresholds, not only the hard-coded
    128k/200k/272k/512k set, so a custom tier boundary is applied past its limit."""
    model = "litellm-test-non-standard-tier"
    custom_llm_provider = "openai"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "input_cost_per_token_above_500k_tokens": 9e-6,
                "output_cost_per_token_above_500k_tokens": 18e-6,
            }
        }
    )

    try:
        prompt_tokens = 600000
        completion_tokens = 1000
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 10) == round(9e-6 * prompt_tokens, 10)
        assert round(completion_cost, 10) == round(18e-6 * completion_tokens, 10)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tiered_pricing_charges_cache_creation_at_tier_rate():
    """Regression for LIT-4375: a tier's cache_creation_input_token_cost must be billed
    on the generic (provider-agnostic) path, not silently dropped."""
    model = "litellm-test-tiered-cache-creation"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "tiered_pricing": [
                    {
                        "range": [0, 256000],
                        "input_cost_per_token": 3.25e-07,
                        "output_cost_per_token": 1.95e-06,
                        "cache_creation_input_token_cost": 4.063e-07,
                        "cache_read_input_token_cost": 3.25e-08,
                    },
                    {
                        "range": [256000, 1000000],
                        "input_cost_per_token": 6.5e-07,
                        "output_cost_per_token": 3.9e-06,
                        "cache_creation_input_token_cost": 8.125e-07,
                        "cache_read_input_token_cost": 6.5e-08,
                    },
                ],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=300000,  # 200k new + 60k cache creation + 40k cache read
            completion_tokens=1000,
            total_tokens=301000,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=40000, cache_creation_tokens=60000
            ),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )

        expected_prompt = (
            (200000 * 6.5e-07) + (60000 * 8.125e-07) + (40000 * 6.5e-08)
        )
        assert round(prompt_cost, 10) == round(expected_prompt, 10)
        assert round(completion_cost, 10) == round(1000 * 3.9e-06, 10)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tiered_pricing_is_all_or_nothing():
    """Tiered pricing bills the whole request at the tier picked from its input tokens,
    for any provider, and falls back to flat pricing when no tier matches."""
    model = "litellm-test-tiered-all-or-nothing"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "tiered_pricing": [
                    {
                        "range": [0, 32000],
                        "input_cost_per_token": 4.6e-07,
                        "output_cost_per_token": 2.3e-06,
                    },
                    {
                        "range": [32000, 128000],
                        "input_cost_per_token": 7e-07,
                        "output_cost_per_token": 3.5e-06,
                    },
                ],
            }
        }
    )

    try:
        usage = Usage(prompt_tokens=40000, completion_tokens=1000, total_tokens=41000)
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 10) == round(40000 * 7e-07, 10)
        assert round(completion_cost, 10) == round(1000 * 3.5e-06, 10)

        boundary_usage = Usage(prompt_tokens=32000, completion_tokens=10, total_tokens=32010)
        boundary_prompt_cost, _ = generic_cost_per_token(
            model=model,
            usage=boundary_usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(boundary_prompt_cost, 10) == round(32000 * 4.6e-07, 10)

        empty_prompt_usage = Usage(prompt_tokens=0, completion_tokens=100, total_tokens=100)
        empty_prompt_cost, empty_completion_cost = generic_cost_per_token(
            model=model,
            usage=empty_prompt_usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert empty_prompt_cost == 0.0
        assert round(empty_completion_cost, 10) == round(100 * 2e-06, 10)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_an_output_rate_bills_the_model_rate():
    """Regression: a tier table that spells out only input rates served every completion for
    free, since a tier's missing output rate has no tier-level fallback to stand in for it."""
    model = "litellm-test-tiered-input-only"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "output_cost_per_token": 2e-06,
                "output_cost_per_reasoning_token": 5e-06,
                "tiered_pricing": [{"range": [0, 128000], "input_cost_per_token": 1e-03}],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=13,
            completion_tokens=182,
            total_tokens=195,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=100),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 12) == round(13 * 1e-03, 12)
        assert round(completion_cost, 12) == round((82 * 2e-06) + (100 * 5e-06), 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_cache_rates_bills_cache_at_the_tier_input_rate():
    model = "litellm-test-tiered-no-cache-rates"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "cache_read_input_token_cost": 9e-09,
                "cache_creation_input_token_cost": 9e-06,
                "tiered_pricing": [
                    {
                        "range": [0, 32000],
                        "input_cost_per_token": 4.6e-07,
                        "output_cost_per_token": 2.3e-06,
                    },
                    {
                        "range": [32000, 128000],
                        "input_cost_per_token": 7e-07,
                        "output_cost_per_token": 3.5e-06,
                    },
                ],
            }
        }
    )

    try:
        uncached = Usage(prompt_tokens=40000, completion_tokens=100, total_tokens=40100)
        cached = Usage(
            prompt_tokens=40000,
            completion_tokens=100,
            total_tokens=40100,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=5000, cache_creation_tokens=15000
            ),
        )
        uncached_prompt_cost, _ = generic_cost_per_token(
            model=model,
            usage=uncached,
            custom_llm_provider=custom_llm_provider,
        )
        cached_prompt_cost, cached_completion_cost = generic_cost_per_token(
            model=model,
            usage=cached,
            custom_llm_provider=custom_llm_provider,
        )

        tier_input_rate = 7e-07
        assert round(cached_prompt_cost, 12) == round(40000 * tier_input_rate, 12)
        assert round(cached_prompt_cost, 12) == round(uncached_prompt_cost, 12)
        assert round(cached_completion_cost, 12) == round(100 * 3.5e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_a_1hr_cache_rate_bills_the_tier_cache_creation_rate():
    model = "litellm-test-tiered-no-1hr-cache-rate"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "cache_creation_input_token_cost_above_1hr": 9e-05,
                "tiered_pricing": [
                    {
                        "range": [0, 128000],
                        "input_cost_per_token": 7e-07,
                        "output_cost_per_token": 3.5e-06,
                        "cache_creation_input_token_cost": 8.75e-07,
                    }
                ],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=10,
            total_tokens=1010,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cache_creation_tokens=800,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=300, ephemeral_1h_input_tokens=500
                ),
            ),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )

        tier_cache_creation_rate = 8.75e-07
        expected_prompt = (200 * 7e-07) + (800 * tier_cache_creation_rate)
        assert round(prompt_cost, 12) == round(expected_prompt, 12)
        assert round(completion_cost, 12) == round(10 * 3.5e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_an_input_rate_is_not_a_priced_tier():
    model = "litellm-test-tiered-no-input-rate"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "tiered_pricing": [{"range": [0, 128000], "output_cost_per_token": 3.5e-06}],
            }
        }
    )

    try:
        usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 12) == round(1000 * 1e-06, 12)
        assert round(completion_cost, 12) == round(100 * 2e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_router_deployment_with_input_only_tiers_bills_completions_at_the_backend_rate():
    """Regression: the router registers a deployment's custom pricing as a standalone
    model_cost entry holding only the supplied fields, so an input-only tier table left
    the output-rate fallback nothing to read and billed every completion at 0."""
    from litellm import Router

    model_id = "litellm-test-router-tiered-input-only"
    backend_model = "anthropic/claude-haiku-4-5"
    backend_output_rate = litellm.get_model_info(backend_model)["output_cost_per_token"]
    Router(
        model_list=[
            {
                "model_name": "tiered-input-only",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "sk-test",
                    "tiered_pricing": [
                        {"range": [0, 3000], "input_cost_per_token": 3.25e-07},
                        {"range": [3000, 128000], "input_cost_per_token": 8.125e-07},
                    ],
                },
                "model_info": {"id": model_id},
            }
        ]
    )

    try:
        usage = Usage(prompt_tokens=21, completion_tokens=4, total_tokens=25)
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model_id,
            usage=usage,
            custom_llm_provider="anthropic",
        )
        assert round(prompt_cost, 12) == round(21 * 3.25e-07, 12)
        assert round(completion_cost, 12) == round(4 * backend_output_rate, 12)
        assert backend_output_rate > 0
    finally:
        litellm.model_cost.pop(model_id, None)


def test_generic_cost_per_token_tiered_pricing_bills_reasoning_at_tier_rate():
    """Regression: a tier's output_cost_per_reasoning_token must price reasoning tokens
    on the generic path and in the logged breakdown, not the tier's plain output rate."""
    model = "litellm-test-tiered-reasoning"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "tiered_pricing": [
                    {
                        "range": [0, 256000],
                        "input_cost_per_token": 4e-07,
                        "output_cost_per_token": 1.2e-06,
                        "output_cost_per_reasoning_token": 4e-06,
                    },
                    {
                        "range": [256000, 1000000],
                        "input_cost_per_token": 1.2e-06,
                        "output_cost_per_token": 3.6e-06,
                        "output_cost_per_reasoning_token": 1.2e-05,
                    },
                ],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=400),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 12) == round(1000 * 4e-07, 12)
        assert round(completion_cost, 12) == round((100 * 1.2e-06) + (400 * 4e-06), 12)

        breakdown = get_token_type_cost_breakdown(
            model=model,
            custom_llm_provider=custom_llm_provider,
            usage=usage,
        )
        assert round(breakdown.reasoning_cost, 12) == round(400 * 4e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_gpt55(_local_model_cost_map):
    """gpt-5.5: base pricing — $5/1M input, $30/1M output, $0.50/1M cached input."""
    model = "gpt-5.5"
    custom_llm_provider = "openai"

    model_cost_map = litellm.model_cost[model]

    # Sanity-check the map values match OpenAI's published pricing.
    assert model_cost_map["input_cost_per_token"] == 5e-6
    assert model_cost_map["output_cost_per_token"] == 3e-5
    assert model_cost_map["cache_read_input_token_cost"] == 5e-7
    assert model_cost_map["litellm_provider"] == "openai"
    assert model_cost_map["mode"] == "chat"
    # gpt-5.5 inherits GPT-5.4's long-context window + tiered pricing.
    assert model_cost_map["max_input_tokens"] == 1050000
    assert model_cost_map["input_cost_per_token_above_272k_tokens"] == 1e-5
    assert model_cost_map["output_cost_per_token_above_272k_tokens"] == 4.5e-5

    prompt_tokens = 1000
    completion_tokens = 500
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * prompt_tokens, 10
    )
    assert round(completion_cost, 10) == round(
        model_cost_map["output_cost_per_token"] * completion_tokens, 10
    )


def test_generic_cost_per_token_gpt55_pro(_local_model_cost_map):
    """gpt-5.5-pro: responses-only model — $30/1M input, $180/1M output, $3/1M cached input."""
    model = "gpt-5.5-pro"
    custom_llm_provider = "openai"

    model_cost_map = litellm.model_cost[model]

    # Sanity-check the map values match OpenAI's published pricing.
    assert model_cost_map["input_cost_per_token"] == 3e-5
    assert model_cost_map["output_cost_per_token"] == 1.8e-4
    assert model_cost_map["cache_read_input_token_cost"] == 3e-6
    assert model_cost_map["litellm_provider"] == "openai"
    # gpt-5.5-pro is a responses-only model (no /v1/chat/completions endpoint).
    assert model_cost_map["mode"] == "responses"
    assert "/v1/chat/completions" not in model_cost_map["supported_endpoints"]
    assert "/v1/responses" in model_cost_map["supported_endpoints"]
    # Inherits GPT-5.4-pro's long-context window + tiered pricing.
    assert model_cost_map["max_input_tokens"] == 1050000
    assert model_cost_map["input_cost_per_token_above_272k_tokens"] == 6e-5
    assert model_cost_map["output_cost_per_token_above_272k_tokens"] == 2.7e-4

    prompt_tokens = 1000
    completion_tokens = 500
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * prompt_tokens, 10
    )
    assert round(completion_cost, 10) == round(
        model_cost_map["output_cost_per_token"] * completion_tokens, 10
    )


@pytest.mark.parametrize(
    "model,input_cost,output_cost,cache_read_cost,cache_write_cost",
    [
        ("gpt-5.6", 4e-6, 2e-5, 4e-7, 5e-6),
        ("gpt-5.6-sol", 4e-6, 2e-5, 4e-7, 5e-6),
        ("gpt-5.6-terra", 2e-6, 1.2e-5, 2e-7, 2.5e-6),
        ("gpt-5.6-luna", 2e-7, 1.2e-6, 2e-8, 2.5e-7),
    ],
)
def test_generic_cost_per_token_gpt56(_local_model_cost_map, 
    model, input_cost, output_cost, cache_read_cost, cache_write_cost
):
    """gpt-5.6 (sol/terra/luna): base pricing + new cache-write cost.

    Cache writes are billed at 1.25x the uncached input rate for this family.
    """
    custom_llm_provider = "openai"

    model_cost_map = litellm.model_cost[model]

    assert model_cost_map["input_cost_per_token"] == input_cost
    assert model_cost_map["output_cost_per_token"] == output_cost
    assert model_cost_map["cache_read_input_token_cost"] == cache_read_cost
    assert model_cost_map["cache_creation_input_token_cost"] == cache_write_cost
    assert model_cost_map["litellm_provider"] == "openai"
    assert model_cost_map["mode"] == "chat"
    assert model_cost_map["cache_creation_input_token_cost"] == pytest.approx(
        input_cost * 1.25
    )
    assert model_cost_map["max_input_tokens"] == 922000
    assert model_cost_map["input_cost_per_token_above_272k_tokens"] == pytest.approx(
        input_cost * 2
    )
    assert model_cost_map["output_cost_per_token_above_272k_tokens"] == pytest.approx(
        output_cost * 1.5
    )

    prompt_tokens = 1000
    completion_tokens = 500
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    assert round(prompt_cost, 10) == round(input_cost * prompt_tokens, 10)
    assert round(completion_cost, 10) == round(output_cost * completion_tokens, 10)


def test_gpt_5_6_alias_prices_match_sol(local_model_cost_map):
    """Regression: the bare gpt-5.6 alias routes to GPT-5.6 Sol, so every cost field on
    the two entries has to hold the same value. They drifted once before, when Sol took
    its promotional cut and gpt-5.6 was left on the pre-cut rates, overbilling callers
    who used the alias."""
    alias = litellm.model_cost["gpt-5.6"]
    sol = litellm.model_cost["gpt-5.6-sol"]

    cost_fields = sorted(field for field in sol if "cost" in field)
    assert len(cost_fields) == 23

    for field in cost_fields:
        assert alias.get(field) == sol.get(field), field


@pytest.mark.parametrize(
    "model,flex_long_input_cost,flex_long_output_cost",
    [
        ("gpt-5.6", 4e-6, 1.5e-5),
        ("gpt-5.6-sol", 4e-6, 1.5e-5),
        ("gpt-5.6-terra", 2e-6, 9e-6),
        ("gpt-5.6-luna", 2e-7, 9e-7),
    ],
)
def test_generic_cost_per_token_gpt56_flex_above_272k(_local_model_cost_map, 
    model, flex_long_input_cost, flex_long_output_cost
):
    """A >272K flex request bills the flex long-context rate, not the standard one.

    Flex long-context is half the standard long-context rate. Without the
    ``*_above_272k_tokens_flex`` keys these requests silently fell back to the
    standard long-context price, billing 2x what OpenAI charges.
    """

    prompt_tokens = 300000
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
        service_tier="flex",
    )

    assert prompt_cost == pytest.approx(flex_long_input_cost * prompt_tokens)
    assert completion_cost == pytest.approx(flex_long_output_cost * completion_tokens)

    standard_long_prompt_cost, standard_long_completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
        service_tier=None,
    )
    assert prompt_cost == pytest.approx(standard_long_prompt_cost / 2)
    assert completion_cost == pytest.approx(standard_long_completion_cost / 2)


@pytest.mark.parametrize(
    "service_tier,prompt_tokens,input_rate,cache_write_rate,cache_read_rate",
    [
        (None, 100000, 2e-6, 2.5e-6, 2e-7),
        ("flex", 100000, 1e-6, 1.25e-6, 1e-7),
        ("priority", 100000, 4e-6, 5e-6, 4e-7),
        (None, 300000, 4e-6, 5e-6, 4e-7),
        ("flex", 300000, 2e-6, 2.5e-6, 2e-7),
    ],
)
def test_generic_cost_per_token_gpt56_terra_cache_costs_by_tier_and_context(_local_model_cost_map, 
    service_tier, prompt_tokens, input_rate, cache_write_rate, cache_read_rate
):

    cached_tokens = 50000
    cache_write_tokens = 40000
    text_tokens = prompt_tokens - cached_tokens - cache_write_tokens
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=100,
        total_tokens=prompt_tokens + 100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens
        ),
    )

    prompt_cost, _ = generic_cost_per_token(
        model="gpt-5.6-terra",
        usage=usage,
        custom_llm_provider="openai",
        service_tier=service_tier,
    )

    expected_prompt_cost = (
        text_tokens * input_rate
        + cached_tokens * cache_read_rate
        + cache_write_tokens * cache_write_rate
    )
    assert prompt_cost == pytest.approx(expected_prompt_cost)


@pytest.mark.parametrize("model", ["gpt-5.6-cyber", "daybreak-red-latest"])
@pytest.mark.parametrize(
    "prompt_tokens,input_rate,cache_write_rate,cache_read_rate,output_rate",
    [
        (100000, 1.25e-5, 1.5625e-5, 1.25e-6, 7.5e-5),
        (300000, 2.5e-5, 3.125e-5, 2.5e-6, 1.125e-4),
    ],
)
def test_generic_cost_per_token_gpt56_cyber(
    model,
    prompt_tokens,
    input_rate,
    cache_write_rate,
    cache_read_rate,
    output_rate,
    monkeypatch,
):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    cached_tokens = 50000
    cache_write_tokens = 40000
    text_tokens = prompt_tokens - cached_tokens - cache_write_tokens
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
    )

    assert prompt_cost == pytest.approx(
        text_tokens * input_rate
        + cached_tokens * cache_read_rate
        + cache_write_tokens * cache_write_rate
    )
    assert completion_cost == pytest.approx(completion_tokens * output_rate)


@pytest.mark.parametrize(
    "model,input_cost,output_cost,cache_read_cost",
    [
        ("azure/gpt-5.6", 5e-6, 3e-5, 5e-7),
        ("azure/gpt-5.6-sol", 5e-6, 3e-5, 5e-7),
        ("azure/gpt-5.6-terra", 2e-6, 1.2e-5, 2e-7),
        ("azure/gpt-5.6-luna", 2e-7, 1.2e-6, 2e-8),
        ("azure/us/gpt-5.6", 5.5e-6, 3.3e-5, 5.5e-7),
        ("azure/eu/gpt-5.6-terra", 2.2e-6, 1.32e-5, 2.2e-7),
        ("azure/eu/gpt-5.6-luna", 2.2e-7, 1.32e-6, 2.2e-8),
    ],
)
def test_generic_cost_per_token_azure_gpt56(_local_model_cost_map, 
    model, input_cost, output_cost, cache_read_cost
):
    """Azure gpt-5.6 (global + us/eu regional): Azure prices this family on its own
    schedule and carries the standard 10% regional uplift on top. It did not take the
    promotional cut OpenAI applied to gpt-5.6-sol, so these rates deliberately sit
    above the openai ones and must not be lowered to match them.
    """

    model_cost_map = litellm.model_cost[model]
    assert model_cost_map["litellm_provider"] == "azure"
    assert model_cost_map["input_cost_per_token"] == input_cost
    assert model_cost_map["output_cost_per_token"] == output_cost
    assert model_cost_map["cache_read_input_token_cost"] == cache_read_cost
    assert model_cost_map["max_input_tokens"] == 922000

    prompt_tokens = 1000
    completion_tokens = 500
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="azure",
    )
    assert round(prompt_cost, 10) == round(input_cost * prompt_tokens, 10)
    assert round(completion_cost, 10) == round(output_cost * completion_tokens, 10)


@pytest.mark.parametrize(
    "model,expected_none,expected_xhigh,expected_minimal",
    [
        # Verified against OpenAI's live API on 2026-04-24:
        #   gpt-5.5   -> supports: none, low, medium, high, xhigh
        #   gpt-5.5-pro -> supports: medium, high, xhigh
        # Neither supports "minimal"; gpt-5.5-pro additionally does not support "none".
        # The JSON must reflect this so LiteLLM rejects unsupported values locally
        # (or drops them with drop_params=True) instead of round-tripping to OpenAI
        # for a 400.
        ("gpt-5.5", True, True, False),
        ("gpt-5.5-2026-04-23", True, True, False),
        ("gpt-5.5-pro", False, True, False),
        ("gpt-5.5-pro-2026-04-23", False, True, False),
    ],
)
def test_gpt55_reasoning_effort_flags_match_live_openai_api(_local_model_cost_map, 
    model, expected_none, expected_xhigh, expected_minimal
):
    """Pin reasoning_effort capability flags to OpenAI's actual API contract.

    Observed via `POST /v1/chat/completions` with reasoning_effort=minimal:
    ``Unsupported value: 'reasoning_effort' does not support 'minimal' with
    this model``. gpt-5.5-pro additionally rejects 'none' and 'low'.
    """

    m = litellm.model_cost[model]
    assert (
        m.get("supports_none_reasoning_effort") is expected_none
    ), f"{model}: supports_none_reasoning_effort expected {expected_none}"
    assert (
        m.get("supports_xhigh_reasoning_effort") is expected_xhigh
    ), f"{model}: supports_xhigh_reasoning_effort expected {expected_xhigh}"
    assert (
        m.get("supports_minimal_reasoning_effort") is expected_minimal
    ), f"{model}: supports_minimal_reasoning_effort expected {expected_minimal}"


@pytest.mark.parametrize(
    "base_model,dated_model",
    [
        ("gpt-5.5", "gpt-5.5-2026-04-23"),
        ("gpt-5.5-pro", "gpt-5.5-pro-2026-04-23"),
    ],
)
def test_gpt55_dated_variants_match_base_reasoning_effort_capabilities(_local_model_cost_map, 
    base_model, dated_model
):
    """Dated snapshots must carry the same reasoning_effort capability flags as
    their non-dated counterparts.

    Regression guard: ``supports_{none,minimal,xhigh}_reasoning_effort`` gate
    downstream routing in ``OpenAIGPT5Config`` — a missing flag is treated as
    ``False`` for opt-in levels (e.g. ``xhigh``), which silently diverges
    behavior between ``gpt-5.5`` and ``gpt-5.5-2026-04-23``. Pinning to a
    dated variant must never lose capabilities relative to the base alias.
    """

    base = litellm.model_cost[base_model]
    dated = litellm.model_cost[dated_model]

    for flag in (
        "supports_none_reasoning_effort",
        "supports_minimal_reasoning_effort",
        "supports_xhigh_reasoning_effort",
    ):
        assert dated.get(flag) == base.get(flag), (
            f"{dated_model} has {flag}={dated.get(flag)!r}, "
            f"but {base_model} has {flag}={base.get(flag)!r}. "
            f"Dated snapshots must inherit the base model's reasoning_effort "
            f"capability profile."
        )


@pytest.mark.parametrize(
    "model,expected_mode,expected_input,expected_output,expected_cache_read",
    [
        ("azure/gpt-5.5", "chat", 5e-6, 3e-5, 5e-7),
        ("azure/gpt-5.5-2026-04-23", "chat", 5e-6, 3e-5, 5e-7),
        ("azure/gpt-5.5-pro", "responses", 3e-5, 1.8e-4, 3e-6),
        ("azure/gpt-5.5-pro-2026-04-23", "responses", 3e-5, 1.8e-4, 3e-6),
    ],
)
def test_azure_gpt55_entries_present_with_correct_pricing(_local_model_cost_map, 
    model, expected_mode, expected_input, expected_output, expected_cache_read
):
    """Day-0 Azure entries for GPT-5.5 mirror the OpenAI pricing structure.

    Pricing parity with openai/gpt-5.5* (verified against OpenAI's pricing page
    on 2026-04-24): $5/$30 input/output per 1M for chat, $30/$180 for pro.
    Cache discount is 10% of input.
    """

    m = litellm.model_cost[model]
    assert m["litellm_provider"] == "azure"
    assert m["mode"] == expected_mode
    assert m["input_cost_per_token"] == expected_input
    assert m["output_cost_per_token"] == expected_output
    assert m["cache_read_input_token_cost"] == expected_cache_read
    # Long-context window inherited from gpt-5.4 / openai gpt-5.5.
    assert m["max_input_tokens"] == 1050000
    assert m["max_output_tokens"] == 128000


@pytest.mark.parametrize(
    "model,expected_none,expected_minimal,expected_xhigh",
    [
        # Mirror live OpenAI API contract (verified via openai/gpt-5.5* on
        # 2026-04-24): chat accepts {none, low, medium, high, xhigh} but NOT
        # minimal; pro accepts {medium, high, xhigh} only.
        # NOTE: openai/gpt-5.5* entries currently set supports_minimal=true on
        # main (pre #26456). Once that PR lands, OpenAI + Azure flags align.
        ("azure/gpt-5.5", True, False, True),
        ("azure/gpt-5.5-pro", False, False, True),
    ],
)
def test_azure_gpt55_reasoning_effort_flags_match_live_openai_api(_local_model_cost_map, 
    model, expected_none, expected_minimal, expected_xhigh
):
    """Azure entries pin reasoning_effort flags to OpenAI's actual API contract."""

    m = litellm.model_cost[model]
    assert m.get("supports_none_reasoning_effort") is expected_none
    assert m.get("supports_minimal_reasoning_effort") is expected_minimal
    assert m.get("supports_xhigh_reasoning_effort") is expected_xhigh


def test_generic_cost_per_token_anthropic_prompt_caching():
    model = "claude-sonnet-4@20250514"
    usage = Usage(
        completion_tokens=90,
        prompt_tokens=28436,
        total_tokens=28526,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=0, text_tokens=None, image_tokens=None
        ),
        cache_creation_input_tokens=118,
        cache_read_input_tokens=28432,
    )

    custom_llm_provider = "vertex_ai"

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    print(f"prompt_cost: {prompt_cost}")
    assert prompt_cost < 0.085


def test_generic_cost_per_token_anthropic_prompt_caching_with_cache_creation():
    model = "claude-haiku-4-5-20251001"
    usage = Usage(
        completion_tokens=90,
        prompt_tokens=28436,
        total_tokens=28526,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
        prompt_tokens_details=None,
        cache_creation_input_tokens=2000,
    )

    custom_llm_provider = "anthropic"

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    print(f"prompt_cost: {prompt_cost}")
    assert round(prompt_cost, 3) == 0.029


def test_string_cost_values():
    """Test that cost values defined as strings are properly converted to floats."""
    from unittest.mock import patch

    # Mock model info with string cost values (as might be read from config.yaml)
    mock_model_info = {
        "input_cost_per_token": "3e-7",  # String representation of scientific notation
        "output_cost_per_token": "6e-7",  # String representation of scientific notation
        "input_cost_per_audio_token": "0.000001",  # String representation of decimal
        "output_cost_per_audio_token": "0.000002",  # String representation of decimal
        "cache_read_input_token_cost": "1.5e-8",  # String representation of scientific notation
        "cache_creation_input_token_cost": "2.5e-8",  # String representation of scientific notation
    }

    # Test usage with various token types
    # Note: prompt_tokens must equal sum of details to avoid double-counting adjustment
    # text_tokens(700) + audio_tokens(100) + cached_tokens(200) + cache_creation_tokens(150) = 1150
    usage = Usage(
        prompt_tokens=1150,
        completion_tokens=500,
        total_tokens=1650,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=100,
            cached_tokens=200,
            text_tokens=700,
            image_tokens=None,
            cache_creation_tokens=150,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=50,
            reasoning_tokens=None,
            text_tokens=450,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
    )

    # Mock get_model_info to return our mock model info
    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Calculate expected costs manually
    # Prompt cost = text_tokens * input_cost + audio_tokens * audio_cost + cached_tokens * cache_read_cost + cache_creation_tokens * cache_creation_cost
    expected_prompt_cost = (
        700 * 3e-7  # text tokens
        + 100 * 1e-6  # audio tokens
        + 200 * 1.5e-8  # cached tokens
        + 150 * 2.5e-8  # cache creation tokens
    )

    # Completion cost = text_tokens * output_cost + audio_tokens * audio_output_cost
    expected_completion_cost = 450 * 6e-7 + 50 * 2e-6  # text tokens  # audio tokens

    # Assert costs are calculated correctly
    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_generic_cost_per_token_overlapping_cached_and_image_tokens():
    """Some providers report cached_tokens and image_tokens as overlapping subsets of
    prompt_tokens. Billing each in full charged the overlap twice, once at the cache rate
    and again at the input rate."""
    model = "litellm-test-overlapping-cached-image"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                "cache_read_input_token_cost": 1e-7,
                "output_cost_per_token": 2e-6,
            }
        }
    )
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=None, cached_tokens=90, image_tokens=80
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="openai"
    )

    # 90 cached at 1e-7, the remaining 10 uncached tokens once at 1e-6
    assert prompt_cost == pytest.approx(90 * 1e-7 + 10 * 1e-6)
    assert completion_cost == pytest.approx(10 * 2e-6)


def test_generic_cost_per_token_warm_prefix_cache_spanning_text_and_image_tokens():
    """xAI reports text_tokens + image_tokens = prompt_tokens with cached_tokens overlapping
    both, so a warm prefix cache covering the whole image exceeds the text-only count.
    Observed live on grok-4.6 (issue #37281): the image tokens were billed a second time at
    the full input rate on top of the cache-read bucket, 0.003500 in vs the provider's own
    0.001274 bill."""
    model = "litellm-test-warm-prefix-cache-overlap"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 2e-6,
                "cache_read_input_token_cost": 5e-7,
                "output_cost_per_token": 6e-6,
            }
        }
    )
    usage = Usage(
        prompt_tokens=2461,
        completion_tokens=440,
        total_tokens=2901,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=1319, cached_tokens=2432, image_tokens=1142
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="openai"
    )

    # 2432 cached at the cache-read rate, the 29 uncached tokens once at the input rate
    assert prompt_cost == pytest.approx(2432 * 5e-7 + 29 * 2e-6)
    assert completion_cost == pytest.approx(440 * 6e-6)


def test_calculate_cost_component_with_string_values():
    """Test the calculate_cost_component function directly with string cost values."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import calculate_cost_component

    # Test with valid string scientific notation
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 1000 * 3e-7

    # Test with valid string decimal notation
    model_info = {"output_cost_per_token": "0.000001"}
    cost = calculate_cost_component(model_info, "output_cost_per_token", 500)
    assert cost == 500 * 0.000001

    # Test with float value (should work as before)
    model_info = {"input_cost_per_token": 3e-7}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 1000 * 3e-7

    # Test with invalid string value (should return 0.0)
    model_info = {"input_cost_per_token": "invalid_number"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with None value (should return 0.0)
    model_info = {"input_cost_per_token": None}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with missing key (should return 0.0)
    model_info = {}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with zero usage (should return 0.0)
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 0)
    assert cost == 0.0

    # Test with None usage (should return 0.0)
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", None)
    assert cost == 0.0


def test_string_cost_values_edge_cases():
    """Test edge cases for string cost value handling."""
    from unittest.mock import patch

    # Test with mixed string and float cost values
    mock_model_info = {
        "input_cost_per_token": "1e-6",  # String
        "output_cost_per_token": 2e-6,  # Float
        "input_cost_per_audio_token": "invalid",  # Invalid string
        "output_cost_per_audio_token": None,  # None value
    }

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=100, cached_tokens=0, text_tokens=1000, image_tokens=None
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=50,
            reasoning_tokens=None,
            text_tokens=500,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Expected costs:
    # Prompt: 1000 * 1e-6 + 100 * 0 (invalid string becomes 0)
    # Completion: 500 * 2e-6 + 50 * 2e-6 (audio tokens fall back to base cost when output_cost_per_audio_token is None)
    expected_prompt_cost = 1000 * 1e-6
    expected_completion_cost = 500 * 2e-6 + 50 * 2e-6

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_string_cost_values_with_threshold():
    """Test that string cost values work correctly with threshold pricing."""
    from unittest.mock import patch

    # Mock model info with string cost values including threshold pricing
    mock_model_info = {
        "input_cost_per_token": "1e-6",  # String base cost
        "output_cost_per_token": "2e-6",  # String base cost
        "input_cost_per_token_above_200k_tokens": "5e-7",  # String threshold cost (lower)
        "output_cost_per_token_above_200k_tokens": "1e-6",  # String threshold cost (lower)
    }

    # Test usage above threshold
    usage = Usage(
        prompt_tokens=250000,  # Above 200k threshold
        completion_tokens=1000,
        total_tokens=251000,
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Expected costs using threshold pricing (string values converted to float)
    expected_prompt_cost = 250000 * 5e-7  # threshold cost
    expected_completion_cost = 1000 * 1e-6  # threshold cost

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_calculate_cache_writing_cost():
    """Test the calculate_cache_writing_cost function with detailed cache creation token breakdown."""

    # Test case 1: With cache creation token details (matching the provided input)
    cache_creation_tokens = 14055
    cache_creation_token_details = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=56, ephemeral_1h_input_tokens=13999
    )
    cache_creation_cost_above_1hr = 6e-06
    cache_creation_cost = 3.75e-06

    result = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_token_details,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        cache_creation_cost=cache_creation_cost,
    )

    # Expected calculation:
    # 5m tokens: 56 * 3.75e-06 = 0.00021
    # 1h tokens: 13999 * 6e-06 = 0.083994
    # Total: 0.00021 + 0.083994 = 0.084204
    expected_cost = (56 * 3.75e-06) + (13999 * 6e-06)

    assert round(result, 6) == round(expected_cost, 6)
    assert round(result, 6) == 0.084204

    # Test case 2: Without cache creation token details (fallback behavior)
    cache_creation_tokens_no_details = 1000
    cache_creation_token_details_none = None
    cache_creation_cost_fallback = 5e-06

    result_no_details = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens_no_details,
        cache_creation_token_details=cache_creation_token_details_none,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        cache_creation_cost=cache_creation_cost_fallback,
    )

    # Expected calculation when no details: 1000 * 5e-06 = 0.005
    expected_cost_no_details = 1000 * 5e-06

    assert round(result_no_details, 6) == round(expected_cost_no_details, 6)
    assert result_no_details == 0.005

    # Test case 3: With cache creation token details but None values
    cache_creation_token_details_partial = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=None, ephemeral_1h_input_tokens=100
    )

    result_partial = calculate_cache_writing_cost(
        cache_creation_tokens=500,
        cache_creation_token_details=cache_creation_token_details_partial,
        cache_creation_cost_above_1hr=6e-06,
        cache_creation_cost=3e-06,
    )

    # Expected calculation: 0 (for None 5m tokens) + (100 * 6e-06) = 0.0006
    expected_cost_partial = (0.0) + (100 * 6e-06)

    assert round(result_partial, 6) == round(expected_cost_partial, 6)
    assert round(result_partial, 6) == 0.0006

    # Test case 4: Zero costs
    result_zero = calculate_cache_writing_cost(
        cache_creation_tokens=1000,
        cache_creation_token_details=CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=50, ephemeral_1h_input_tokens=950
        ),
        cache_creation_cost_above_1hr=0.0,
        cache_creation_cost=0.0,
    )

    assert result_zero == 0.0


def test_cache_writing_cost_with_zero_creation_tokens_and_ephemeral_details():
    """
    Regression test: when cache_creation_tokens is 0 but cache_creation_token_details
    has non-zero ephemeral tokens, the cost must still be calculated.
    This ensures the guard in _calculate_input_cost doesn't skip
    calculate_cache_writing_cost when only ephemeral token details are present.
    """
    cache_creation_cost = 3.75e-06
    cache_creation_cost_above_1hr = 6e-06

    prompt_tokens_details: PromptTokensDetailsResult = {
        "cache_hit_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_creation_token_details": CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=100,
            ephemeral_1h_input_tokens=200,
        ),
        "text_tokens": 0,
        "audio_tokens": 0,
        "image_tokens": 0,
        "video_tokens": 0,
        "character_count": 0,
        "image_count": 0,
        "video_length_seconds": 0.0,
        "audio_length_seconds": 0.0,
    }

    model_info: ModelInfo = {}

    result = _calculate_input_cost(
        prompt_tokens_details=prompt_tokens_details,
        model_info=model_info,
        prompt_base_cost=0.0,
        cache_read_cost=0.0,
        cache_creation_cost=cache_creation_cost,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
    )

    # Expected: (100 * 3.75e-06) + (200 * 6e-06) = 0.000375 + 0.0012 = 0.001575
    expected = (100 * cache_creation_cost) + (200 * cache_creation_cost_above_1hr)
    assert (
        result > 0
    ), "Cost should not be zero when ephemeral token details are present"
    assert round(result, 6) == round(expected, 6)


def test_service_tier_flex_pricing(_local_model_cost_map):
    """Test that flex service tier uses correct pricing (approximately 50% of standard)."""
    # Set up environment for local model cost map

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"
    custom_llm_provider = "openai"

    # Create usage object
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    # Test standard pricing
    std_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=None,
    )
    std_total = std_cost[0] + std_cost[1]

    # Test flex pricing
    flex_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="flex",
    )
    flex_total = flex_cost[0] + flex_cost[1]

    # Verify flex is approximately 50% of standard
    assert std_total > 0, "Standard cost should be greater than 0"
    assert flex_total > 0, "Flex cost should be greater than 0"

    flex_ratio = flex_total / std_total
    assert (
        0.45 <= flex_ratio <= 0.55
    ), f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"

    # Verify specific costs match expected values
    # gpt-5-nano flex: input=2.5e-08, output=2e-07
    expected_flex_prompt = 1000 * 2.5e-08  # 0.000025
    expected_flex_completion = 500 * 2e-07  # 0.0001
    expected_flex_total = expected_flex_prompt + expected_flex_completion

    assert (
        abs(flex_cost[0] - expected_flex_prompt) < 1e-10
    ), f"Flex prompt cost mismatch: {flex_cost[0]} vs {expected_flex_prompt}"
    assert (
        abs(flex_cost[1] - expected_flex_completion) < 1e-10
    ), f"Flex completion cost mismatch: {flex_cost[1]} vs {expected_flex_completion}"
    assert (
        abs(flex_total - expected_flex_total) < 1e-10
    ), f"Flex total cost mismatch: {flex_total} vs {expected_flex_total}"


def test_service_tier_default_pricing(_local_model_cost_map):
    """Test that when no service tier is provided, standard pricing is used."""
    # Set up environment for local model cost map

    # Test with gpt-5-nano
    model = "gpt-5-nano"
    custom_llm_provider = "openai"

    # Create usage object
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    # Test with no service tier (should use standard)
    default_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=None,
    )

    # Test with explicit standard service tier
    standard_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="standard",
    )

    # Both should be identical
    assert (
        abs(default_cost[0] - standard_cost[0]) < 1e-10
    ), "Default and standard prompt costs should be identical"
    assert (
        abs(default_cost[1] - standard_cost[1]) < 1e-10
    ), "Default and standard completion costs should be identical"

    # Verify specific costs match expected standard values
    # gpt-5-nano standard: input=5e-08, output=4e-07
    expected_standard_prompt = 1000 * 5e-08  # 0.00005
    expected_standard_completion = 500 * 4e-07  # 0.0002
    expected_standard_total = expected_standard_prompt + expected_standard_completion

    assert (
        abs(default_cost[0] - expected_standard_prompt) < 1e-10
    ), f"Standard prompt cost mismatch: {default_cost[0]} vs {expected_standard_prompt}"
    assert (
        abs(default_cost[1] - expected_standard_completion) < 1e-10
    ), f"Standard completion cost mismatch: {default_cost[1]} vs {expected_standard_completion}"


def test_service_tier_fallback_pricing(_local_model_cost_map):
    """Test that when service tier is provided but model doesn't have those keys, it falls back to standard pricing."""
    # Set up environment for local model cost map

    # Test with gpt-4 which doesn't have flex pricing keys
    model = "gpt-4"
    custom_llm_provider = "openai"

    # Create usage object
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    # Test standard pricing
    std_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=None,
    )
    std_total = std_cost[0] + std_cost[1]

    # Test flex pricing (should fall back to standard since gpt-4 doesn't have flex keys)
    flex_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="flex",
    )
    flex_total = flex_cost[0] + flex_cost[1]

    # Test priority pricing (should fall back to standard since gpt-4 doesn't have priority keys)
    priority_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="priority",
    )
    priority_total = priority_cost[0] + priority_cost[1]

    # All should be identical (fallback to standard)
    assert (
        abs(std_total - flex_total) < 1e-10
    ), f"Standard and flex costs should be identical (fallback): {std_total} vs {flex_total}"
    assert (
        abs(std_total - priority_total) < 1e-10
    ), f"Standard and priority costs should be identical (fallback): {std_total} vs {priority_total}"

    # Verify costs are reasonable (not zero)
    assert std_total > 0, "Standard cost should be greater than 0"
    assert flex_total > 0, "Flex cost should be greater than 0 (fallback)"
    assert priority_total > 0, "Priority cost should be greater than 0 (fallback)"

    # Verify specific costs match expected gpt-4 values
    # gpt-4 standard: input=3e-05, output=6e-05
    expected_standard_prompt = 1000 * 3e-05  # 0.03
    expected_standard_completion = 500 * 6e-05  # 0.03
    expected_standard_total = expected_standard_prompt + expected_standard_completion

    assert (
        abs(std_cost[0] - expected_standard_prompt) < 1e-10
    ), f"Standard prompt cost mismatch: {std_cost[0]} vs {expected_standard_prompt}"
    assert (
        abs(std_cost[1] - expected_standard_completion) < 1e-10
    ), f"Standard completion cost mismatch: {std_cost[1]} vs {expected_standard_completion}"


def test_service_tier_ultrafast_pricing():
    """An ultrafast request bills the *_ultrafast rates for all token types.

    Regression for the ultrafast service tier being absent from ServiceTier:
    the cost-key lookup silently returned the standard keys, undercounting
    every ultrafast request.
    """
    cached_tokens = 200
    cache_write_tokens = 300
    text_tokens = 500
    usage = Usage(
        prompt_tokens=text_tokens + cached_tokens + cache_write_tokens,
        completion_tokens=400,
        total_tokens=text_tokens + cached_tokens + cache_write_tokens + 400,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens
        ),
    )
    model_info: ModelInfo = {
        "key": "gpt-5.6-sol",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 3e-05,
        "cache_creation_input_token_cost": 6.25e-06,
        "cache_read_input_token_cost": 5e-07,
        "input_cost_per_token_ultrafast": 5e-05,
        "output_cost_per_token_ultrafast": 3e-04,
        "cache_creation_input_token_cost_ultrafast": 6.25e-05,
        "cache_read_input_token_cost_ultrafast": 5e-06,
    }

    prompt_cost, completion_cost = generic_cost_per_token(
        model="gpt-5.6-sol",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="ultrafast",
        model_info=model_info,
    )

    expected_prompt_cost = (
        text_tokens * 5e-05 + cached_tokens * 5e-06 + cache_write_tokens * 6.25e-05
    )
    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(400 * 3e-04)


def test_service_tier_ultrafast_fallback_pricing(_local_model_cost_map):
    """Without *_ultrafast keys an ultrafast request bills the standard rate, not zero.

    Guards the suffix fallback in _get_cost_per_unit: "_fast" is a substring of
    "_ultrafast", so a shortest-first suffix match would strip the wrong suffix
    and price the request at 0.
    """

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    std_prompt_cost, std_completion_cost = generic_cost_per_token(
        model="gpt-5.6-sol",
        usage=usage,
        custom_llm_provider="openai",
        service_tier=None,
    )
    ultrafast_prompt_cost, ultrafast_completion_cost = generic_cost_per_token(
        model="gpt-5.6-sol",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="ultrafast",
    )

    assert std_prompt_cost + std_completion_cost > 0
    assert ultrafast_prompt_cost == pytest.approx(std_prompt_cost)
    assert ultrafast_completion_cost == pytest.approx(std_completion_cost)


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-lite-image",
    ],
)
def test_gemini_image_generation_cost_with_zero_text_tokens(_local_model_cost_map, model: str):
    """
    Test that image_tokens are correctly costed when text_tokens=0.

    Reproduces issue #17410: completion_cost calculates incorrectly for
    Gemini-3-pro-image model - image_tokens were treated as text tokens
    when text_tokens=0.

    https://github.com/BerriAI/litellm/issues/17410
    """

    custom_llm_provider = "vertex_ai"

    # Usage from the issue: text_tokens=0, image_tokens=1120, reasoning_tokens=225
    usage = Usage(
        completion_tokens=1345,
        prompt_tokens=10,
        total_tokens=1355,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=225,
            rejected_prediction_tokens=None,
            text_tokens=0,  # This is the key: text_tokens=0
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=10, image_tokens=None
        ),
    )

    model_cost_map = litellm.model_cost[model]
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    # Expected costs:
    # - text_tokens: 0 * output_cost_per_token = 0
    # - image_tokens: 1120 * output_cost_per_image_token
    # - reasoning_tokens: 225 * output_cost_per_token
    # Total completion should include both image + reasoning costs.

    output_cost_per_image_token = model_cost_map.get("output_cost_per_image_token", 0)
    output_cost_per_token = model_cost_map.get("output_cost_per_token", 0)

    expected_image_cost = 1120 * output_cost_per_image_token
    expected_reasoning_cost = (
        225 * output_cost_per_token
    )  # reasoning uses base token cost
    expected_completion_cost = expected_image_cost + expected_reasoning_cost

    # The bug was: all completion tokens were treated as text tokens only.
    bugged_text_only_cost = 1345 * output_cost_per_token
    assert completion_cost > bugged_text_only_cost * 2, (
        f"Completion cost should be significantly larger than text-only bugged path. "
        f"Expected > {bugged_text_only_cost * 2:.6f}, got {completion_cost:.6f}"
    )
    assert round(completion_cost, 4) == round(
        expected_completion_cost, 4
    ), f"Expected completion cost ${expected_completion_cost:.6f}, got ${completion_cost:.6f}"


def test_vertex_image_generation_cost_prefers_token_usage_metadata(_local_model_cost_map):
    """
    When usage metadata exists on image responses, Vertex image generation cost
    should be calculated from token pricing, not flat output_cost_per_image.
    """

    model = "gemini-3.1-flash-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")

    input_text_tokens = 50
    input_image_tokens = 1120
    output_image_tokens = 1120
    prompt_tokens = input_text_tokens + input_image_tokens

    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")],
        usage=ImageUsage(
            input_tokens=prompt_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=input_image_tokens,
            ),
            output_tokens=output_image_tokens,
            total_tokens=prompt_tokens + output_image_tokens,
        ),
    )

    cost = vertex_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_prompt_cost = prompt_tokens * model_info["input_cost_per_token"]
    expected_completion_cost = (
        output_image_tokens * model_info["output_cost_per_image_token"]
    )
    expected_total_cost = expected_prompt_cost + expected_completion_cost

    assert round(cost, 10) == round(expected_total_cost, 10)
    # Ensure this is not falling back to flat per-image pricing.
    assert cost != len(image_response.data) * model_info["output_cost_per_image"]


def test_vertex_image_generation_cost_falls_back_to_flat_image_pricing(_local_model_cost_map):
    """
    Without usage metadata, Vertex image generation cost should fall back to
    output_cost_per_image * number_of_images.
    """

    model = "gemini-3.1-flash-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")

    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")]
    )

    cost = vertex_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = len(image_response.data) * model_info["output_cost_per_image"]
    assert round(cost, 10) == round(expected_cost, 10)


def test_gemini_image_generation_cost_prefers_token_usage_metadata(_local_model_cost_map):
    """
    When usage metadata exists on image responses, Gemini image generation cost
    should be calculated from token pricing, not flat output_cost_per_image.
    """

    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    input_text_tokens = 20
    input_image_tokens = 1120
    output_image_tokens = 1120
    prompt_tokens = input_text_tokens + input_image_tokens

    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")],
        usage=ImageUsage(
            input_tokens=prompt_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=input_image_tokens,
            ),
            output_tokens=output_image_tokens,
            total_tokens=prompt_tokens + output_image_tokens,
        ),
    )

    cost = gemini_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_prompt_cost = prompt_tokens * model_info["input_cost_per_token"]
    expected_completion_cost = (
        output_image_tokens * model_info["output_cost_per_image_token"]
    )
    expected_total_cost = expected_prompt_cost + expected_completion_cost

    assert round(cost, 10) == round(expected_total_cost, 10)
    # Ensure this is not falling back to flat per-image pricing.
    assert cost != len(image_response.data) * model_info["output_cost_per_image"]


def test_gemini_image_generation_cost_falls_back_to_flat_image_pricing(_local_model_cost_map):
    """
    Without usage metadata, Gemini image generation cost should fall back to
    output_cost_per_image * number_of_images.
    """

    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")]
    )

    cost = gemini_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = len(image_response.data) * model_info["output_cost_per_image"]
    assert round(cost, 10) == round(expected_cost, 10)


def test_bedrock_anthropic_prompt_caching():
    """Test Bedrock Anthropic models with prompt caching return correct costs."""
    model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    usage = Usage(
        prompt_tokens=52123,
        completion_tokens=497,
        total_tokens=52620,
        cache_creation_input_tokens=7183,
        cache_read_input_tokens=22465,
    )

    custom_llm_provider = "bedrock"

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    assert prompt_cost >= 0
    assert completion_cost >= 0
    assert round(prompt_cost, 3) == 0.111
    assert round(completion_cost, 5) == 0.00820


def test_reasoning_tokens_without_text_tokens_gpt5_nano():
    """
    Test fix for GitHub issue #18599:
    https://github.com/BerriAI/litellm/issues/18599

    When OpenAI models (gpt-5-nano, o1, o3) return reasoning_tokens but don't provide
    text_tokens, LiteLLM should calculate text_tokens as:
      text_tokens = completion_tokens - reasoning_tokens - audio_tokens - image_tokens

    This ensures ALL completion tokens are billed, not just reasoning tokens.
    """
    model = "gpt-5-nano"
    custom_llm_provider = "openai"

    # Simulate OpenAI gpt-5-nano response where text_tokens is NOT provided
    # completion_tokens: 977 total
    # reasoning_tokens: 768
    # text_tokens: should be calculated as 977 - 768 = 209
    usage = Usage(
        prompt_tokens=17,
        completion_tokens=977,
        total_tokens=994,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=768,
            audio_tokens=0,
            # text_tokens NOT provided - this is the key part of the bug
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    # gpt-5-nano pricing: $0.05/1M input, $0.40/1M output
    expected_prompt_cost = 17 * 0.05 / 1_000_000
    expected_completion_cost = 977 * 0.40 / 1_000_000  # ALL tokens, not just reasoning

    assert (
        abs(prompt_cost - expected_prompt_cost) < 1e-10
    ), f"Prompt cost incorrect: {prompt_cost} vs {expected_prompt_cost}"

    assert (
        abs(completion_cost - expected_completion_cost) < 1e-10
    ), f"Completion cost incorrect: {completion_cost} vs {expected_completion_cost}"

    # Verify it's NOT using only reasoning_tokens (the bug)
    wrong_cost = 768 * 0.40 / 1_000_000  # Only reasoning tokens
    assert (
        abs(completion_cost - wrong_cost) > 1e-6
    ), "Bug detected: Cost calculation is using only reasoning_tokens instead of all completion_tokens!"


def test_image_count_prevents_text_tokens_fallback(_local_model_cost_map):
    """
    Test that the text_tokens fallback in generic_cost_per_token does not
    override text_tokens=0 when image_count > 0.

    Regression test for: Bedrock image embedding double-charging bug.
    When image_count > 0, text_tokens=0 is intentional (image-only request),
    not "text_tokens not set by provider."
    """

    # Simulate Nova image-only embedding: prompt_tokens estimated from
    # embedding dimensions (768 for 3072-dim), image_count=1
    usage = Usage(
        prompt_tokens=768,
        completion_tokens=0,
        total_tokens=768,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            image_count=1,
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="amazon.nova-2-multimodal-embeddings-v1:0",
        usage=usage,
        custom_llm_provider="bedrock",
    )

    # Cost should be 1 * input_cost_per_image ($6e-05) = $0.00006
    # NOT 768 * input_cost_per_token ($1.35e-07) + $0.00006 = $0.000164
    expected_image_cost = 1 * 6e-05
    assert prompt_cost == expected_image_cost, (
        f"Expected prompt_cost={expected_image_cost} (image-only), "
        f"got {prompt_cost}. text_tokens fallback may be double-charging."
    )
    assert completion_cost == 0.0


# ---------------------------------------------------------------------------
# Data-residency (OpenAI regional processing) tests
# ---------------------------------------------------------------------------




@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-realtime-2.1", "gpt-realtime-2.1-mini"])
@pytest.mark.parametrize("data_residency", ["eu", "us"])
def test_data_residency_applies_uplift(data_residency, model, _local_model_cost_map):
    """Models released on/after 2026-03-05 (gpt-5.4/5.5 and gpt-realtime-2.1
    series) apply the 10% regional processing uplift multiplier when
    data_residency is set; gpt-5 and older models do not."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
    )
    regional = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
        data_residency=data_residency,
    )

    base_total = base[0] + base[1]
    regional_total = regional[0] + regional[1]

    assert base_total > 0
    assert regional_total == pytest.approx(base_total * 1.10, rel=1e-9)
    assert regional[0] == pytest.approx(base[0] * 1.10, rel=1e-9)
    assert regional[1] == pytest.approx(base[1] * 1.10, rel=1e-9)


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-4o", "gpt-4.1"])
def test_data_residency_no_uplift_for_pre_march_2026_models(model, _local_model_cost_map):
    """Models released before 2026-03-05 must not have the regional uplift."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")
    regional = generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="openai", data_residency="eu"
    )

    assert base == regional, (
        f"{model} should not have a regional uplift, but cost changed with data_residency"
    )


def test_data_residency_no_uplift_for_unmarked_model(_local_model_cost_map):
    """A model without a regional_processing_uplift_multiplier_* entry should
    fall back to base pricing, not error."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(
        model="gpt-3.5-turbo",
        usage=usage,
        custom_llm_provider="openai",
    )
    with_residency = generic_cost_per_token(
        model="gpt-3.5-turbo",
        usage=usage,
        custom_llm_provider="openai",
        data_residency="eu",
    )

    assert base == with_residency


def test_data_residency_none_no_uplift(_local_model_cost_map):
    """data_residency=None should be a no-op even for models with a multiplier."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
    )
    explicit_none = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
        data_residency=None,
    )

    assert base == explicit_none


def test_data_residency_composes_with_service_tier(_local_model_cost_map):
    """The uplift multiplies the priority-tier cost, not the standard one."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    priority_base = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
    )
    priority_eu = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        data_residency="eu",
    )

    priority_base_total = priority_base[0] + priority_base[1]
    priority_eu_total = priority_eu[0] + priority_eu[1]

    assert priority_base_total > 0
    assert priority_eu_total == pytest.approx(priority_base_total * 1.10, rel=1e-9)


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "claude-haiku-4-5@20251001"])
@pytest.mark.parametrize("vertex_location", ["us-central1", "us-east5", "europe-west1", "asia-southeast1"])
def test_vertex_regional_location_applies_uplift(vertex_location, model, _local_model_cost_map):
    """Google bills every non-global Vertex endpoint at 1.1x the global rate for GA
    Gemini 3+ and regional-pricing Claude models, so a request served from a regional
    location must cost 1.1x what the same usage costs on the global endpoint."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="vertex_ai")
    regional = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="vertex_ai",
        vertex_location=vertex_location,
    )

    base_total = base[0] + base[1]
    regional_total = regional[0] + regional[1]

    assert base_total > 0
    assert regional_total == pytest.approx(base_total * 1.10, rel=1e-9)
    assert regional[0] == pytest.approx(base[0] * 1.10, rel=1e-9)
    assert regional[1] == pytest.approx(base[1] * 1.10, rel=1e-9)


@pytest.mark.parametrize("vertex_location", [None, "global", "GLOBAL"])
def test_vertex_global_or_absent_location_no_uplift(vertex_location, _local_model_cost_map):
    """The global endpoint prices at the base rate, whatever the casing, and an
    unresolved location must never uplift."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(
        model="claude-haiku-4-5@20251001", usage=usage, custom_llm_provider="vertex_ai"
    )
    located = generic_cost_per_token(
        model="claude-haiku-4-5@20251001",
        usage=usage,
        custom_llm_provider="vertex_ai",
        vertex_location=vertex_location,
    )

    assert base == located


@pytest.mark.parametrize("model", ["claude-opus-4-1", "gemini-2.0-flash-001"])
def test_vertex_location_no_uplift_for_uniformly_priced_model(model, _local_model_cost_map):
    """Models Google prices uniformly across endpoints (Gemini 2.x, Claude Opus 4.1
    and older) carry no multiplier and must not move with the location."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="vertex_ai")
    regional = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="vertex_ai",
        vertex_location="us-east5",
    )

    assert base == regional, f"{model} should not have a regional-endpoint uplift"


def test_vertex_uplift_invalid_multiplier_defaults_to_one():
    """A malformed multiplier in the cost map degrades to base pricing, never raises."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import (
        get_vertex_regional_endpoint_uplift,
    )

    assert (
        get_vertex_regional_endpoint_uplift(
            {"regional_endpoint_uplift_multiplier": "not-a-number"}, "us-east5"
        )
        == 1.0
    )


def test_priority_service_tier_above_threshold_uses_priority_tier_rates_for_cached_tokens(
    _local_model_cost_map,
):
    """Regression: for a model that publishes both service_tier and above_threshold rate
    variants, a priority request over the threshold must bill cached tokens at
    cache_read_input_token_cost_above_200k_tokens_priority (and analogously for
    input/output above-threshold), not the standard above-threshold rate."""
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=200_000, text_tokens=50_000
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=1_000),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="gemini-3-pro-preview",
        usage=usage,
        custom_llm_provider="gemini",
        service_tier="priority",
    )

    # gemini-3-pro-preview priority + above_200k rates from the pricing JSON:
    #   input  7.2e-6, output 3.24e-5, cache_read 7.2e-7
    expected_prompt = 50_000 * 7.2e-6 + 200_000 * 7.2e-7
    expected_completion = 1_000 * 3.24e-5
    assert prompt_cost == pytest.approx(expected_prompt, rel=1e-9)
    assert completion_cost == pytest.approx(expected_completion, rel=1e-9)


def test_priority_service_tier_above_threshold_falls_back_to_standard_for_cache_creation(
    _local_model_cost_map,
):
    """Regression: priority requests against models that publish standard above-threshold
    cache_creation rates but no priority variant must fall back to the standard
    above-threshold rate, not the priority-base rate. vertex_ai/claude-sonnet-4-5
    has cache_creation_input_token_cost_above_200k_tokens but no _priority sibling."""
    usage = Usage(
        prompt_tokens=350_000,
        completion_tokens=1_000,
        total_tokens=351_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=200_000,
            cache_creation_tokens=100_000,
            text_tokens=50_000,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=1_000),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="vertex_ai/claude-sonnet-4-5",
        usage=usage,
        custom_llm_provider="vertex_ai",
        service_tier="priority",
    )

    # vertex_ai/claude-sonnet-4-5 above_200k (no _priority variants):
    #   input 6e-6, output 2.25e-5, cache_read 6e-7, cache_creation 7.5e-6
    # text                  50_000  * 6e-6   = 0.30
    # cache_read           200_000  * 6e-7   = 0.12
    # cache_creation       100_000  * 7.5e-6 = 0.75
    expected_prompt = 50_000 * 6e-6 + 200_000 * 6e-7 + 100_000 * 7.5e-6
    expected_completion = 1_000 * 2.25e-5
    assert prompt_cost == pytest.approx(expected_prompt, rel=1e-9)
    assert completion_cost == pytest.approx(expected_completion, rel=1e-9)


def test_service_tier_suffixes_constant_in_sync_with_enum():
    from litellm.litellm_core_utils.llm_cost_calc.utils import _SERVICE_TIER_SUFFIXES
    from litellm.types.utils import ServiceTier

    assert set(_SERVICE_TIER_SUFFIXES) == {f"_{st.value}" for st in ServiceTier}
    # longest-first so a substring match resolves "_ultrafast" before "_fast"
    assert list(_SERVICE_TIER_SUFFIXES) == sorted(
        _SERVICE_TIER_SUFFIXES, key=len, reverse=True
    )


def test_get_cost_per_unit_falls_back_from_service_tier_key_to_base():
    from litellm.litellm_core_utils.llm_cost_calc.utils import _get_cost_per_unit

    model_info = {"input_cost_per_token": 2e-6}
    # service-tier key is absent -> falls back to the base key
    assert _get_cost_per_unit(model_info, "input_cost_per_token_priority") == 2e-6
    # service-tier key present -> used directly, no fallback
    model_info_direct = {
        "input_cost_per_token_priority": 5e-6,
        "input_cost_per_token": 2e-6,
    }
    assert (
        _get_cost_per_unit(model_info_direct, "input_cost_per_token_priority") == 5e-6
    )


def test_threshold_keys_exclude_service_tier_variants():
    from typing import cast

    from litellm.litellm_core_utils.llm_cost_calc.utils import _get_token_base_cost
    from litellm.types.utils import ModelInfo, Usage

    # The service-tier-suffixed above-threshold key must be excluded from
    # threshold detection. The _priority variant has a higher threshold (300k),
    # so if it were not excluded it would sort first and drive a 9e-6 rate for
    # this non-tier request. With the exclusion only the standard 200k key
    # applies, giving 3e-6.
    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "input_cost_per_token_above_200k_tokens": 3e-6,
            "input_cost_per_token_above_300k_tokens_priority": 9e-6,
            "output_cost_per_token": 2e-6,
        },
    )
    usage = Usage(prompt_tokens=350_000, completion_tokens=1_000, total_tokens=351_000)
    prompt_base, *_ = _get_token_base_cost(model_info=model_info, usage=usage)
    assert prompt_base == 3e-6


@pytest.mark.parametrize(
    "model,custom_llm_provider,reasoning_tokens,cached_tokens",
    [
        ("gemini-2.5-flash", "vertex_ai", 3114, 100),
        ("o3", "openai", 500, 200),
        ("azure/gpt-5", "azure", 300, 150),
        ("us.amazon.nova-2-lite-v1:0", "bedrock", 120, 80),
        ("perplexity/sonar-reasoning", "perplexity", 400, 0),
        ("cerebras/qwen-3-32b", "cerebras", 250, 0),
    ],
)
def test_token_type_cost_breakdown_is_provider_agnostic(_local_model_cost_map, 
    model, custom_llm_provider, reasoning_tokens, cached_tokens
):
    """
    Reasoning and cache-read costs must be surfaced for every provider that reports
    those tokens, regardless of which cost calculator the provider routes through
    (Perplexity, Cerebras, Dashscope bypass generic_cost_per_token entirely).

    Cache tokens always land in prompt_tokens_details.cached_tokens, so reading from
    there - not the top-level cache_read_input_tokens attribute the old breakdown code
    relied on - is what makes Vertex/OpenAI/Azure cache costs show up at all.
    """

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=2000,
        total_tokens=3000,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=reasoning_tokens, text_tokens=2000 - reasoning_tokens
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens, text_tokens=1000 - cached_tokens
        ),
    )

    breakdown = get_token_type_cost_breakdown(
        model=model, custom_llm_provider=custom_llm_provider, usage=usage
    )

    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )
    reasoning_rate = (
        model_info.get("output_cost_per_reasoning_token")
        or model_info["output_cost_per_token"]
    )
    cache_read_rate = model_info.get("cache_read_input_token_cost") or 0.0

    assert breakdown.reasoning_cost == pytest.approx(reasoning_tokens * reasoning_rate)
    assert breakdown.cache_read_cost == pytest.approx(cached_tokens * cache_read_rate)


def test_token_type_cost_breakdown_matches_real_gemini_numbers(_local_model_cost_map):
    """Hard-coded against the exact gemini-2.5-flash response that exposed the gap."""

    usage = Usage(
        prompt_tokens=209,
        completion_tokens=3996,
        total_tokens=4205,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=3114, text_tokens=882
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=100, text_tokens=109
        ),
    )

    breakdown = get_token_type_cost_breakdown(
        model="gemini-2.5-flash", custom_llm_provider="vertex_ai", usage=usage
    )

    assert breakdown.reasoning_cost == pytest.approx(3114 * 2.5e-06)
    assert breakdown.cache_read_cost == pytest.approx(100 * 3e-08)
    assert breakdown.cache_creation_cost == 0.0


def test_token_type_cost_breakdown_flex_tier_prices_reasoning_at_flex_rate(_local_model_cost_map):
    """Regression for the flex-tier breakdown drift: gemini-3.5-flash defines a flat
    output_cost_per_reasoning_token (9e-06, the standard output rate) but no _flex
    variant, so the breakdown priced reasoning at the standard rate on flex requests
    while the total billed it at the flex output rate (4.5e-06). The reasoning
    sub-cost then exceeded the entire flex completion cost."""

    usage = Usage(
        prompt_tokens=7,
        completion_tokens=320,
        total_tokens=327,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=315, text_tokens=5),
    )

    breakdown = get_token_type_cost_breakdown(
        model="gemini-3.5-flash",
        custom_llm_provider="vertex_ai",
        usage=usage,
        service_tier="flex",
    )

    assert breakdown.reasoning_cost == pytest.approx(315 * 4.5e-06)

    _, flex_completion_cost = generic_cost_per_token(
        model="gemini-3.5-flash",
        usage=usage,
        custom_llm_provider="vertex_ai",
        service_tier="flex",
    )
    assert breakdown.reasoning_cost <= flex_completion_cost

    standard_breakdown = get_token_type_cost_breakdown(
        model="gemini-3.5-flash",
        custom_llm_provider="vertex_ai",
        usage=usage,
        service_tier=None,
    )
    assert standard_breakdown.reasoning_cost == pytest.approx(315 * 9e-06)


def test_token_type_cost_breakdown_xai_at_exactly_200k_uses_higher_tier_rates(_local_model_cost_map):

    usage = Usage(
        prompt_tokens=200_000,
        completion_tokens=2_000,
        total_tokens=202_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=1_500, text_tokens=500
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=50_000, text_tokens=150_000
        ),
    )

    breakdown = get_token_type_cost_breakdown(
        model="grok-4.20-0309-reasoning", custom_llm_provider="xai", usage=usage
    )

    assert breakdown.reasoning_cost == pytest.approx(1_500 * 5e-06)
    assert breakdown.cache_read_cost == pytest.approx(50_000 * 4e-07)


def test_token_type_cost_breakdown_xai_just_below_200k_uses_base_tier_rates(_local_model_cost_map):

    usage = Usage(
        prompt_tokens=199_999,
        completion_tokens=2_000,
        total_tokens=201_999,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=1_500, text_tokens=500
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=50_000, text_tokens=149_999
        ),
    )

    breakdown = get_token_type_cost_breakdown(
        model="grok-4.20-0309-reasoning", custom_llm_provider="xai", usage=usage
    )

    assert breakdown.reasoning_cost == pytest.approx(1_500 * 2.5e-06)
    assert breakdown.cache_read_cost == pytest.approx(50_000 * 2e-07)


def test_token_type_cost_breakdown_includes_cache_creation_from_top_level_usage(_local_model_cost_map):
    """
    Bedrock/Anthropic report cache tokens as top-level usage fields; the Usage
    constructor maps them onto prompt_tokens_details, so the breakdown must still
    pick up both cache-read and cache-creation costs.
    """

    model = "anthropic.claude-3-5-haiku-20241022-v1:0"
    usage = Usage(
        prompt_tokens=500,
        completion_tokens=50,
        total_tokens=550,
        cache_creation_input_tokens=300,
        cache_read_input_tokens=120,
    )

    breakdown = get_token_type_cost_breakdown(
        model=model, custom_llm_provider="bedrock", usage=usage
    )

    model_info = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    assert breakdown.cache_creation_cost == pytest.approx(
        300 * model_info["cache_creation_input_token_cost"]
    )
    assert breakdown.cache_read_cost == pytest.approx(
        120 * model_info["cache_read_input_token_cost"]
    )


def test_token_type_cost_breakdown_reads_cache_write_tokens(_local_model_cost_map):
    """
    Some OpenAI-compatible providers (e.g. kimi-k2) report cache-write tokens under
    `cache_write_tokens` rather than `cache_creation_tokens`. The breakdown must read
    it the same way the total-cost normalization does, so the two agree.
    """

    model = "anthropic.claude-3-5-haiku-20241022-v1:0"
    usage = Usage(
        prompt_tokens=500,
        completion_tokens=50,
        total_tokens=550,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=0, cache_write_tokens=300
        ),
    )

    breakdown = get_token_type_cost_breakdown(
        model=model, custom_llm_provider="bedrock", usage=usage
    )
    model_info = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    assert breakdown.cache_creation_cost == pytest.approx(
        300 * model_info["cache_creation_input_token_cost"]
    )


def test_generic_cost_per_token_openai_cache_write_tokens_gpt_5_6(_local_model_cost_map):
    """
    Regression: OpenAI gpt-5.6 reports cache-write tokens under
    prompt_tokens_details.cache_write_tokens (not the Anthropic cache_creation_tokens
    name). Those tokens must be billed at the cache-write rate rather than the plain
    input rate. Customer report: cache creation tokens were never counted for the
    GPT-5.6 series, so cost was undercounted on cache-write requests.
    """

    model = "gpt-5.6"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=0, cache_write_tokens=800),
    )

    assert usage.prompt_tokens_details.cache_write_tokens == 800
    assert usage.prompt_tokens_details.cache_creation_tokens == 800

    prompt_cost, _ = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    expected_prompt = (1000 - 800) * info["input_cost_per_token"] + 800 * info["cache_creation_input_token_cost"]
    assert prompt_cost == pytest.approx(expected_prompt)
    assert info["cache_creation_input_token_cost"] > info["input_cost_per_token"]
    assert prompt_cost > 1000 * info["input_cost_per_token"]


def test_generic_cost_per_token_backs_out_cache_write_tokens_from_text_tokens(_local_model_cost_map):
    """
    Regression for #34801: when a provider reports text_tokens covering the whole
    prompt alongside cache-write tokens (and no cache reads), the cache-write tokens
    must be backed out of the text total instead of being billed twice.
    """

    model = "gpt-5.6"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=0, cache_write_tokens=800, text_tokens=1000
        ),
    )

    prompt_cost, _ = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    expected_prompt = 200 * info["input_cost_per_token"] + 800 * info["cache_creation_input_token_cost"]
    assert prompt_cost == pytest.approx(expected_prompt)


def test_token_type_cost_breakdown_reconciles_with_generic_total(_local_model_cost_map):
    """
    Both-ways check: the reasoning subset must sum with the remaining (text) output
    cost to exactly the completion total, and the cache-read subset with the remaining
    input cost to exactly the prompt total, as computed by generic_cost_per_token.
    A mismatch here would mean the breakdown misrepresents what was actually billed.
    """

    model = "gemini-2.5-flash"
    custom_llm_provider = "vertex_ai"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=2000,
        total_tokens=3000,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=1200, text_tokens=800
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=300, text_tokens=700
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider=custom_llm_provider
    )
    breakdown = get_token_type_cost_breakdown(
        model=model, custom_llm_provider=custom_llm_provider, usage=usage
    )

    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )
    text_output_cost = 800 * model_info["output_cost_per_token"]
    text_input_cost = 700 * model_info["input_cost_per_token"]

    assert text_output_cost + breakdown.reasoning_cost == pytest.approx(completion_cost)
    assert text_input_cost + breakdown.cache_read_cost == pytest.approx(prompt_cost)


def test_token_type_cost_breakdown_zero_without_special_tokens(_local_model_cost_map):

    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    breakdown = get_token_type_cost_breakdown(
        model="gpt-4o", custom_llm_provider="openai", usage=usage
    )

    assert breakdown == TokenTypeCostBreakdown(
        reasoning_cost=0.0, cache_read_cost=0.0, cache_creation_cost=0.0
    )


@pytest.mark.parametrize(
    "raw_usage, expect_read, expect_write",
    [
        (
            {
                "input_tokens": 5000,
                "output_tokens": 10,
                "total_tokens": 5010,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4012},
            },
            False,
            True,
        ),
        (
            {
                "input_tokens": 5000,
                "output_tokens": 10,
                "total_tokens": 5010,
                "input_tokens_details": {"cached_tokens": 4012, "cache_write_tokens": 0},
            },
            True,
            False,
        ),
    ],
)
def test_token_type_cost_breakdown_openai_responses_api_cache_write_read(_local_model_cost_map, 
    raw_usage, expect_read, expect_write
):
    """Regression for #34309: OpenAI Responses API reports cache tokens under
    input_tokens_details.{cached_tokens, cache_write_tokens}, not the Anthropic-style
    top-level cache_creation_input_tokens. The itemized breakdown must still populate
    cache_read_cost / cache_creation_cost from the transformed usage."""
    from litellm.responses.utils import ResponseAPILoggingUtils


    model = "gpt-5.6"
    usage = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(raw_usage)

    breakdown = get_token_type_cost_breakdown(
        model=model, custom_llm_provider="openai", usage=usage
    )

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    if expect_write:
        assert breakdown.cache_creation_cost == pytest.approx(
            4012 * info["cache_creation_input_token_cost"]
        )
        assert breakdown.cache_creation_cost > 0
        assert breakdown.cache_read_cost == 0.0
    if expect_read:
        assert breakdown.cache_read_cost == pytest.approx(
            4012 * info["cache_read_input_token_cost"]
        )
        assert breakdown.cache_read_cost > 0
        assert breakdown.cache_creation_cost == 0.0


def test_token_type_cost_breakdown_handles_unknown_model_gracefully():
    """A model with no pricing must yield zeros, never raise."""
    breakdown = get_token_type_cost_breakdown(
        model="this-model-does-not-exist-anywhere",
        custom_llm_provider="openai",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=5),
        ),
    )
    assert breakdown == TokenTypeCostBreakdown(
        reasoning_cost=0.0, cache_read_cost=0.0, cache_creation_cost=0.0
    )


def test_token_type_cost_breakdown_applies_regional_uplift(_local_model_cost_map):
    """
    Regional OpenAI hosts (eu./us.) apply a flat uplift to every token cost. The
    per-type breakdown must apply the same uplift via data_residency so it stays
    reconciled with the uplifted input_cost/output_cost totals, instead of being
    logged at the base rate.
    """

    model = "gpt-5.4"
    custom_llm_provider = "openai"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=200, text_tokens=300
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=400, text_tokens=600
        ),
    )

    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )
    uplift = model_info["regional_processing_uplift_multiplier_eu"]
    assert uplift > 1.0

    base = get_token_type_cost_breakdown(
        model=model, custom_llm_provider=custom_llm_provider, usage=usage
    )
    eu = get_token_type_cost_breakdown(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        data_residency="eu",
    )

    assert eu.reasoning_cost == pytest.approx(base.reasoning_cost * uplift)
    assert eu.cache_read_cost == pytest.approx(base.cache_read_cost * uplift)

    # The uplifted breakdown must still reconcile with the uplifted totals.
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        data_residency="eu",
    )
    text_output_cost = 300 * model_info["output_cost_per_token"] * uplift
    text_input_cost = 600 * model_info["input_cost_per_token"] * uplift
    assert text_output_cost + eu.reasoning_cost == pytest.approx(completion_cost)
    assert text_input_cost + eu.cache_read_cost == pytest.approx(prompt_cost)


def test_token_type_cost_breakdown_applies_vertex_regional_uplift(_local_model_cost_map):
    """
    Non-global Vertex endpoints apply a flat 1.1x uplift to every token cost. The
    per-type breakdown must apply the same uplift via vertex_location so it stays
    reconciled with the uplifted input_cost/output_cost totals, instead of being
    logged at the global rate.
    """

    model = "claude-haiku-4-5@20251001"
    custom_llm_provider = "vertex_ai"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=400, text_tokens=600
        ),
    )

    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )
    uplift = model_info["regional_endpoint_uplift_multiplier"]
    assert uplift > 1.0

    base = get_token_type_cost_breakdown(
        model=model, custom_llm_provider=custom_llm_provider, usage=usage
    )
    regional = get_token_type_cost_breakdown(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        vertex_location="us-east5",
    )

    assert base.cache_read_cost > 0
    assert regional.cache_read_cost == pytest.approx(base.cache_read_cost * uplift)

    # The uplifted breakdown must still reconcile with the uplifted totals.
    prompt_cost, _completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        vertex_location="us-east5",
    )
    text_input_cost = 600 * model_info["input_cost_per_token"] * uplift
    assert text_input_cost + regional.cache_read_cost == pytest.approx(prompt_cost)


def test_token_type_cost_breakdown_applies_anthropic_geo_multiplier(_local_model_cost_map, monkeypatch):
    """
    Anthropic's regional (geo) uplift lives in provider_specific_entry and is
    applied to every token type in the totals, so the per-type breakdown must
    scale its cache and reasoning line items by it too. Otherwise the logged
    cache costs stay at the base rate and the cache uplift is misattributed to
    plain input for exactly the cache-heavy regional traffic the uplift targets.
    """
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    model = "claude-test-geo-breakdown-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 5e-6,
                "output_cost_per_token": 25e-6,
                "cache_creation_input_token_cost": 6.25e-6,
                "cache_read_input_token_cost": 0.5e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
                "provider_specific_entry": {"us": 1.1},
            }
        }
    )

    def make_usage() -> Usage:
        return Usage(
            prompt_tokens=10_000,
            completion_tokens=500,
            total_tokens=10_500,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=2_000,
                cache_creation_tokens=6_000,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=200, text_tokens=300
            ),
        )

    base_usage = make_usage()
    geo_usage = make_usage()
    geo_usage.inference_geo = "us"

    base = get_token_type_cost_breakdown(
        model=model, custom_llm_provider="anthropic", usage=base_usage
    )
    geo = get_token_type_cost_breakdown(
        model=model, custom_llm_provider="anthropic", usage=geo_usage
    )

    assert base.cache_read_cost == pytest.approx(2_000 * 0.5e-6)
    assert base.cache_creation_cost == pytest.approx(6_000 * 6.25e-6)
    assert geo.cache_read_cost == pytest.approx(base.cache_read_cost * 1.1)
    assert geo.cache_creation_cost == pytest.approx(base.cache_creation_cost * 1.1)
    assert geo.reasoning_cost == pytest.approx(base.reasoning_cost * 1.1)

    # The uplifted breakdown must still reconcile with the uplifted totals.
    prompt_cost, completion_cost = anthropic_cost_per_token(model=model, usage=geo_usage)
    text_input_cost = 2_000 * 5e-6 * 1.1
    text_output_cost = 300 * 25e-6 * 1.1
    assert text_input_cost + geo.cache_read_cost + geo.cache_creation_cost == pytest.approx(prompt_cost)
    assert text_output_cost + geo.reasoning_cost == pytest.approx(completion_cost)


@pytest.mark.parametrize("details_as_dict", [True, False])
def test_image_response_input_image_tokens_priced_at_image_rate(details_as_dict):
    """
    Image input tokens must be priced at input_cost_per_image_token even when
    input_tokens_details is a plain dict, as in OpenAI image edit responses.

    Regression test: dict-shaped input_tokens_details was read with getattr(),
    which returns None for dicts, so image input tokens silently fell back to
    the text input rate (e.g. $5/M instead of $8/M for gpt-image-2).
    """
    from unittest.mock import patch

    from litellm.litellm_core_utils.llm_cost_calc.utils import (
        calculate_image_response_cost_from_usage,
    )
    from litellm.types.utils import Usage

    mock_model_info = {
        "input_cost_per_token": 5e-6,
        "input_cost_per_image_token": 8e-6,
        "output_cost_per_image_token": 3e-5,
    }

    input_details = {"text_tokens": 19, "image_tokens": 512}
    image_response = ImageResponse(data=[ImageObject(b64_json="x")])
    # Mirror the usage shape of a real OpenAI images.edit response:
    # a Usage object carrying input_tokens/output_tokens with detail dicts.
    image_response.usage = Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=689,
        input_tokens=531,
        input_tokens_details=(
            input_details
            if details_as_dict
            else ImageUsageInputTokensDetails(**input_details)
        ),
        output_tokens=158,
        output_tokens_details={"image_tokens": 158, "text_tokens": 0},
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        cost = calculate_image_response_cost_from_usage(
            model="gpt-image-2",
            image_response=image_response,
            custom_llm_provider="openai",
        )

    expected = 19 * 5e-6 + 512 * 8e-6 + 158 * 3e-5
    assert cost is not None
    assert round(cost, 12) == round(expected, 12)
GEMINI_DAY0_LAUNCH_PRICING = [
    ("gemini-3.6-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("gemini/gemini-3.6-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("vertex_ai/gemini-3.6-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("gemini-3.5-flash-lite", 3e-07, 2.5e-06, 3e-08),
    ("gemini/gemini-3.5-flash-lite", 3e-07, 2.5e-06, 3e-08),
    ("vertex_ai/gemini-3.5-flash-lite", 3e-07, 2.5e-06, 3e-08),
]


@pytest.mark.parametrize("model,input_cost,output_cost,cache_read_cost", GEMINI_DAY0_LAUNCH_PRICING)
def test_gemini_36_flash_and_35_flash_lite_launch_pricing(_local_model_cost_map, model, input_cost, output_cost, cache_read_cost):

    model_cost_map = litellm.model_cost[model]
    assert model_cost_map["input_cost_per_token"] == input_cost
    assert model_cost_map["output_cost_per_token"] == output_cost
    assert model_cost_map["output_cost_per_reasoning_token"] == output_cost
    assert model_cost_map["cache_read_input_token_cost"] == cache_read_cost
    assert model_cost_map["mode"] == "chat"
    assert model_cost_map["supports_reasoning"] is True
    assert model_cost_map["supports_function_calling"] is True
    assert model_cost_map["max_input_tokens"] == 1048576


def test_generic_cost_per_token_gemini_36_flash(_local_model_cost_map):

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=200,
            text_tokens=300,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=1000),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model="gemini-3.6-flash",
        usage=usage,
        custom_llm_provider="gemini",
    )
    assert prompt_cost == pytest.approx(0.00075)
    assert completion_cost == pytest.approx(0.001875)


GEMINI_36_FLASH_SERVICE_TIER_PRICING = [
    (None, 7.5e-07, 3.75e-06, 7.5e-08),
    ("flex", 3.75e-07, 1.875e-06, 3.75e-08),
    ("priority", 1.35e-06, 6.75e-06, 1.35e-07),
]


@pytest.mark.parametrize(
    "service_tier,input_rate,output_rate,cache_read_rate", GEMINI_36_FLASH_SERVICE_TIER_PRICING
)
@pytest.mark.parametrize(
    "model", ["gemini-3.6-flash", "gemini/gemini-3.6-flash", "vertex_ai/gemini-3.6-flash"]
)
def test_gemini_36_flash_service_tier_introductory_pricing(
    model, service_tier, input_rate, output_rate, cache_read_rate, _local_model_cost_map
):
    """Regression: every 3.6 Flash tier is on Google's introductory rates through 2026-12-31,
    so flex and priority requests must not be billed at the post-introductory rates."""
    usage = Usage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=200, text_tokens=800),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model.split("/")[-1],
        usage=usage,
        custom_llm_provider=model.split("/")[0] if "/" in model else "gemini",
        service_tier=service_tier,
    )

    assert prompt_cost == pytest.approx(800 * input_rate + 200 * cache_read_rate, rel=1e-9)
    assert completion_cost == pytest.approx(500 * output_rate, rel=1e-9)


@pytest.mark.parametrize(
    "model", ["gemini-3.6-flash", "gemini/gemini-3.6-flash", "vertex_ai/gemini-3.6-flash"]
)
def test_gemini_36_flash_batch_introductory_pricing(model, _local_model_cost_map):
    model_cost_map = litellm.model_cost[model]
    assert model_cost_map["input_cost_per_token_batches"] == 3.75e-07
    assert model_cost_map["output_cost_per_token_batches"] == 1.875e-06


def test_generic_cost_per_token_gemini_35_flash_lite(_local_model_cost_map):

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=200,
            text_tokens=300,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=1000),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model="gemini-3.5-flash-lite",
        usage=usage,
        custom_llm_provider="gemini",
    )
    assert prompt_cost == pytest.approx(0.0003)
    assert completion_cost == pytest.approx(0.00125)


GEMINI_35_FLASH_LITE_TIER_RATES_BY_SURFACE = [
    ("gemini", None, 3e-07, 2.5e-06, 3e-08),
    ("gemini", "flex", 1.5e-07, 1.25e-06, 2e-08),
    ("gemini", "priority", 5.4e-07, 4.5e-06, 5e-08),
    ("vertex_ai", None, 3e-07, 2.5e-06, 3e-08),
    ("vertex_ai", "flex", 1.5e-07, 1.25e-06, 1.5e-08),
    ("vertex_ai", "priority", 5.4e-07, 4.5e-06, 5e-08),
]


@pytest.mark.parametrize(
    "custom_llm_provider,service_tier,input_rate,output_rate,cache_read_rate",
    GEMINI_35_FLASH_LITE_TIER_RATES_BY_SURFACE,
)
def test_gemini_35_flash_lite_service_tier_pricing(
    custom_llm_provider, service_tier, input_rate, output_rate, cache_read_rate, _local_model_cost_map
):
    """Regression: Vertex publishes flash-lite flex context caching at $0.015/M while the
    Gemini API publishes $0.02/M, so vertex_ai flex cache reads must bill 1.5e-08/token
    instead of the 2e-08 the map used to carry, without disturbing the Gemini API rate."""
    usage = Usage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=200, text_tokens=800),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="gemini-3.5-flash-lite",
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=service_tier,
    )

    assert prompt_cost == pytest.approx(800 * input_rate + 200 * cache_read_rate, rel=1e-9)
    assert completion_cost == pytest.approx(500 * output_rate, rel=1e-9)


def test_gemini_35_flash_lite_flex_cache_read_map_entries(_local_model_cost_map):
    """Each map entry carries its own surface's published flex cache-read rate: the bare
    and vertex_ai keys are the Vertex surface at $0.015/M, the gemini key is the Gemini
    API surface at $0.02/M."""
    assert litellm.model_cost["gemini-3.5-flash-lite"]["cache_read_input_token_cost_flex"] == 1.5e-08
    assert litellm.model_cost["vertex_ai/gemini-3.5-flash-lite"]["cache_read_input_token_cost_flex"] == 1.5e-08
    assert litellm.model_cost["gemini/gemini-3.5-flash-lite"]["cache_read_input_token_cost_flex"] == 2e-08


@pytest.mark.parametrize(
    "service_tier,input_rate,cache_read_rate,cache_write_rate,output_rate",
    [
        ("flex", 2e-6, 2e-7, 2.5e-6, 1e-5),
        ("priority", 8e-6, 8e-7, 1e-5, 4e-5),
    ],
)
def test_service_tier_cache_creation_rates_for_gpt_5_6(
    _local_model_cost_map,
    service_tier,
    input_rate,
    cache_read_rate,
    cache_write_rate,
    output_rate,
):
    """Regression: gpt-5.6 publishes cache_creation_input_token_cost_flex/_priority, so a
    flex or priority request must bill cache writes at that tier's rate instead of falling
    back to the standard cache-write rate."""
    usage = Usage(
        prompt_tokens=10_000,
        completion_tokens=500,
        total_tokens=10_500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=6_000,
            cache_write_tokens=3_000,
            text_tokens=1_000,
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="gpt-5.6-sol",
        usage=usage,
        custom_llm_provider="openai",
        service_tier=service_tier,
    )

    expected_prompt = 1_000 * input_rate + 6_000 * cache_read_rate + 3_000 * cache_write_rate
    assert prompt_cost == pytest.approx(expected_prompt, rel=1e-9)
    assert completion_cost == pytest.approx(500 * output_rate, rel=1e-9)


def test_fast_service_tier_bills_at_the_priority_rate(_local_model_cost_map):
    """Regression: OpenAI's Fast mode replaced Priority Processing and costs 2x standard.

    Before the fix "fast" fell through to standard pricing, so a Fast mode request
    was billed at half of what it actually costs."""
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000,
        completion_tokens=500,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=200),
    )

    standard = generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier=None
    )
    priority = generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="priority"
    )
    fast = generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="fast"
    )

    expected_prompt = 800 * 8e-06 + 200 * 8e-07
    expected_completion = 500 * 4e-05

    assert fast == priority
    assert fast[0] == pytest.approx(expected_prompt, rel=1e-9)
    assert fast[1] == pytest.approx(expected_completion, rel=1e-9)
    assert fast[0] == pytest.approx(standard[0] * 2, rel=1e-9)
    assert fast[1] == pytest.approx(standard[1] * 2, rel=1e-9)


def test_fast_service_tier_is_case_insensitive(_local_model_cost_map):
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1_000, completion_tokens=500)

    assert generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="FAST"
    ) == generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="fast"
    )


def test_fast_service_tier_matches_priority_above_the_context_threshold(_local_model_cost_map):
    """The above-threshold branch resolves its own cost keys, so the alias has to hold there too."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=300_000, completion_tokens=1_000)

    fast = generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="fast"
    )
    priority = generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="priority"
    )

    assert fast == priority
    assert fast[0] == pytest.approx(300_000 * 8e-06, rel=1e-9)
    assert fast[1] == pytest.approx(1_000 * 3e-05, rel=1e-9)


def test_priority_reasoning_tokens_bill_at_the_priority_output_rate(_local_model_cost_map):
    """Regression: gemini-3.5-flash publishes priority output pricing but no priority
    reasoning key, so reasoning tokens under priority/fast were billed at the standard
    output_cost_per_reasoning_token instead of following the tier's output rate."""
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000,
        completion_tokens=5_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=4_000),
    )

    model_info = litellm.get_model_info(model="gemini-3.5-flash", custom_llm_provider="gemini")
    standard_output_rate = model_info["output_cost_per_token"]
    standard_reasoning_rate = model_info["output_cost_per_reasoning_token"]
    priority_output_rate = model_info["output_cost_per_token_priority"]
    assert priority_output_rate is not None
    assert priority_output_rate != standard_reasoning_rate

    standard = generic_cost_per_token(
        model="gemini-3.5-flash", usage=usage, custom_llm_provider="gemini", service_tier=None
    )
    priority = generic_cost_per_token(
        model="gemini-3.5-flash", usage=usage, custom_llm_provider="gemini", service_tier="priority"
    )
    fast = generic_cost_per_token(
        model="gemini-3.5-flash", usage=usage, custom_llm_provider="gemini", service_tier="fast"
    )

    assert standard[1] == pytest.approx(1_000 * standard_output_rate + 4_000 * standard_reasoning_rate, rel=1e-9)
    assert priority[1] == pytest.approx(5_000 * priority_output_rate, rel=1e-9)
    assert fast == priority


def test_explicit_tier_reasoning_key_wins_over_the_tier_output_rate():
    from litellm.types.utils import Usage

    model_info = {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4e-06,
        "output_cost_per_reasoning_token": 6e-06,
        "input_cost_per_token_priority": 2e-06,
        "output_cost_per_token_priority": 8e-06,
        "output_cost_per_reasoning_token_priority": 1.2e-05,
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=1_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=600),
    )

    _, completion_cost = generic_cost_per_token(
        model="synthetic-model",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        model_info=model_info,
    )

    assert completion_cost == pytest.approx(400 * 8e-06 + 600 * 1.2e-05, rel=1e-9)


def test_null_tier_reasoning_key_falls_back_to_the_tier_output_rate():
    """get_model_info dumps every ModelInfo field, so an unpublished tier reasoning key
    arrives as an explicit None and must not shadow the tier output rate."""
    from litellm.types.utils import Usage

    model_info = {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4e-06,
        "output_cost_per_reasoning_token": 6e-06,
        "output_cost_per_reasoning_token_priority": None,
        "input_cost_per_token_priority": 2e-06,
        "output_cost_per_token_priority": 8e-06,
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=1_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=600),
    )

    _, completion_cost = generic_cost_per_token(
        model="synthetic-model",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        model_info=model_info,
    )

    assert completion_cost == pytest.approx(1_000 * 8e-06, rel=1e-9)


def test_tier_request_without_tier_pricing_keeps_the_standard_reasoning_rate():
    from litellm.types.utils import Usage

    model_info = {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4e-06,
        "output_cost_per_reasoning_token": 6e-06,
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=1_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=600),
    )

    _, completion_cost = generic_cost_per_token(
        model="synthetic-model",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        model_info=model_info,
    )

    assert completion_cost == pytest.approx(400 * 4e-06 + 600 * 6e-06, rel=1e-9)


GEMINI_37_FLASH_LAUNCH_PRICING = [
    ("gemini-3.7-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("gemini/gemini-3.7-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("vertex_ai/gemini-3.7-flash", 7.5e-07, 3.75e-06, 7.5e-08),
]


@pytest.mark.parametrize("model,input_cost,output_cost,cache_read_cost", GEMINI_37_FLASH_LAUNCH_PRICING)
def test_gemini_37_flash_launch_pricing(model, input_cost, output_cost, cache_read_cost, _local_model_cost_map):
    model_cost_map = litellm.model_cost[model]
    assert model_cost_map["input_cost_per_token"] == input_cost
    assert model_cost_map["output_cost_per_token"] == output_cost
    assert model_cost_map["output_cost_per_reasoning_token"] == output_cost
    assert model_cost_map["cache_read_input_token_cost"] == cache_read_cost
    assert model_cost_map["mode"] == "chat"
    assert model_cost_map["supports_reasoning"] is True
    assert model_cost_map["supports_function_calling"] is True
    assert model_cost_map["max_input_tokens"] == 1048576


def test_generic_cost_per_token_gemini_37_flash(_local_model_cost_map):
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=200,
            text_tokens=300,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=1000),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model="gemini-3.7-flash",
        usage=usage,
        custom_llm_provider="gemini",
    )
    assert prompt_cost == pytest.approx(0.00075)
    assert completion_cost == pytest.approx(0.001875)


def test_grok_46_launch_pricing(_local_model_cost_map):
    model_cost_map = litellm.model_cost["xai/grok-4.6"]
    assert model_cost_map["input_cost_per_token"] == 2e-06
    assert model_cost_map["output_cost_per_token"] == 6e-06
    assert model_cost_map["cache_read_input_token_cost"] == 5e-07
    assert model_cost_map["input_cost_per_token_above_200k_tokens"] == 4e-06
    assert model_cost_map["output_cost_per_token_above_200k_tokens"] == 1.2e-05
    assert model_cost_map["cache_read_input_token_cost_above_200k_tokens"] == 1e-06
    assert model_cost_map["mode"] == "chat"
    assert model_cost_map["supports_reasoning"] is True
    assert model_cost_map["supports_function_calling"] is True
    assert model_cost_map["max_input_tokens"] == 500000


def test_generic_cost_per_token_grok_46(_local_model_cost_map):
    usage = Usage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=1_000),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model="grok-4.6",
        usage=usage,
        custom_llm_provider="xai",
    )
    assert prompt_cost == pytest.approx(1_000 * 2e-06)
    assert completion_cost == pytest.approx(500 * 6e-06)


def test_generic_cost_per_token_grok_46_long_context(_local_model_cost_map):
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=50_000, text_tokens=200_000
        ),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model="grok-4.6",
        usage=usage,
        custom_llm_provider="xai",
    )
    assert prompt_cost == pytest.approx(200_000 * 4e-06 + 50_000 * 1e-06)
    assert completion_cost == pytest.approx(1_000 * 1.2e-05)
