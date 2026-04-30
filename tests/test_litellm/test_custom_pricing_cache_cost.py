import pytest

from litellm import completion_cost
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


def test_custom_cost_per_token_uses_cache_read_pricing():
    usage = Usage(
        prompt_tokens=6074,
        completion_tokens=285,
        total_tokens=6359,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=3456,
            audio_tokens=0,
        ),
    )
    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="openai/gpt-5.4",
        object="chat.completion",
        choices=[],
        usage=usage,
    )

    cost = completion_cost(
        completion_response=response,
        model="openai/gpt-5.4",
        custom_llm_provider="openai",
        custom_cost_per_token={
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.000015,
            "cache_read_input_token_cost": 0.00000025,
        },
    )

    assert cost == pytest.approx(0.011684)


def test_custom_cost_per_token_uses_cache_read_usage_field():
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=40,
    )
    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="openai/custom",
        object="chat.completion",
        choices=[],
        usage=usage,
    )

    cost = completion_cost(
        completion_response=response,
        model="openai/custom",
        custom_llm_provider="openai",
        custom_cost_per_token={
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "cache_read_input_token_cost": 0.001,
        },
    )

    assert cost == pytest.approx((60 * 0.01) + (40 * 0.001) + (10 * 0.02))


def test_custom_cost_per_token_uses_cache_creation_pricing():
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_creation_input_tokens=30,
    )
    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="openai/custom",
        object="chat.completion",
        choices=[],
        usage=usage,
    )

    cost = completion_cost(
        completion_response=response,
        model="openai/custom",
        custom_llm_provider="openai",
        custom_cost_per_token={
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "cache_creation_input_token_cost": 0.015,
        },
    )

    assert cost == pytest.approx((70 * 0.01) + (30 * 0.015) + (10 * 0.02))
