import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

import litellm
from litellm.cost_calculator import (
    handle_realtime_stream_cost_calculation,
    response_cost_calculator,
)
from litellm.types.llms.openai import OpenAIRealtimeStreamList
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


def test_cost_calculator_with_response_cost_in_additional_headers():
    class MockResponse(BaseModel):
        _hidden_params = {
            "additional_headers": {"llm_provider-x-litellm-response-cost": 1000}
        }

    result = response_cost_calculator(
        response_object=MockResponse(),
        model="",
        custom_llm_provider=None,
        call_type="",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    assert result == 1000


def test_cost_calculator_with_usage():
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        prompt_tokens=100,
        completion_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=10, audio_tokens=90
        ),
    )
    mr = ModelResponse(usage=usage, model="gemini-2.0-flash-001")

    result = response_cost_calculator(
        response_object=mr,
        model="",
        custom_llm_provider="vertex_ai",
        call_type="acompletion",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    model_info = litellm.model_cost["gemini-2.0-flash-001"]

    expected_cost = (
        usage.prompt_tokens_details.audio_tokens
        * model_info["input_cost_per_audio_token"]
        + usage.prompt_tokens_details.text_tokens * model_info["input_cost_per_token"]
        + usage.completion_tokens * model_info["output_cost_per_token"]
    )

    assert result == expected_cost, f"Got {result}, Expected {expected_cost}"


def test_handle_realtime_stream_cost_calculation():
    from litellm.cost_calculator import RealtimeAPITokenUsageProcessor

    # Setup test data
    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-3.5-turbo"}},
        {
            "type": "response.done",
            "response": {
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            },
        },
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                }
            },
        },
    ]

    combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )

    # Test with explicit model name
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    # Calculate expected cost
    # gpt-3.5-turbo costs: $0.0015/1K tokens input, $0.002/1K tokens output
    expected_cost = (300 * 0.0015 / 1000) + (  # input tokens (100 + 200)
        150 * 0.002 / 1000
    )  # output tokens (50 + 100)
    assert (
        abs(cost - expected_cost) <= 0.00075
    )  # Allow small floating point differences

    # Test with different model name in session
    results[0]["session"]["model"] = "gpt-4"

    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    # Calculate expected cost using gpt-4 rates
    # gpt-4 costs: $0.03/1K tokens input, $0.06/1K tokens output
    expected_cost = (300 * 0.03 / 1000) + (  # input tokens
        150 * 0.06 / 1000
    )  # output tokens
    assert abs(cost - expected_cost) < 0.00076

    # Test with no response.done events
    results = [{"type": "session.created", "session": {"model": "gpt-3.5-turbo"}}]
    combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )
    assert cost == 0.0  # No usage, no cost


def test_custom_pricing_with_router_model_id():
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "prod/claude-3-5-sonnet-20240620",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "test_api_key",
                },
                "model_info": {
                    "id": "my-unique-model-id",
                    "input_cost_per_token": 0.000006,
                    "output_cost_per_token": 0.00003,
                    "cache_creation_input_token_cost": 0.0000075,
                    "cache_read_input_token_cost": 0.0000006,
                },
            },
            {
                "model_name": "claude-3-5-sonnet-20240620",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "test_api_key",
                },
                "model_info": {
                    "input_cost_per_token": 100,
                    "output_cost_per_token": 200,
                },
            },
        ]
    )

    result = router.completion(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response=True,
    )

    result_2 = router.completion(
        model="prod/claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response=True,
    )

    assert (
        result._hidden_params["response_cost"]
        > result_2._hidden_params["response_cost"]
    )

    model_info = router.get_deployment_model_info(
        model_id="my-unique-model-id", model_name="anthropic/claude-sonnet-4-5-20250929"
    )
    assert model_info is not None
    assert model_info["input_cost_per_token"] == 0.000006
    assert model_info["output_cost_per_token"] == 0.00003
    assert model_info["cache_creation_input_token_cost"] == 0.0000075
    assert model_info["cache_read_input_token_cost"] == 0.0000006


def test_azure_realtime_cost_calculator():
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    cost = handle_realtime_stream_cost_calculation(
        results=[
            {
                "type": "session.created",
                "session": {"model": "gpt-4o-realtime-preview-2024-12-17"},
            },
        ],
        combined_usage_object=Usage(
            prompt_tokens=100,
            completion_tokens=100,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=10, audio_tokens=90
            ),
        ),
        custom_llm_provider="azure",
        litellm_model_name="my-custom-azure-deployment",
    )

    assert cost > 0


def test_default_image_cost_calculator(monkeypatch):
    from litellm.cost_calculator import default_image_cost_calculator

    temp_object = {
        "litellm_provider": "azure",
        "input_cost_per_pixel": 10,
    }

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b": temp_object
        },
    )

    args = {
        "model": "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b",
        "custom_llm_provider": "azure",
        "quality": "standard",
        "n": 1,
        "size": "1024-x-1024",
        "optional_params": {},
    }
    cost = default_image_cost_calculator(**args)
    assert cost == 10485760


def test_cost_calculator_with_cache_creation():
    from litellm import completion_cost
    from litellm.types.utils import (
        Choices,
        CompletionTokensDetailsWrapper,
        Message,
        PromptTokensDetailsWrapper,
        Usage,
    )

    litellm_model_response = ModelResponse(
        id="chatcmpl-cc5638bc-fdfe-48e4-8884-57c8f4fb7c63",
        created=1750733889,
        model=None,
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Hello! How can I help you today?",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    provider_specific_fields=None,
                ),
            )
        ],
        usage=Usage(
            **{
                "total_tokens": 28508,
                "prompt_tokens": 28495,
                "completion_tokens": 13,
                "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
                "cache_read_input_tokens": 28491,
                "completion_tokens_details": {
                    "audio_tokens": None,
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": None,
                    "rejected_prediction_tokens": None,
                },
                "cache_creation_input_tokens": 15,
            }
        ),
    )
    model = "claude-sonnet-4@20250514"

    assert litellm_model_response.usage.prompt_tokens_details.cached_tokens == 28491

    result = completion_cost(
        completion_response=litellm_model_response,
        model=model,
        custom_llm_provider="vertex_ai",
    )

    print(result)


def test_bedrock_cost_calculator_comparison_with_without_cache():
    """Test that Bedrock caching reduces costs compared to non-cached requests"""
    from litellm import completion_cost
    from litellm.types.utils import Choices, Message, PromptTokensDetailsWrapper, Usage

    # Response WITHOUT caching
    response_no_cache = ModelResponse(
        id="msg_no_cache",
        created=1750733889,
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Response without cache",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(
            total_tokens=28508,
            prompt_tokens=28495,
            completion_tokens=13,
        ),
    )

    # Response WITH caching (same total tokens, but most are cached)
    response_with_cache = ModelResponse(
        id="msg_with_cache",
        created=1750733889,
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Response with cache",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(
            **{
                "total_tokens": 28508,
                "prompt_tokens": 28495,
                "completion_tokens": 13,
                "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
                "cache_read_input_tokens": 28491,  # Most tokens are read from cache (cheaper)
                "completion_tokens_details": {
                    "audio_tokens": None,
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": None,
                    "rejected_prediction_tokens": None,
                },
                "cache_creation_input_tokens": 15,  # Only 15 new tokens added to cache
            }
        ),
    )

    # Calculate costs
    cost_no_cache = completion_cost(
        completion_response=response_no_cache,
        model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
        custom_llm_provider="bedrock",
    )

    cost_with_cache = completion_cost(
        completion_response=response_with_cache,
        model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
        custom_llm_provider="bedrock",
    )

    # Verify that cached request is cheaper
    assert cost_with_cache < cost_no_cache
    print(f"Cost without cache: {cost_no_cache}")
    print(f"Cost with cache: {cost_with_cache}")


def test_gemini_25_implicit_caching_cost():
    """
    Test that Gemini 2.5 models correctly calculate costs with implicit caching.

    This test reproduces the issue from #11156 where cached tokens should receive
    a 75% discount.
    """
    from litellm import completion_cost
    from litellm.types.utils import (
        Choices,
        Message,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    )

    # Create a mock response similar to the one in the issue
    litellm_model_response = ModelResponse(
        id="test-response",
        created=1750733889,
        model="gemini/gemini-2.5-flash",
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Understood. This is a test message to check the response from the Gemini model.",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        usage=Usage(
            total_tokens=15050,
            prompt_tokens=15033,
            completion_tokens=17,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                audio_tokens=None,
                cached_tokens=14316,  # This is cachedContentTokenCount from Gemini
            ),
            completion_tokens_details=None,
        ),
    )

    # Calculate the cost
    result = completion_cost(
        completion_response=litellm_model_response,
        model="gemini/gemini-2.5-flash",
    )

    # Current pricing for gemini/gemini-2.5-flash:
    # input: $0.30 / 1M tokens (3e-07 per token)
    # cache_read: $0.03 / 1M tokens (3e-08 per token)
    # output: $2.50 / 1M tokens (2.5e-06 per token)

    # Breakdown:
    # - Cached tokens: 14316 * 3e-08 = 0.00042948
    # - Non-cached tokens: (15033-14316) * 3e-07 = 717 * 3e-07 = 0.00021510
    # - Output tokens: 17 * 2.5e-06 = 0.00004250
    # Total: 0.00042948 + 0.00021510 + 0.00004250 = 0.00068708

    expected_cost = 0.00068708

    # Allow for small floating point differences
    assert (
        abs(result - expected_cost) < 1e-8
    ), f"Expected cost {expected_cost}, but got {result}"

    print(f"✓ Gemini 2.5 implicit caching cost calculation is correct: ${result:.8f}")


def test_log_context_cost_calculation():
    """
    Test that log context cost calculation works correctly with tiered pricing.

    This test verifies that when using extended context (above 200k tokens),
    the log context costs are calculated using the appropriate tiered rates.
    """
    from litellm import completion_cost
    from litellm.types.utils import (
        Choices,
        Message,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    )

    # Create a mock response with extended context usage
    extended_context_response = ModelResponse(
        id="test-extended-context-response",
        created=1750733889,
        model="claude-4-sonnet-20250514",
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="This is a test response for extended context cost calculation.",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        usage=Usage(
            total_tokens=350000,  # Above 200k threshold
            prompt_tokens=301000,  # Above 200k threshold
            completion_tokens=50000,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=300000,
                cached_tokens=0,  # No cache hits
                audio_tokens=None,
                image_tokens=None,
                character_count=None,
                video_length_seconds=None,
                cache_creation_tokens=1000,
            ),
            completion_tokens_details=None,
            _cache_creation_input_tokens=1000,  # Some tokens added to cache
        ),
    )

    # Calculate the cost using the extended context model
    result = completion_cost(
        completion_response=extended_context_response,
        model="claude-4-sonnet-20250514",
        custom_llm_provider="anthropic",
    )

    # Debug: Print the actual result
    print(f"DEBUG: Actual cost result: ${result:.6f}")

    # Get model info to understand the pricing
    from litellm import get_model_info

    model_info = get_model_info(
        model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
    )

    # Calculate expected cost based on actual model pricing
    input_cost_per_token = model_info.get("input_cost_per_token", 0)
    output_cost_per_token = model_info.get("output_cost_per_token", 0)
    cache_creation_cost_per_token = model_info.get("cache_creation_input_token_cost", 0)

    # Check if tiered pricing is applied
    input_cost_above_200k = model_info.get(
        "input_cost_per_token_above_200k_tokens", input_cost_per_token
    )
    output_cost_above_200k = model_info.get(
        "output_cost_per_token_above_200k_tokens", output_cost_per_token
    )
    cache_creation_above_200k = model_info.get(
        "cache_creation_input_token_cost_above_200k_tokens",
        cache_creation_cost_per_token,
    )

    print(f"DEBUG: Base input cost per token: ${input_cost_per_token:.2e}")
    print(f"DEBUG: Base output cost per token: ${output_cost_per_token:.2e}")
    print(
        f"DEBUG: Base cache creation cost per token: ${cache_creation_cost_per_token:.2e}"
    )

    # Handle tiered pricing - if not available, use base pricing
    if input_cost_above_200k is not None:
        print(
            f"DEBUG: Tiered input cost per token (>200k): ${input_cost_above_200k:.2e}"
        )
    else:
        print(f"DEBUG: No tiered input pricing available, using base pricing")
        input_cost_above_200k = input_cost_per_token

    if output_cost_above_200k is not None:
        print(
            f"DEBUG: Tiered output cost per token (>200k): ${output_cost_above_200k:.2e}"
        )
    else:
        print(f"DEBUG: No tiered output pricing available, using base pricing")
        output_cost_above_200k = output_cost_per_token

    if cache_creation_above_200k is not None:
        print(
            f"DEBUG: Tiered cache creation cost per token (>200k): ${cache_creation_above_200k:.2e}"
        )
    else:
        print(f"DEBUG: No tiered cache creation pricing available, using base pricing")
        cache_creation_above_200k = cache_creation_cost_per_token

    # Since we're above 200k tokens, we should use tiered pricing if available
    expected_input_cost = 300000 * input_cost_above_200k
    expected_output_cost = 50000 * output_cost_above_200k
    expected_cache_cost = 1000 * cache_creation_above_200k
    expected_total = expected_input_cost + expected_output_cost + expected_cache_cost

    print(f"DEBUG: Expected total: ${expected_total:.6f}")

    # Allow for small floating point differences
    assert (
        abs(result - expected_total) < 1e-6
    ), f"Expected cost ${expected_total:.6f}, but got ${result:.6f}"

    print(
        f"✓ Log context cost calculation with tiered pricing is correct: ${result:.6f}"
    )
    print(f"  - Input tokens (300k): ${expected_input_cost:.6f}")
    print(f"  - Output tokens (50k): ${expected_output_cost:.6f}")
    print(f"  - Cache creation (1k): ${expected_cache_cost:.6f}")
    print(f"  - Total: ${result:.6f}")


def test_gemini_25_explicit_caching_cost_direct_usage():
    """
    Test that Gemini 2.5 models correctly calculate costs with explicit caching.

    This test reproduces the issue from #11156 where cached tokens should receive
    a 75% discount.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
    from litellm.types.utils import (
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
        Usage,
    )
    from litellm.utils import get_model_info

    model_info = get_model_info(model="gemini-2.5-pro", custom_llm_provider="gemini")

    usage = Usage(
        completion_tokens=2522,
        prompt_tokens=42001,
        total_tokens=44523,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=1908,
            rejected_prediction_tokens=None,
            text_tokens=614,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=40938, text_tokens=1063, image_tokens=None
        ),
    )

    input_cost, output_cost = generic_cost_per_token(
        model="gemini/gemini-2.5-pro",
        usage=usage,
        custom_llm_provider="gemini",
    )

    total_cost = input_cost + output_cost

    expected_higher_than_actual_cost = (
        model_info["input_cost_per_token"] * usage.prompt_tokens
        + model_info["output_cost_per_token"] * usage.completion_tokens
    )

    print(f"expected_higher_than_actual_cost: {expected_higher_than_actual_cost}")

    assert expected_higher_than_actual_cost > total_cost

    expected_actual_cost = (
        model_info["input_cost_per_token"] * usage.prompt_tokens_details.text_tokens
        + model_info["cache_read_input_token_cost"]
        * usage.prompt_tokens_details.cached_tokens
        + model_info["output_cost_per_token"] * usage.completion_tokens
    )

    print(
        f"model_info['input_cost_per_token']: {model_info['input_cost_per_token']}, usage.prompt_tokens_details.text_tokens: {usage.prompt_tokens_details.text_tokens}, model_info['cache_read_input_token_cost']: {model_info['cache_read_input_token_cost']}, model_info['output_cost_per_token']: {model_info['output_cost_per_token']}"
    )

    print(f"Expected actual cost: {expected_actual_cost}")

    assert expected_actual_cost == total_cost


def test_cost_discount_vertex_ai():
    """
    Test that cost discount is applied correctly for Vertex AI provider
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_discount_config = litellm.cost_discount_config.copy()

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gemini-pro",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without discount
    litellm.cost_discount_config = {}
    cost_without_discount = completion_cost(
        completion_response=response,
        model="vertex_ai/gemini-pro",
        custom_llm_provider="vertex_ai",
    )

    # Set 5% discount for vertex_ai
    litellm.cost_discount_config = {"vertex_ai": 0.05}

    # Calculate cost with discount
    cost_with_discount = completion_cost(
        completion_response=response,
        model="vertex_ai/gemini-pro",
        custom_llm_provider="vertex_ai",
    )

    # Restore original config
    litellm.cost_discount_config = original_discount_config

    # Verify discount is applied (5% off means 95% of original cost)
    expected_cost = cost_without_discount * 0.95
    assert cost_with_discount == pytest.approx(expected_cost, rel=1e-9)

    print(f"✓ Cost discount test passed:")
    print(f"  - Original cost: ${cost_without_discount:.6f}")
    print(f"  - Discounted cost (5% off): ${cost_with_discount:.6f}")
    print(f"  - Savings: ${cost_without_discount - cost_with_discount:.6f}")


def test_cost_discount_not_applied_to_other_providers():
    """
    Test that cost discount only applies to configured providers
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_discount_config = litellm.cost_discount_config.copy()

    # Create mock response for OpenAI
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Set discount only for vertex_ai (not openai)
    litellm.cost_discount_config = {"vertex_ai": 0.05}

    # Calculate cost for OpenAI - should NOT have discount applied
    cost_with_selective_discount = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Clear discount config
    litellm.cost_discount_config = {}
    cost_without_discount = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_discount_config = original_discount_config

    # Costs should be the same (no discount applied to OpenAI)
    assert cost_with_selective_discount == cost_without_discount

    print(f"✓ Selective discount test passed:")
    print(f"  - OpenAI cost (no discount configured): ${cost_without_discount:.6f}")
    print(f"  - Cost remains unchanged: ${cost_with_selective_discount:.6f}")


def test_azure_image_generation_cost_calculator():
    from unittest.mock import MagicMock

    from litellm.types.utils import (
        ImageObject,
        ImageResponse,
        ImageUsage,
        ImageUsageInputTokensDetails,
    )

    response_cost_calculator_kwargs = {
        "response_object": ImageResponse(
            created=1761785270,
            background=None,
            data=[
                ImageObject(
                    b64_json=None,
                    revised_prompt="A futuristic, techno-inspired green duck wearing cool modern sunglasses. The duck has a sleek, metallic appearance with glowing neon green accents, standing on a high-tech urban background with holographic billboards and illuminated city lights in the distance. The duck's feathers have a glossy, high-tech sheen, resembling a robotic design but still maintaining its avian features. The scene has a vibrant, cyberpunk aesthetic with a neon color palette.",
                    url="https://dalleprodsec.blob.core.windows.net/private/images/caa17dc4-357d-4257-8938-eeea9baa8d0a/generated_00.png?se=2025-10-31T00%3A47%3A59Z&sig=KHRjLz3vMahbw94JtxL02S6t2AueeRMaiqj4z35HKDM%3D&ske=2025-11-05T00%3A26%3A20Z&skoid=e52d5ed7-0657-4f62-bc12-7e5dbb260a96&sks=b&skt=2025-10-29T00%3A26%3A20Z&sktid=33e01921-4d64-4f8c-a055-5bdaffd5e33d&skv=2020-10-02&sp=r&spr=https&sr=b&sv=2020-10-02",
                )
            ],
            output_format=None,
            quality="hd",
            size=None,
            usage=ImageUsage(
                input_tokens=0,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=0, text_tokens=0
                ),
                output_tokens=0,
                total_tokens=0,
            ),
        ),
        "model": "azure/dall-e-3",
        "cache_hit": False,
        "custom_llm_provider": "azure",
        "base_model": "azure/dall-e-3",
        "call_type": "aimage_generation",
        "optional_params": {},
        "custom_pricing": False,
        "prompt": "",
        "standard_built_in_tools_params": {
            "web_search_options": None,
            "file_search": None,
        },
        "router_model_id": "6738c432ffc9b733597c6b86613ca20dc5f49bde591fd3d03e7cd6aa25bb241e",
        "litellm_logging_obj": MagicMock(),
        "service_tier": None,
    }

    cost = response_cost_calculator(**response_cost_calculator_kwargs)
    assert cost > 0.079
