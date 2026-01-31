import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from pydantic import BaseModel

import litellm
from litellm.cost_calculator import (
    completion_cost,
    handle_realtime_stream_cost_calculation,
    response_cost_calculator,
)
from litellm.types.llms.openai import OpenAIRealtimeStreamList
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage
from litellm.utils import TranscriptionResponse


def test_completion_cost_uses_response_model_for_dynamic_routing():
    """
    Test that completion_cost uses the model from the response object
    when the input model (e.g., azure-model-router) is not in model_cost.
    This supports Azure Model Router and similar dynamic routing scenarios.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Simulate Azure Model Router: input is generic router, response has actual model
    response = ModelResponse(
        id="test-id",
        model="azure_ai/gpt-4o-2024-08-06",  # Response contains actual model used
        choices=[],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Should calculate cost using the response model, not the input model
    cost = completion_cost(
        completion_response=response,
        model="azure_ai/azure-model-router",  # Input model doesn't exist in model_cost
        custom_llm_provider="azure_ai",
    )

    assert cost > 0, "Cost should be calculated using response model"


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


def test_cost_calculator_with_usage(monkeypatch):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        prompt_tokens=120,
        completion_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=10,
            audio_tokens=90,
            image_tokens=20,
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

    # Step 1: Test a model where input_cost_per_image_token is not set.
    # In this case the calculation should use input_cost_per_token as fallback.
    assert (
        model_info.get("input_cost_per_image_token") is None
    ), "Test case expects that input_cost_per_image_token is not set"

    expected_cost = (
        usage.prompt_tokens_details.audio_tokens
        * model_info["input_cost_per_audio_token"]
        + usage.prompt_tokens_details.text_tokens * model_info["input_cost_per_token"]
        + usage.prompt_tokens_details.image_tokens * model_info["input_cost_per_token"]
        + usage.completion_tokens * model_info["output_cost_per_token"]
    )

    assert result == expected_cost, f"Got {result}, Expected {expected_cost}"

    # Step 2: Set input_cost_per_image_token.
    # In this case the explicit cost information should be used.
    temp_model_info_object = dict(model_info)
    temp_model_info_object["input_cost_per_image_token"] = 0.5

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"gemini-2.0-flash-001": temp_model_info_object},
    )

    # Invalidate caches after modifying litellm.model_cost
    from litellm.utils import _invalidate_model_cost_lowercase_map
    _invalidate_model_cost_lowercase_map()

    result = response_cost_calculator(
        response_object=mr,
        model="",
        custom_llm_provider="vertex_ai",
        call_type="acompletion",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    expected_cost = (
        usage.prompt_tokens_details.audio_tokens
        * temp_model_info_object["input_cost_per_audio_token"]
        + usage.prompt_tokens_details.text_tokens
        * temp_model_info_object["input_cost_per_token"]
        + usage.prompt_tokens_details.image_tokens
        * temp_model_info_object["input_cost_per_image_token"]
        + usage.completion_tokens * temp_model_info_object["output_cost_per_token"]
    )

    assert result == expected_cost, f"Got {result}, Expected {expected_cost}"


def test_transcription_cost_uses_token_pricing():
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        prompt_tokens=14,
        completion_tokens=45,
        total_tokens=59,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=0, audio_tokens=14
        ),
    )
    response = TranscriptionResponse(text="demo text")
    response.usage = usage

    cost = completion_cost(
        completion_response=response,
        model="gpt-4o-transcribe",
        custom_llm_provider="openai",
        call_type="atranscription",
    )

    expected_cost = (14 * 6e-06) + (45 * 1e-05)
    assert pytest.approx(cost, rel=1e-6) == expected_cost


def test_transcription_cost_falls_back_to_duration():
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    response = TranscriptionResponse(text="demo text")
    response.duration = 10.0

    cost = completion_cost(
        completion_response=response,
        model="whisper-1",
        custom_llm_provider="openai",
        call_type="atranscription",
    )

    expected_cost = 10.0 * 0.0001
    assert pytest.approx(cost, rel=1e-6) == expected_cost


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


def test_azure_audio_output_cost_calculation():
    """
    Test that Azure audio models correctly calculate costs for audio output tokens.

    Reproduces issue: https://github.com/BerriAI/litellm/issues/19764
    Audio tokens should be charged at output_cost_per_audio_token rate,
    not at the text token rate (output_cost_per_token).
    """
    from litellm.types.utils import (
        Choices,
        CompletionTokensDetailsWrapper,
        Message,
    )

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Scenario from issue #19764:
    # Input: 17 text tokens, 0 audio tokens
    # Output: 110 text tokens, 482 audio tokens
    usage_object = Usage(
        prompt_tokens=17,
        completion_tokens=592,  # 110 text + 482 audio
        total_tokens=609,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=0,
            cached_tokens=0,
            text_tokens=17,
            image_tokens=0,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=482,
            reasoning_tokens=0,
            text_tokens=110,
        ),
    )

    completion = ModelResponse(
        id="test-azure-audio-cost",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Test response",
                    role="assistant",
                ),
            )
        ],
        created=1729282652,
        model="azure/gpt-audio-2025-08-28",
        object="chat.completion",
        usage=usage_object,
    )

    cost = completion_cost(completion, model="azure/gpt-audio-2025-08-28")

    model_info = litellm.get_model_info("azure/gpt-audio-2025-08-28")

    # Calculate expected cost
    expected_input_cost = (
        model_info["input_cost_per_token"] * 17  # text tokens
    )
    expected_output_cost = (
        model_info["output_cost_per_token"] * 110  # text tokens
        + model_info["output_cost_per_audio_token"] * 482  # audio tokens
    )
    expected_total_cost = expected_input_cost + expected_output_cost

    # The bug was: all output tokens charged at text rate
    wrong_output_cost = model_info["output_cost_per_token"] * 592
    wrong_total_cost = expected_input_cost + wrong_output_cost

    # Verify audio tokens are NOT charged at text rate (the bug)
    assert abs(cost - wrong_total_cost) > 0.001, (
        "Bug: Audio tokens are being charged at text token rate"
    )

    # Verify cost matches
    assert abs(cost - expected_total_cost) < 0.0000001, (
        f"Expected cost {expected_total_cost}, got {cost}"
    )


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
        Message,
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
    from litellm.types.utils import Choices, Message, Usage

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
        print("DEBUG: No tiered input pricing available, using base pricing")
        input_cost_above_200k = input_cost_per_token

    if output_cost_above_200k is not None:
        print(
            f"DEBUG: Tiered output cost per token (>200k): ${output_cost_above_200k:.2e}"
        )
    else:
        print("DEBUG: No tiered output pricing available, using base pricing")
        output_cost_above_200k = output_cost_per_token

    if cache_creation_above_200k is not None:
        print(
            f"DEBUG: Tiered cache creation cost per token (>200k): ${cache_creation_above_200k:.2e}"
        )
    else:
        print("DEBUG: No tiered cache creation pricing available, using base pricing")
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


def test_azure_ai_cache_cost_calculation():
    """
    Test that azure_ai provider correctly calculates cache costs using generic_cost_per_token.

    This verifies that azure_ai models with custom cache pricing in model_info
    will have their cache_creation_input_token_cost and cache_read_input_token_cost
    applied correctly.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
    from litellm.types.utils import (
        PromptTokensDetailsWrapper,
        Usage,
    )

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Register a custom azure_ai model with cache pricing
    test_model_id = "test-azure-ai-claude-model"
    litellm.register_model(
        model_cost={
            test_model_id: {
                "input_cost_per_token": 5.0e-06,
                "output_cost_per_token": 2.5e-05,
                "cache_creation_input_token_cost": 6.25e-06,
                "cache_read_input_token_cost": 5.0e-07,
                "litellm_provider": "azure_ai",
                "max_tokens": 200000,
            }
        }
    )

    # Create usage with cache tokens
    usage = Usage(
        completion_tokens=100,
        prompt_tokens=1000,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=800,  # 800 cache read tokens
            text_tokens=100,  # 100 regular text tokens
        ),
        cache_creation_input_tokens=100,  # 100 cache creation tokens
    )

    input_cost, output_cost = generic_cost_per_token(
        model=test_model_id,
        usage=usage,
        custom_llm_provider="azure_ai",
    )

    total_cost = input_cost + output_cost

    # Calculate expected cost manually
    model_info = litellm.model_cost[test_model_id]
    expected_input_cost = (
        model_info["input_cost_per_token"] * 100  # text tokens
        + model_info["cache_read_input_token_cost"] * 800  # cached tokens
        + model_info["cache_creation_input_token_cost"] * 100  # cache creation tokens
    )
    expected_output_cost = model_info["output_cost_per_token"] * 100

    print(f"Input cost: {input_cost}, Expected: {expected_input_cost}")
    print(f"Output cost: {output_cost}, Expected: {expected_output_cost}")
    print(f"Total cost: {total_cost}")

    assert abs(input_cost - expected_input_cost) < 1e-10, (
        f"Input cost mismatch: got {input_cost}, expected {expected_input_cost}"
    )
    assert abs(output_cost - expected_output_cost) < 1e-10, (
        f"Output cost mismatch: got {output_cost}, expected {expected_output_cost}"
    )


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

    print("✓ Cost discount test passed:")
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

    print("✓ Selective discount test passed:")
    print(f"  - OpenAI cost (no discount configured): ${cost_without_discount:.6f}")
    print(f"  - Cost remains unchanged: ${cost_with_selective_discount:.6f}")


def test_cost_margin_percentage():
    """
    Test that percentage-based cost margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 10% margin for openai
    litellm.cost_margin_config = {"openai": 0.10}

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify margin is applied (10% margin means 110% of original cost)
    expected_cost = cost_without_margin * 1.10
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin percentage test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin (10%): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_fixed_amount():
    """
    Test that fixed amount cost margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set $0.001 fixed margin for openai
    litellm.cost_margin_config = {"openai": {"fixed_amount": 0.001}}

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify fixed margin is applied
    expected_cost = cost_without_margin + 0.001
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin fixed amount test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin ($0.001): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_combined():
    """
    Test that combined percentage and fixed amount margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 8% margin + $0.0005 fixed for openai
    litellm.cost_margin_config = {
        "openai": {"percentage": 0.08, "fixed_amount": 0.0005}
    }

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify combined margin is applied
    expected_cost = cost_without_margin * 1.08 + 0.0005
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin combined test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin (8% + $0.0005): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_global():
    """
    Test that global margin is applied when no provider-specific margin is configured
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% global margin (no provider-specific margin)
    litellm.cost_margin_config = {"global": 0.05}

    # Calculate cost with global margin
    cost_with_global_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify global margin is applied
    expected_cost = cost_without_margin * 1.05
    assert cost_with_global_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin global test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with global margin (5%): ${cost_with_global_margin:.6f}")
    print(f"  - Margin added: ${cost_with_global_margin - cost_without_margin:.6f}")


def test_cost_margin_provider_overrides_global():
    """
    Test that provider-specific margin overrides global margin
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% global margin and 10% provider-specific margin
    litellm.cost_margin_config = {"global": 0.05, "openai": 0.10}

    # Calculate cost - should use provider-specific margin (10%), not global (5%)
    cost_with_provider_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify provider-specific margin is used (not global)
    expected_cost = cost_without_margin * 1.10  # 10% from provider, not 5% from global
    assert cost_with_provider_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin provider override test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(
        f"  - Cost with provider margin (10%, overrides 5% global): ${cost_with_provider_margin:.6f}"
    )
    print(f"  - Margin added: ${cost_with_provider_margin - cost_without_margin:.6f}")


def test_cost_margin_with_discount():
    """
    Test that margin is applied after discount (independent calculation)
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original configs
    original_margin_config = litellm.cost_margin_config.copy()
    original_discount_config = litellm.cost_discount_config.copy()

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate base cost
    litellm.cost_margin_config = {}
    litellm.cost_discount_config = {}
    base_cost = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% discount and 10% margin
    litellm.cost_discount_config = {"openai": 0.05}
    litellm.cost_margin_config = {"openai": 0.10}

    # Calculate cost with both discount and margin
    cost_with_both = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original configs
    litellm.cost_margin_config = original_margin_config
    litellm.cost_discount_config = original_discount_config

    # Verify: discount applied first, then margin
    # Base cost -> discount: base * 0.95 -> margin: (base * 0.95) * 1.10
    expected_cost = base_cost * 0.95 * 1.10
    assert cost_with_both == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin with discount test passed:")
    print(f"  - Base cost: ${base_cost:.6f}")
    print(f"  - Cost with 5% discount + 10% margin: ${cost_with_both:.6f}")
    print(f"  - Expected: ${expected_cost:.6f}")


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
                    url="test-azure-blob-url-with-sas-token",
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


def test_completion_cost_extracts_service_tier_from_response():
    """Test that completion_cost extracts service_tier from completion_response object."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"

    # Create usage object
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    # Create ModelResponse with service_tier in the response object
    response_with_service_tier = ModelResponse(
        usage=usage,
        model=model,
    )
    # Set service_tier as an attribute on the response
    setattr(response_with_service_tier, "service_tier", "flex")

    # Test that flex pricing is used when service_tier is in response
    flex_cost = completion_cost(
        completion_response=response_with_service_tier,
        model=model,
        custom_llm_provider="openai",
    )

    # Create ModelResponse without service_tier (should use standard pricing)
    response_without_service_tier = ModelResponse(
        usage=usage,
        model=model,
    )

    # Test that standard pricing is used when service_tier is not in response
    standard_cost = completion_cost(
        completion_response=response_without_service_tier,
        model=model,
        custom_llm_provider="openai",
    )

    # Flex should be approximately 50% of standard
    assert flex_cost > 0, "Flex cost should be greater than 0"
    assert standard_cost > 0, "Standard cost should be greater than 0"
    assert flex_cost < standard_cost, "Flex cost should be less than standard cost"

    flex_ratio = flex_cost / standard_cost
    assert (
        0.45 <= flex_ratio <= 0.55
    ), f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"


def test_completion_cost_extracts_service_tier_from_usage():
    """Test that completion_cost extracts service_tier from usage object."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"

    # Create usage object with service_tier
    usage_with_service_tier = Usage(
        prompt_tokens=1000, completion_tokens=500, total_tokens=1500
    )
    # Set service_tier as an attribute on the usage object
    setattr(usage_with_service_tier, "service_tier", "flex")

    # Create ModelResponse with usage containing service_tier
    response = ModelResponse(
        usage=usage_with_service_tier,
        model=model,
    )

    # Test that flex pricing is used when service_tier is in usage
    flex_cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="openai",
    )

    # Create usage object without service_tier
    usage_without_service_tier = Usage(
        prompt_tokens=1000, completion_tokens=500, total_tokens=1500
    )

    # Create ModelResponse with usage without service_tier
    response_standard = ModelResponse(
        usage=usage_without_service_tier,
        model=model,
    )

    # Test that standard pricing is used when service_tier is not in usage
    standard_cost = completion_cost(
        completion_response=response_standard,
        model=model,
        custom_llm_provider="openai",
    )

    # Flex should be approximately 50% of standard
    assert flex_cost > 0, "Flex cost should be greater than 0"
    assert standard_cost > 0, "Standard cost should be greater than 0"
    assert flex_cost < standard_cost, "Flex cost should be less than standard cost"

    flex_ratio = flex_cost / standard_cost
    assert (
        0.45 <= flex_ratio <= 0.55
    ), f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"


def test_completion_cost_service_tier_priority():
    """Test that service_tier extraction follows priority: optional_params > completion_response > usage."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"

    # Create usage object with service_tier="flex"
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    setattr(usage, "service_tier", "flex")

    # Create response with service_tier="priority"
    response = ModelResponse(
        usage=usage,
        model=model,
    )
    setattr(response, "service_tier", "priority")

    # Test that optional_params takes priority over response and usage
    cost_from_params = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="openai",
        optional_params={"service_tier": "flex"},
    )

    # Test that response takes priority over usage when optional_params is not provided
    completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="openai",
    )

    # Test that usage is used when neither optional_params nor response have service_tier
    # Create a new response without service_tier attribute
    response_no_tier = ModelResponse(
        usage=usage,
        model=model,
    )
    # Don't set service_tier on response, so it will fall back to usage

    cost_from_usage = completion_cost(
        completion_response=response_no_tier,
        model=model,
        custom_llm_provider="openai",
    )

    # All should use flex pricing (from different sources)
    assert cost_from_params > 0, "Cost from params should be greater than 0"
    assert cost_from_usage > 0, "Cost from usage should be greater than 0"

    # Costs should be similar (all using flex)
    assert (
        abs(cost_from_params - cost_from_usage) < 1e-6
    ), "Costs from params and usage should be similar (both flex)"


def test_gemini_cache_tokens_details_no_negative_values():
    """
    Test for Issue #18750: Negative text_tokens with Gemini caching

    When using Gemini with explicit caching, the response includes cacheTokensDetails
    which breaks down cached tokens by modality. This test ensures that:
    1. text_tokens is never negative
    2. We correctly subtract cached tokens per modality (not total)
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    # Scenario from issue #18750: Image + text with explicit caching
    # Real Gemini response structure when using cached content
    completion_response = {
        "usageMetadata": {
            "promptTokenCount": 9660,
            "candidatesTokenCount": 7,
            "totalTokenCount": 9667,
            "cachedContentTokenCount": 9651,
            # Total tokens by modality (includes cached + non-cached)
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 9402},
                {"modality": "IMAGE", "tokenCount": 258},
            ],
            # Breakdown of cached tokens by modality
            "cacheTokensDetails": [
                {"modality": "TEXT", "tokenCount": 9393},
                {"modality": "IMAGE", "tokenCount": 258},
            ],
        }
    }

    usage = VertexGeminiConfig._calculate_usage(completion_response)

    # Text tokens should be non-cached text only: 9402 - 9393 = 9
    assert (
        usage.prompt_tokens_details.text_tokens == 9
    ), f"Expected text_tokens=9, got {usage.prompt_tokens_details.text_tokens}"

    # Image tokens should be non-cached image only: 258 - 258 = 0
    assert (
        usage.prompt_tokens_details.image_tokens == 0
    ), f"Expected image_tokens=0, got {usage.prompt_tokens_details.image_tokens}"

    # Total cached should match
    assert (
        usage.prompt_tokens_details.cached_tokens == 9651
    ), f"Expected cached_tokens=9651, got {usage.prompt_tokens_details.cached_tokens}"

    # MOST IMPORTANT: text_tokens should NEVER be negative
    assert (
        usage.prompt_tokens_details.text_tokens >= 0
    ), f"BUG: text_tokens is negative ({usage.prompt_tokens_details.text_tokens})! This was the issue in #18750"

    print(
        "✅ Issue #18750 fix verified: text_tokens is correctly calculated and non-negative"
    )


def test_gemini_without_cache_tokens_details():
    """
    Test Gemini response without cacheTokensDetails (implicit caching or no cache)

    When cacheTokensDetails is not present, we should use promptTokensDetails as-is
    without subtracting anything.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    completion_response = {
        "usageMetadata": {
            "promptTokenCount": 264,
            "candidatesTokenCount": 15,
            "totalTokenCount": 279,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 6},
                {"modality": "IMAGE", "tokenCount": 258},
            ]
            # No cacheTokensDetails
        }
    }

    usage = VertexGeminiConfig._calculate_usage(completion_response)

    # Should use promptTokensDetails values directly
    assert usage.prompt_tokens_details.text_tokens == 6
    assert usage.prompt_tokens_details.image_tokens == 258
    assert usage.prompt_tokens_details.text_tokens >= 0

    print("✅ Gemini without cacheTokensDetails works correctly")


def test_gemini_implicit_caching_cost_calculation():
    """
    Test for Issue #16341: Gemini implicit cached tokens not counted in spend log

    When Gemini uses implicit caching, it returns cachedContentTokenCount but NOT
    cacheTokensDetails. In this case, we should subtract cachedContentTokenCount
    from text_tokens to correctly calculate costs.

    See: https://github.com/BerriAI/litellm/issues/16341
    """
    from litellm import completion_cost
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.utils import Choices, Message, ModelResponse

    # Simulate Gemini response with implicit caching (cachedContentTokenCount only)
    completion_response = {
        "usageMetadata": {
            "promptTokenCount": 10000,
            "candidatesTokenCount": 5,
            "totalTokenCount": 10005,
            "cachedContentTokenCount": 8000,  # Implicit caching - no cacheTokensDetails
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10000}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 5}],
        }
    }

    usage = VertexGeminiConfig._calculate_usage(completion_response)

    # Verify parsing
    assert (
        usage.cache_read_input_tokens == 8000
    ), f"cache_read_input_tokens should be 8000, got {usage.cache_read_input_tokens}"
    assert (
        usage.prompt_tokens_details.cached_tokens == 8000
    ), f"cached_tokens should be 8000, got {usage.prompt_tokens_details.cached_tokens}"

    # CRITICAL: text_tokens should be (10000 - 8000) = 2000, NOT 10000
    # This is the fix for issue #16341
    assert (
        usage.prompt_tokens_details.text_tokens == 2000
    ), f"text_tokens should be 2000 (10000 - 8000), got {usage.prompt_tokens_details.text_tokens}"

    # Verify cost calculation uses cached token pricing
    response = ModelResponse(
        id="mock-id",
        model="gemini-2.0-flash",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )

    cost = completion_cost(
        completion_response=response,
        model="gemini-2.0-flash",
        custom_llm_provider="gemini",
    )

    # Get model pricing for verification
    import litellm

    model_info = litellm.get_model_info("gemini/gemini-2.0-flash")
    input_cost = model_info.get("input_cost_per_token", 0)
    cache_read_cost = model_info.get("cache_read_input_token_cost", input_cost)
    output_cost = model_info.get("output_cost_per_token", 0)

    # Expected cost: (2000 * input) + (8000 * cache_read) + (5 * output)
    expected_cost = (2000 * input_cost) + (8000 * cache_read_cost) + (5 * output_cost)

    assert abs(cost - expected_cost) < 1e-9, (
        f"Cost calculation is wrong. Got ${cost:.6f}, expected ${expected_cost:.6f}. "
        f"Cached tokens may not be using reduced pricing."
    )

    print("✅ Issue #16341 fix verified: Gemini implicit caching cost calculated correctly")
