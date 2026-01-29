import os
import sys
from typing import Optional
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponsesAPIResponse,
)
from litellm.types.utils import Choices, Message, ModelResponse, Usage

"""
Test that all providers can transform completion responses to Responses API format
without breaking due to required fields in InputTokensDetails and OutputTokensDetails.

This is a regression test for the change where reasoning_tokens and cached_tokens
were made non-optional (must be int, not Optional[int]).
"""
class _CompletedEvent:
    def __init__(self, response):
        self.response = response


class _FakeResponsesStream:
    def __init__(self, response):
        self._emitted = False
        self._response = response
        self.completed_response = None
        self._hidden_params = {"headers": {"x-test": "1"}}

    def __iter__(self):
        return self

    def __next__(self):
        if not self._emitted:
            self._emitted = True
            self.completed_response = _CompletedEvent(self._response)
            return {"type": "response.completed"}
        raise StopIteration


def test_should_collect_response_from_stream():
    handler = ResponsesToCompletionBridgeHandler()
    response = ResponsesAPIResponse.model_construct(
        id="resp-1",
        created_at=0,
        output=[],
        object="response",
        model="gpt-5.2",
    )
    stream = _FakeResponsesStream(response)

    collected = handler._collect_response_from_stream(stream)

    assert collected.id == "resp-1"
    assert collected._hidden_params.get("headers") == {"x-test": "1"}


def create_mock_completion_response(
    model: str = "gpt-4",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
    reasoning_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    text_tokens: Optional[int] = None,
) -> ModelResponse:
    """
    Create a mock ModelResponse (chat completion) with various token details.
    
    This simulates responses from different providers that may or may not include
    reasoning_tokens, cached_tokens, etc.
    """
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    
    # Add prompt_tokens_details if we have cached_tokens or text_tokens
    if cached_tokens is not None or text_tokens is not None:
        from litellm.types.utils import PromptTokensDetails
        usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=cached_tokens,
            text_tokens=text_tokens,
        )
    
    # Add completion_tokens_details if we have reasoning_tokens or text_tokens
    if reasoning_tokens is not None or text_tokens is not None:
        from litellm.types.utils import CompletionTokensDetails
        usage.completion_tokens_details = CompletionTokensDetails(
            reasoning_tokens=reasoning_tokens,
            text_tokens=text_tokens,
        )
    
    return ModelResponse(
        id="chatcmpl-test",
        created=1234567890,
        model=model,
        object="chat.completion",
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
        usage=usage,
    )


def test_transform_usage_no_token_details():
    """
    Test that transformation works when completion response has NO token details.
    
    This simulates providers that don't return detailed token breakdowns.
    """
    completion_response = create_mock_completion_response(
        model="gpt-4",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    
    # Transform to Responses API usage format
    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )
    
    # Should succeed without errors
    assert responses_usage.input_tokens == 10
    assert responses_usage.output_tokens == 20
    assert responses_usage.total_tokens == 30
    
    # Token details should not be present when not provided
    assert responses_usage.input_tokens_details is None
    assert responses_usage.output_tokens_details is None
    
    print("✓ Transformation works with no token details")


def test_transform_usage_with_cached_tokens_only():
    """
    Test transformation when only cached_tokens is provided (no reasoning_tokens).
    
    This simulates providers like Anthropic that support prompt caching but not reasoning.
    """
    completion_response = create_mock_completion_response(
        model="claude-3-opus",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=80,  # Has cached tokens
        reasoning_tokens=None,  # No reasoning tokens
    )
    
    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )
    
    # Should succeed and default reasoning_tokens to 0
    assert responses_usage.input_tokens == 100
    assert responses_usage.output_tokens == 50
    assert responses_usage.total_tokens == 150
    
    # Input details should be present with cached_tokens
    assert responses_usage.input_tokens_details is not None
    assert isinstance(responses_usage.input_tokens_details, InputTokensDetails)
    assert responses_usage.input_tokens_details.cached_tokens == 80
    
    # Output details should not be present (no reasoning_tokens provided)
    assert responses_usage.output_tokens_details is None
    
    print("✓ Transformation works with cached_tokens only")


def test_transform_usage_with_reasoning_tokens_only():
    """
    Test transformation when only reasoning_tokens is provided (no cached_tokens).
    
    This simulates providers like OpenAI o1 that support reasoning but not caching.
    """
    completion_response = create_mock_completion_response(
        model="o1-preview",
        prompt_tokens=50,
        completion_tokens=100,
        total_tokens=150,
        cached_tokens=None,  # No cached tokens
        reasoning_tokens=60,  # Has reasoning tokens
    )
    
    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )
    
    # Should succeed and default cached_tokens to 0
    assert responses_usage.input_tokens == 50
    assert responses_usage.output_tokens == 100
    assert responses_usage.total_tokens == 150
    
    # Input details should not be present (no cached_tokens provided)
    assert responses_usage.input_tokens_details is None
    
    # Output details should be present with reasoning_tokens
    assert responses_usage.output_tokens_details is not None
    assert isinstance(responses_usage.output_tokens_details, OutputTokensDetails)
    assert responses_usage.output_tokens_details.reasoning_tokens == 60
    
    print("✓ Transformation works with reasoning_tokens only")


def test_transform_usage_with_both_token_details():
    """
    Test transformation when both cached_tokens and reasoning_tokens are provided.
    
    This simulates advanced providers that support both features.
    """
    completion_response = create_mock_completion_response(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=80,
        total_tokens=180,
        cached_tokens=50,
        reasoning_tokens=30,
        text_tokens=50,  # Also include text_tokens
    )
    
    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )
    
    # Should succeed with all details
    assert responses_usage.input_tokens == 100
    assert responses_usage.output_tokens == 80
    assert responses_usage.total_tokens == 180
    
    # Input details should have cached_tokens
    assert responses_usage.input_tokens_details is not None
    assert responses_usage.input_tokens_details.cached_tokens == 50
    assert responses_usage.input_tokens_details.text_tokens == 50
    
    # Output details should have reasoning_tokens
    assert responses_usage.output_tokens_details is not None
    assert responses_usage.output_tokens_details.reasoning_tokens == 30
    assert responses_usage.output_tokens_details.text_tokens == 50
    
    print("✓ Transformation works with both cached_tokens and reasoning_tokens")


def test_transform_usage_with_zero_values():
    """
    Test transformation when token details are explicitly set to 0.
    
    This ensures 0 values are preserved and not treated as None.
    """
    completion_response = create_mock_completion_response(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=0,  # Explicitly 0
        reasoning_tokens=0,  # Explicitly 0
    )
    
    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )
    
    # Should preserve 0 values
    assert responses_usage.input_tokens_details is not None
    assert responses_usage.input_tokens_details.cached_tokens == 0
    
    assert responses_usage.output_tokens_details is not None
    assert responses_usage.output_tokens_details.reasoning_tokens == 0
    
    print("✓ Transformation preserves explicit 0 values")


def test_input_tokens_details_requires_cached_tokens():
    """
    Test that InputTokensDetails has cached_tokens as an int with default value 0.
    
    This ensures backward compatibility while making the field non-optional.
    """
    # Should work with cached_tokens=0
    details1 = InputTokensDetails(cached_tokens=0)
    assert details1.cached_tokens == 0
    
    # Should work with cached_tokens=100
    details2 = InputTokensDetails(cached_tokens=100)
    assert details2.cached_tokens == 100
    
    # Should work without cached_tokens (defaults to 0)
    details3 = InputTokensDetails()
    assert details3.cached_tokens == 0
    
    print("✓ InputTokensDetails correctly defaults cached_tokens to 0")


def test_output_tokens_details_requires_reasoning_tokens():
    """
    Test that OutputTokensDetails has reasoning_tokens as an int with default value 0.
    
    This ensures backward compatibility while making the field non-optional.
    """
    # Should work with reasoning_tokens=0
    details1 = OutputTokensDetails(reasoning_tokens=0)
    assert details1.reasoning_tokens == 0
    
    # Should work with reasoning_tokens=100
    details2 = OutputTokensDetails(reasoning_tokens=100)
    assert details2.reasoning_tokens == 100
    
    # Should work without reasoning_tokens (defaults to 0)
    details3 = OutputTokensDetails()
    assert details3.reasoning_tokens == 0
    
    print("✓ OutputTokensDetails correctly defaults reasoning_tokens to 0")


def test_all_providers_transformation_scenarios():
    """
    Test various provider scenarios to ensure none break after the field requirement change.
    
    This tests the most common scenarios across different providers:
    - OpenAI: may have reasoning_tokens
    - Anthropic: may have cached_tokens
    - Azure: similar to OpenAI
    - Other providers: basic usage only
    """
    test_scenarios = [
        {
            "name": "Basic provider (no details)",
            "model": "gpt-3.5-turbo",
            "kwargs": {},
        },
        {
            "name": "OpenAI with reasoning",
            "model": "o1-preview",
            "kwargs": {"reasoning_tokens": 100},
        },
        {
            "name": "Anthropic with caching",
            "model": "claude-3-opus",
            "kwargs": {"cached_tokens": 50},
        },
        {
            "name": "OpenAI with caching",
            "model": "gpt-4o",
            "kwargs": {"cached_tokens": 30},
        },
        {
            "name": "Full details (both)",
            "model": "gpt-4o",
            "kwargs": {"cached_tokens": 40, "reasoning_tokens": 60, "text_tokens": 100},
        },
        {
            "name": "Zero values",
            "model": "gpt-4",
            "kwargs": {"cached_tokens": 0, "reasoning_tokens": 0},
        },
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        completion_response = create_mock_completion_response(
            model=scenario["model"],
            **scenario["kwargs"]
        )
        
        # This should not raise any errors
        responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            completion_response
        )
        
        # Basic assertions
        assert responses_usage.input_tokens >= 0
        assert responses_usage.output_tokens >= 0
        assert responses_usage.total_tokens >= 0
        
        # If input_tokens_details exists, cached_tokens must be an int
        if responses_usage.input_tokens_details is not None:
            assert isinstance(responses_usage.input_tokens_details.cached_tokens, int)
        
        # If output_tokens_details exists, reasoning_tokens must be an int
        if responses_usage.output_tokens_details is not None:
            assert isinstance(responses_usage.output_tokens_details.reasoning_tokens, int)
        
        print(f"  ✓ {scenario['name']} transformation successful")
    
    print("\n✓ All provider scenarios work correctly")


if __name__ == "__main__":
    # Run all tests
    test_transform_usage_no_token_details()
    test_transform_usage_with_cached_tokens_only()
    test_transform_usage_with_reasoning_tokens_only()
    test_transform_usage_with_both_token_details()
    test_transform_usage_with_zero_values()
    test_input_tokens_details_requires_cached_tokens()
    test_output_tokens_details_requires_reasoning_tokens()
    test_all_providers_transformation_scenarios()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
