import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import json

from litellm.types.utils import HiddenParams


def test_hidden_params_response_ms():
    hidden_params = HiddenParams()
    setattr(hidden_params, "_response_ms", 100)
    hidden_params_dict = hidden_params.model_dump()
    assert hidden_params_dict.get("_response_ms") == 100


def test_chat_completion_delta_tool_call():
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    tool = ChatCompletionDeltaToolCall(
        id="call_m87w",
        function=Function(
            arguments='{"location": "San Francisco", "unit": "imperial"}',
            name="get_current_weather",
        ),
        type="function",
        index=0,
    )

    assert "function" in tool


def test_empty_choices():
    from litellm.types.utils import Choices

    Choices()


def test_usage_dump():
    from litellm.types.utils import (
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
        Usage,
    )

    current_usage = Usage(
        completion_tokens=37,
        prompt_tokens=7,
        total_tokens=44,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=None,
            text_tokens=7,
            image_tokens=None,
            web_search_requests=1,
        ),
        web_search_requests=None,
    )

    assert current_usage.prompt_tokens_details.web_search_requests == 1

    new_usage = Usage(**current_usage.model_dump())
    assert new_usage.prompt_tokens_details.web_search_requests == 1


def test_usage_completion_tokens_details_text_tokens():
    from litellm.types.utils import Usage

    # Test data from the reported issue
    usage_data = {
        'completion_tokens': 77,
        'prompt_tokens': 11937,
        'total_tokens': 12014,
        'completion_tokens_details': {
            'accepted_prediction_tokens': None,
            'audio_tokens': None,
            'reasoning_tokens': 65,
            'rejected_prediction_tokens': None,
            'text_tokens': 12
        },
        'prompt_tokens_details': {
            'audio_tokens': None,
            'cached_tokens': None,
            'text_tokens': 11937,
            'image_tokens': None
        }
    }

    # Create Usage object
    u = Usage(**usage_data)
    
    # Verify the object has the text_tokens field
    assert hasattr(u.completion_tokens_details, 'text_tokens')
    assert u.completion_tokens_details.text_tokens == 12
    
    # Get model_dump output
    dump_result = u.model_dump()
    
    # Verify text_tokens is present in the model_dump output
    assert 'completion_tokens_details' in dump_result
    assert 'text_tokens' in dump_result['completion_tokens_details']
    assert dump_result['completion_tokens_details']['text_tokens'] == 12
    
    # Verify the full completion_tokens_details structure
    expected_completion_details = {
        'accepted_prediction_tokens': None,
        'audio_tokens': None,
        'reasoning_tokens': 65,
        'rejected_prediction_tokens': None,
        'text_tokens': 12,
        'image_tokens': None
    }
    assert dump_result['completion_tokens_details'] == expected_completion_details
    
    # Verify round-trip serialization works
    new_usage = Usage(**dump_result)
    assert new_usage.completion_tokens_details.text_tokens == 12


def test_usage_openai_cached_tokens_populates_cache_read_input_tokens():
    """
    Test that OpenAI's prompt_tokens_details.cached_tokens populates
    _cache_read_input_tokens. This is the fix for GH issue #19684.

    OpenAI returns cached tokens in prompt_tokens_details.cached_tokens,
    but _cache_read_input_tokens was not being set, causing the UI to
    show "Cache Read Tokens: 0".
    """
    from litellm.types.utils import Usage

    # Simulate OpenAI response usage (exactly what comes from response.model_dump())
    openai_usage = {
        "prompt_tokens": 2829,
        "completion_tokens": 29,
        "total_tokens": 2858,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 2816,
        },
    }

    usage = Usage(**openai_usage)

    # _cache_read_input_tokens should be populated from prompt_tokens_details.cached_tokens
    assert usage._cache_read_input_tokens == 2816
    # prompt_tokens_details.cached_tokens should also be preserved
    assert usage.prompt_tokens_details.cached_tokens == 2816


def test_usage_openai_cached_tokens_zero_does_not_set_cache_read():
    """
    When OpenAI returns cached_tokens=0, _cache_read_input_tokens should stay 0.
    """
    from litellm.types.utils import Usage

    openai_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0,
        },
    }

    usage = Usage(**openai_usage)
    assert usage._cache_read_input_tokens == 0


def test_usage_anthropic_cache_read_not_overwritten_by_prompt_details():
    """
    When Anthropic explicitly passes cache_read_input_tokens, the OpenAI
    fallback mapping should NOT overwrite it.
    """
    from litellm.types.utils import Usage

    # Anthropic passes cache_read_input_tokens explicitly in **params
    # Use different values to verify the explicit param wins over prompt_tokens_details
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=50,
        total_tokens=1050,
        prompt_tokens_details={"cached_tokens": 300},
        cache_read_input_tokens=500,
    )

    # Should use the explicit Anthropic value (500), not the prompt_tokens_details value (300)
    assert usage._cache_read_input_tokens == 500
    assert usage.prompt_tokens_details.cached_tokens == 500


def test_usage_no_prompt_tokens_details_no_error():
    """
    When there's no prompt_tokens_details at all, nothing should break.
    """
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
    )

    assert usage._cache_read_input_tokens == 0
