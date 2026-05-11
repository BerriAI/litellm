"""
Test that tool_choice is correctly transformed from Chat Completions format
to Responses API format in OpenAIResponsesAPIConfig.
"""

import pytest
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig


def test_normalize_tool_choice_nested_to_flat():
    """Chat Completions nested format should be flattened for Responses API."""
    result = OpenAIResponsesAPIConfig._normalize_tool_choice_for_responses_api(
        {"type": "function", "function": {"name": "Echo"}}
    )
    assert result == {"type": "function", "name": "Echo"}


def test_normalize_tool_choice_already_flat():
    """Already flat format should pass through unchanged."""
    result = OpenAIResponsesAPIConfig._normalize_tool_choice_for_responses_api(
        {"type": "function", "name": "Echo"}
    )
    assert result == {"type": "function", "name": "Echo"}


def test_normalize_tool_choice_string_passthrough():
    """String values like 'auto', 'required', 'none' should pass through unchanged."""
    for value in ["auto", "required", "none"]:
        result = OpenAIResponsesAPIConfig._normalize_tool_choice_for_responses_api(value)
        assert result == value


def test_transform_responses_api_request_normalizes_tool_choice():
    """transform_responses_api_request should normalize tool_choice before sending."""
    config = OpenAIResponsesAPIConfig()
    params = {
        "tool_choice": {"type": "function", "function": {"name": "Echo"}},
        "tools": [],
    }
    result = config.transform_responses_api_request(
        model="gpt-5.4-nano",
        input="Say hello",
        response_api_optional_request_params=params,
        litellm_params=None,
        headers={},
    )
    assert result["tool_choice"] == {"type": "function", "name": "Echo"}

    