import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm import supports_reasoning
from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig
from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message


def test_handle_message_content_with_tool_calls():
    config = FireworksAIConfig()
    message = Message(
        content='{"type": "function", "name": "get_current_weather", "parameters": {"location": "Boston, MA", "unit": "fahrenheit"}}',
        role="assistant",
        tool_calls=None,
        function_call=None,
        provider_specific_fields=None,
    )
    expected_tool_call = ChatCompletionMessageToolCall(
        function=Function(**json.loads(message.content)), id=None, type=None
    )
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    updated_message = config._handle_message_content_with_tool_calls(
        message, tool_calls
    )
    assert updated_message.tool_calls is not None
    assert len(updated_message.tool_calls) == 1
    assert updated_message.tool_calls[0].function.name == "get_current_weather"
    assert (
        updated_message.tool_calls[0].function.arguments
        == expected_tool_call.function.arguments
    )


def test_supports_reasoning_effort():
    """Test that reasoning_effort is only supported for specific Fireworks AI models."""
    # Models that support reasoning_effort
    supported_models = [
        "fireworks_ai/accounts/fireworks/models/qwen3-8b",
        "fireworks_ai/accounts/fireworks/models/qwen3-32b",
        "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
        "fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
        "fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
        "fireworks_ai/accounts/fireworks/models/glm-4p5",
        "fireworks_ai/accounts/fireworks/models/glm-4p5-air",
        "fireworks_ai/accounts/fireworks/models/glm-4p6",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-20b",
    ]

    # Models that don't support reasoning_effort
    unsupported_models = [
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct",
        "fireworks_ai/accounts/fireworks/models/mixtral-8x7b-instruct",
    ]

    for model in supported_models:
        assert (
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") == True
        ), f"{model} should support reasoning_effort"

    for model in unsupported_models:
        assert (
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") == False
        ), f"{model} should not support reasoning_effort"


def test_get_supported_openai_params_reasoning_effort():
    """Test that reasoning_effort is only included in supported params for models that support it."""
    config = FireworksAIConfig()

    # Model that supports reasoning_effort
    supported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/qwen3-8b"
    )
    assert "reasoning_effort" in supported_params

    # Model that doesn't support reasoning_effort
    unsupported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"
    )
    assert "reasoning_effort" not in unsupported_params


def test_transform_messages_helper_removes_provider_specific_fields():
    """
    Test that _transform_messages_helper removes provider_specific_fields from messages.
    """
    config = FireworksAIConfig()
    # Simulated messages, as dicts, including provider_specific_fields
    messages = [
        {
            "role": "user",
            "content": "Hello!",
            "provider_specific_fields": {"extra": "should be removed"},
        },
        {
            "role": "assistant",
            "content": "Hi there!",
            "provider_specific_fields": {"more": "remove this"},
        },
        {
            "role": "user",
            "content": "How are you?",
            # no provider_specific_fields
        }
    ]
    # Call helper
    out = config._transform_messages_helper(messages, model="fireworks/test", litellm_params={})
    for msg in out:
        assert "provider_specific_fields" not in msg
