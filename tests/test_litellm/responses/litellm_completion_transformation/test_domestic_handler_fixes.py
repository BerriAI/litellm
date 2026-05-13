"""
Test domestic model compatibility fixes in handler.py
"""

import json
import pytest

from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)


class TestEnsureValidJsonArguments:
    """Test _ensure_all_tool_calls_have_valid_json_arguments method"""

    def test_valid_json_arguments_preserved(self):
        """Test that valid JSON arguments are preserved"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expr": "15 + 27"}',
                        },
                    }
                ],
            }
        ]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        assert len(result) == 1
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"expr": "15 + 27"}'

    def test_null_arguments_converted_to_empty_json(self):
        """Test that null arguments are converted to {}"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": None},
                    }
                ],
            }
        ]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_empty_string_arguments_converted(self):
        """Test that empty string arguments are converted to {}"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": ""},
                    }
                ],
            }
        ]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_invalid_json_arguments_converted(self):
        """Test that invalid JSON arguments are converted to {}"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "not valid json",
                        },
                    }
                ],
            }
        ]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_dict_arguments_converted_to_json_string(self):
        """Test that dict arguments are converted to JSON string"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": {"expr": "15 + 27"},  # dict, not string
                        },
                    }
                ],
            }
        ]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"expr": "15 + 27"}

    def test_non_assistant_messages_not_modified(self):
        """Test that non-assistant messages are not modified"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are helpful"},
        ]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "system"

    def test_multiple_tool_calls_all_fixed(self):
        """Test that all invalid tool calls in a message are fixed"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool1", "arguments": None},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool2", "arguments": "invalid"},
                    },
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "tool3", "arguments": '{"valid": true}'},
                    },
                ],
            }
        ]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        tool_calls = result[0]["tool_calls"]
        assert tool_calls[0]["function"]["arguments"] == "{}"
        assert tool_calls[1]["function"]["arguments"] == "{}"
        assert tool_calls[2]["function"]["arguments"] == '{"valid": true}'


class TestToolCallsStructureRebuild:
    """Test that tool_calls are rebuilt as pure dict structures"""

    def test_pydantic_like_object_converted_to_dict(self):
        """Test that Pydantic-like objects are converted to plain dicts"""

        # Create a mock object that behaves like Pydantic model
        class MockFunction:
            def __init__(self):
                self.name = "calculator"
                self.arguments = "invalid json"

        class MockToolCall:
            def __init__(self):
                self.id = "call_123"
                self.type = "function"
                self.function = MockFunction()

        class MockMessage:
            def __init__(self):
                self.role = "assistant"
                self.tool_calls = [MockToolCall()]

        messages = [MockMessage()]

        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )

        # Should be rebuilt as dict
        assert isinstance(result[0], dict)
        assert isinstance(result[0]["tool_calls"][0], dict)
        assert isinstance(result[0]["tool_calls"][0]["function"], dict)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"