"""
Test domestic model compatibility fixes in handler.py and domestic_utils.py
"""

import json

from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.llms.domestic.domestic_utils import (
    is_domestic_model_or_endpoint,
)


class TestDomesticModelDetection:
    """Test is_domestic_model_or_endpoint function"""

    def test_qwen_model_detected(self):
        """Test qwen models are detected as domestic"""
        assert is_domestic_model_or_endpoint("qwen3.5-plus", None)
        assert is_domestic_model_or_endpoint("qwen3-max", None)

    def test_deepseek_model_detected(self):
        """Test deepseek models are detected as domestic"""
        assert is_domestic_model_or_endpoint("deepseek-chat", None)
        assert is_domestic_model_or_endpoint("deepseek-coder", None)

    def test_doubao_model_detected(self):
        """Test doubao models are detected as domestic"""
        assert is_domestic_model_or_endpoint("doubao-pro", None)

    def test_openai_model_not_domestic(self):
        """Test OpenAI models are not domestic"""
        assert not is_domestic_model_or_endpoint("gpt-4", None)
        assert not is_domestic_model_or_endpoint("gpt-3.5-turbo", None)

    def test_dashscope_endpoint_detected(self):
        """Test DashScope endpoint is detected"""
        assert is_domestic_model_or_endpoint(
            "any-model", "https://dashscope.aliyuncs.com/api/v1"
        )

    def test_volcengine_endpoint_detected(self):
        """Test Volcengine endpoint is detected"""
        assert is_domestic_model_or_endpoint(
            "any-model", "https://ark.cn-beijing.volces.com/api/v3"
        )

    def test_openai_endpoint_not_domestic(self):
        """Test OpenAI endpoint is not domestic"""
        assert not is_domestic_model_or_endpoint("gpt-4", "https://api.openai.com/v1")

    def test_none_model_and_endpoint(self):
        """Test None model and endpoint returns False"""
        assert not is_domestic_model_or_endpoint(None, None)

    def test_empty_strings(self):
        """Test empty strings return False"""
        assert not is_domestic_model_or_endpoint("", "")


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
        assert (
            result[0]["tool_calls"][0]["function"]["arguments"] == '{"expr": "15 + 27"}'
        )

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
                            "arguments": {"expr": "15 + 27"},
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


class TestEdgeCases:
    """Test edge cases in arguments validation"""

    def test_whitespace_only_arguments(self):
        """Test whitespace-only arguments converted to {}"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "   "},
                    }
                ],
            }
        ]
        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_valid_nested_json_preserved(self):
        """Test valid nested JSON is preserved"""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "tool",
                            "arguments": '{"nested": {"key": "value"}}',
                        },
                    }
                ],
            }
        ]
        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )
        assert (
            result[0]["tool_calls"][0]["function"]["arguments"]
            == '{"nested": {"key": "value"}}'
        )

    def test_user_message_not_modified(self):
        """Test user messages are not modified"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_assistant_without_tool_calls_not_modified(self):
        """Test assistant messages without tool_calls not modified"""
        messages = [
            {"role": "assistant", "content": "Hello"},
        ]
        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello"

    def test_empty_tool_calls_list(self):
        """Test empty tool_calls list"""
        messages = [
            {"role": "assistant", "tool_calls": []},
        ]
        result = LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
            messages
        )
        assert result[0]["tool_calls"] == []
