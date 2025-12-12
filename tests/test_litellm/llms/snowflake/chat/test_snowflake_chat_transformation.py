"""
Unit tests for Snowflake chat transformation
Tests tool calling request/response transformations
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.snowflake.chat.transformation import SnowflakeConfig
from litellm.types.utils import ModelResponse


class TestSnowflakeToolTransformation:
    """Test suite for Snowflake tool calling transformations"""

    def test_transform_request_with_tools(self):
        """
        Test that OpenAI tool format is correctly transformed to Snowflake's tool_spec format.
        """
        config = SnowflakeConfig()

        # OpenAI format tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        optional_params = {"tools": tools}

        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "What's the weather?"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify tools were transformed to Snowflake format
        assert "tools" in transformed_request
        assert len(transformed_request["tools"]) == 1

        snowflake_tool = transformed_request["tools"][0]
        assert "tool_spec" in snowflake_tool
        assert snowflake_tool["tool_spec"]["type"] == "generic"
        assert snowflake_tool["tool_spec"]["name"] == "get_weather"
        assert snowflake_tool["tool_spec"]["description"] == "Get the current weather in a given location"
        assert "input_schema" in snowflake_tool["tool_spec"]
        assert snowflake_tool["tool_spec"]["input_schema"]["type"] == "object"
        assert "location" in snowflake_tool["tool_spec"]["input_schema"]["properties"]

    def test_transform_request_with_tool_choice(self):
        """
        Test that OpenAI tool_choice format is correctly transformed to Snowflake format.
        """
        config = SnowflakeConfig()

        # OpenAI format tool_choice
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        optional_params = {"tool_choice": tool_choice}

        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "What's the weather?"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify tool_choice was transformed to Snowflake format
        assert "tool_choice" in transformed_request
        assert transformed_request["tool_choice"]["type"] == "tool"
        assert transformed_request["tool_choice"]["name"] == ["get_weather"]  # Array format

    def test_transform_request_with_string_tool_choice(self):
        """
        Test that string tool_choice values are transformed to Snowflake object format.
        """
        config = SnowflakeConfig()

        # Test "auto" - should become {"type": "auto"}
        optional_params = {"tool_choice": "auto"}
        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Test"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert transformed_request["tool_choice"] == {"type": "auto"}

        # Test "none" - should become {"type": "none"}
        optional_params = {"tool_choice": "none"}
        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Test"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert transformed_request["tool_choice"] == {"type": "none"}

        # Test "required" - should become {"type": "required", "name": [...]} if tools present
        optional_params = {"tool_choice": "required"}
        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Test"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        # Without tools, "required" becomes just {"type": "required"}
        assert transformed_request["tool_choice"]["type"] == "required"

    def test_transform_response_with_tool_calls(self):
        """
        Test that Snowflake's content_list with tool_use is transformed to OpenAI format.
        """
        config = SnowflakeConfig()

        # Mock Snowflake response with tool call
        mock_snowflake_response = {
            "choices": [
                {
                    "message": {
                        "content_list": [
                            {"type": "text", "text": ""},
                            {
                                "type": "tool_use",
                                "tool_use": {
                                    "tool_use_id": "tooluse_abc123",
                                    "name": "get_weather",
                                    "input": {"location": "Paris, France", "unit": "celsius"},
                                },
                            },
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        response = httpx.Response(
            status_code=200,
            json=mock_snowflake_response,
            headers={"Content-Type": "application/json"},
        )

        model_response = ModelResponse(
            choices=[litellm.Choices(index=0, message=litellm.Message())]
        )

        logging_obj = MagicMock()

        result = config.transform_response(
            model="claude-3-5-sonnet",
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # General assertions
        assert isinstance(result, ModelResponse)
        assert len(result.choices) == 1

        choice = result.choices[0]
        assert isinstance(choice, litellm.Choices)

        # Message and tool_calls assertions
        message = choice.message
        assert isinstance(message, litellm.Message)
        assert hasattr(message, "tool_calls")
        assert isinstance(message.tool_calls, list)
        assert len(message.tool_calls) == 1

        # Specific tool_call assertions
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, litellm.utils.ChatCompletionMessageToolCall)
        assert tool_call.id == "tooluse_abc123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"

        # Verify arguments are properly JSON serialized
        arguments = json.loads(tool_call.function.arguments)
        assert arguments["location"] == "Paris, France"
        assert arguments["unit"] == "celsius"

        # Verify content_list was removed and content was set
        assert message.content == ""

    def test_transform_response_with_mixed_content(self):
        """
        Test that responses with both text and tool calls are handled correctly.
        """
        config = SnowflakeConfig()

        # Mock Snowflake response with text and tool call
        mock_snowflake_response = {
            "choices": [
                {
                    "message": {
                        "content_list": [
                            {"type": "text", "text": "Let me check the weather for you. "},
                            {
                                "type": "tool_use",
                                "tool_use": {
                                    "tool_use_id": "tooluse_xyz789",
                                    "name": "get_weather",
                                    "input": {"location": "Tokyo, Japan"},
                                },
                            },
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        }

        response = httpx.Response(
            status_code=200,
            json=mock_snowflake_response,
            headers={"Content-Type": "application/json"},
        )

        model_response = ModelResponse(
            choices=[litellm.Choices(index=0, message=litellm.Message())]
        )

        logging_obj = MagicMock()

        result = config.transform_response(
            model="claude-3-5-sonnet",
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # Verify text content was extracted
        message = result.choices[0].message
        assert message.content == "Let me check the weather for you. "

        # Verify tool call was also extracted
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].function.name == "get_weather"

    def test_transform_response_without_tool_calls(self):
        """
        Test that regular text responses (without tools) work correctly.
        """
        config = SnowflakeConfig()

        # Mock Snowflake response without tool calls (standard response)
        mock_snowflake_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello! I'm doing well, thank you for asking.",
                        "role": "assistant",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

        response = httpx.Response(
            status_code=200,
            json=mock_snowflake_response,
            headers={"Content-Type": "application/json"},
        )

        model_response = ModelResponse(
            choices=[litellm.Choices(index=0, message=litellm.Message())]
        )

        logging_obj = MagicMock()

        result = config.transform_response(
            model="mistral-7b",
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # Verify standard response works
        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

    def test_get_supported_openai_params_includes_tools(self):
        """
        Test that tools and tool_choice are in supported params.
        """
        config = SnowflakeConfig()
        supported_params = config.get_supported_openai_params("claude-3-5-sonnet")

        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params


class TestSnowflakeAuthenticationHeaders:
    """Test suite for Snowflake authentication header handling"""

    def test_validate_environment_with_jwt(self):
        """
        Test that JWT tokens are handled correctly with KEYPAIR_JWT header.
        """
        config = SnowflakeConfig()
        headers = {}

        jwt_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.test"

        result_headers = config.validate_environment(
            headers=headers,
            model="mistral-7b",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=jwt_token,
            api_base=None,
        )

        assert result_headers["Authorization"] == f"Bearer {jwt_token}"
        assert result_headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"
        assert result_headers["Content-Type"] == "application/json"
        assert result_headers["Accept"] == "application/json"

    def test_validate_environment_with_pat_token(self):
        """
        Test that PAT tokens with pat/ prefix are handled correctly.
        The pat/ prefix should be stripped and PROGRAMMATIC_ACCESS_TOKEN should be used.
        """
        config = SnowflakeConfig()
        headers = {}

        pat_token = "pat/abc123xyz789"
        expected_token = "abc123xyz789"

        result_headers = config.validate_environment(
            headers=headers,
            model="mistral-7b",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=pat_token,
            api_base=None,
        )

        assert result_headers["Authorization"] == f"Bearer {expected_token}"
        assert result_headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
        assert result_headers["Content-Type"] == "application/json"
        assert result_headers["Accept"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """
        Test that missing API key raises ValueError.
        """
        config = SnowflakeConfig()
        headers = {}

        with pytest.raises(ValueError, match="Missing Snowflake JWT or PAT key"):
            config.validate_environment(
                headers=headers,
                model="mistral-7b",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )


class TestSnowflakeMessageTransformation:
    """Test suite for Snowflake message transformation (tool results and tool calls)"""

    def test_transform_messages_with_tool_result(self):
        """
        Test that OpenAI tool result messages are transformed to Snowflake format.
        """
        config = SnowflakeConfig()

        # OpenAI format with tool result
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris, France"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "Temperature: 18°C, Condition: Partly cloudy",
            },
        ]

        transformed = config._transform_messages(messages)

        # First message (user) should pass through
        assert transformed[0]["role"] == "user"
        assert transformed[0]["content"] == "What's the weather in Paris?"

        # Second message (assistant with tool_calls) should be transformed
        assert transformed[1]["role"] == "assistant"
        assert "content" in transformed[1]
        assert transformed[1]["content"] == "Let me check the weather for you."
        assert "content_list" in transformed[1]
        assert len(transformed[1]["content_list"]) == 1  # Only tool_use, text is in content field
        assert transformed[1]["content_list"][0]["type"] == "tool_use"
        assert transformed[1]["content_list"][0]["tool_use"]["tool_use_id"] == "call_abc123"
        assert transformed[1]["content_list"][0]["tool_use"]["name"] == "get_weather"
        assert transformed[1]["content_list"][0]["tool_use"]["input"]["location"] == "Paris, France"

        # Third message (tool result) should be transformed to user message with content_list
        assert transformed[2]["role"] == "user"
        assert "content" in transformed[2]  # Must have content field
        assert transformed[2]["content"] == ""  # Empty for tool results
        assert "content_list" in transformed[2]
        assert len(transformed[2]["content_list"]) == 1
        assert transformed[2]["content_list"][0]["type"] == "tool_results"
        tool_results = transformed[2]["content_list"][0]["tool_results"]
        assert tool_results["tool_use_id"] == "call_abc123"
        assert tool_results["name"] == "get_weather"  # Name should be tracked from tool_call
        assert len(tool_results["content"]) == 1
        assert tool_results["content"][0]["type"] == "text"
        assert tool_results["content"][0]["text"] == "Temperature: 18°C, Condition: Partly cloudy"

    def test_transform_messages_regular_conversation(self):
        """
        Test that regular messages without tools pass through unchanged.
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
        ]

        transformed = config._transform_messages(messages)

        # All messages should pass through unchanged
        assert len(transformed) == 3
        assert transformed[0] == messages[0]
        assert transformed[1] == messages[1]
        assert transformed[2] == messages[2]

    def test_transform_messages_assistant_without_content(self):
        """
        Test assistant messages with only tool_calls (no text content).
        """
        config = SnowflakeConfig()

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz789",
                        "type": "function",
                        "function": {
                            "name": "search_database",
                            "arguments": '{"query": "users"}',
                        },
                    }
                ],
            }
        ]

        transformed = config._transform_messages(messages)

        assert len(transformed) == 1
        assert transformed[0]["role"] == "assistant"
        assert "content" in transformed[0]
        assert transformed[0]["content"] == ""  # Empty when no text content
        assert "content_list" in transformed[0]
        # Should only have tool_use
        assert len(transformed[0]["content_list"]) == 1
        assert transformed[0]["content_list"][0]["type"] == "tool_use"

    def test_transform_messages_multiple_tool_calls(self):
        """
        Test assistant message with multiple tool calls.
        """
        config = SnowflakeConfig()

        messages = [
            {
                "role": "assistant",
                "content": "I'll check both locations for you.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    },
                ],
            }
        ]

        transformed = config._transform_messages(messages)

        assert len(transformed) == 1
        assert transformed[0]["role"] == "assistant"
        assert "content" in transformed[0]
        assert transformed[0]["content"] == "I'll check both locations for you."
        assert len(transformed[0]["content_list"]) == 2  # 2 tool_use (text is in content field)
        assert transformed[0]["content_list"][0]["type"] == "tool_use"
        assert transformed[0]["content_list"][1]["type"] == "tool_use"
        assert transformed[0]["content_list"][0]["tool_use"]["tool_use_id"] == "call_1"
        assert transformed[0]["content_list"][1]["tool_use"]["tool_use_id"] == "call_2"

    def test_transform_request_integrates_message_transformation(self):
        """
        Test that transform_request properly calls _transform_messages.
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "tool",
                "tool_call_id": "call_test",
                "content": "Tool result",
            },
        ]

        result = config.transform_request(
            model="claude-3-5-sonnet",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Check that messages were transformed
        assert "messages" in result
        assert len(result["messages"]) == 2
        # Second message should be transformed from tool to user with content_list
        assert result["messages"][1]["role"] == "user"
        assert "content" in result["messages"][1]  # Must have content field
        assert "content_list" in result["messages"][1]

    def test_transform_messages_with_none_content(self):
        """
        Test that messages with content=None are normalized to have content=""
        This is critical for previous_response_id scenarios where loaded messages
        may have None content.
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},  # content=None (no tool_calls)
            {"role": "assistant"},  # Missing content field entirely
        ]

        transformed = config._transform_messages(messages)

        # All messages should have content field, even if it was None or missing
        assert len(transformed) == 3
        assert transformed[0]["content"] == "Hello"
        assert "content" in transformed[1]
        assert transformed[1]["content"] == ""  # None should become empty string
        assert "content" in transformed[2]
        assert transformed[2]["content"] == ""  # Missing should become empty string

    def test_transform_messages_preserves_existing_content(self):
        """
        Test that messages with valid content are not modified
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I can help with that."},
        ]

        transformed = config._transform_messages(messages)

        assert len(transformed) == 3
        assert transformed[0]["content"] == "You are a helpful assistant."
        assert transformed[1]["content"] == "What's the weather?"
        assert transformed[2]["content"] == "I can help with that."

    def test_transform_messages_previous_response_id_scenario(self):
        """
        Test the exact scenario that causes the error:
        - First request returns assistant message with tool_calls (content may be None)
        - Message is loaded from spend logs without tool_calls field
        - Should be normalized to have content field
        """
        config = SnowflakeConfig()

        # Simulate messages loaded from previous_response_id
        # The assistant message from the first response no longer has tool_calls
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,  # Was returned from Snowflake with no text, only tool_calls
                # Note: tool_calls field is NOT present because it was consumed
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "Temperature: 18°C",
            },
        ]

        transformed = config._transform_messages(messages)

        # All messages should be properly formatted
        assert len(transformed) == 3
        assert transformed[0]["role"] == "user"
        assert transformed[0]["content"] == "What's the weather in Paris?"

        # The problematic assistant message should now have content field
        assert transformed[1]["role"] == "assistant"
        assert "content" in transformed[1]
        assert transformed[1]["content"] == ""  # Normalized from None

        # Tool result should be transformed properly
        assert transformed[2]["role"] == "user"
        assert "content" in transformed[2]
        assert transformed[2]["content"] == ""


class TestSnowflakeStreamingHandler:
    """Test suite for Snowflake streaming response handling"""

    def test_chunk_parser_with_created_field(self):
        """
        Test that streaming chunks with 'created' field are parsed correctly.
        This is the standard case for models like mistral-7b and llama3.3.
        """
        from litellm.llms.snowflake.chat.transformation import (
            SnowflakeStreamingHandler,
        )

        handler = SnowflakeStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = {
            "id": "chatcmpl-123",
            "created": 1234567890,
            "model": "mistral-7b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "chatcmpl-123"
        assert result.created == 1234567890
        assert result.model == "mistral-7b"
        assert result.object == "chat.completion.chunk"
        assert len(result.choices) == 1

    def test_chunk_parser_without_created_field(self):
        """
        Test that streaming chunks WITHOUT 'created' field are parsed correctly.
        This handles the case for Claude models (sonnet-3.5, sonnet-4-5) which
        don't include the 'created' field in their streaming responses.
        """
        from litellm.llms.snowflake.chat.transformation import (
            SnowflakeStreamingHandler,
        )

        handler = SnowflakeStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # Chunk without 'created' field (like claude-sonnet-4-5)
        chunk = {
            "id": "chatcmpl-456",
            "model": "claude-sonnet-4-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hi there"},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "chatcmpl-456"
        assert result.created is not None  # Should have a default timestamp
        assert isinstance(result.created, int)  # Should be an integer timestamp
        assert result.model == "claude-sonnet-4-5"
        assert result.object == "chat.completion.chunk"
        assert len(result.choices) == 1

    def test_get_model_response_iterator(self):
        """
        Test that SnowflakeConfig returns the custom streaming handler.
        """
        from litellm.llms.snowflake.chat.transformation import (
            SnowflakeStreamingHandler,
            SnowflakeConfig,
        )

        config = SnowflakeConfig()

        handler = config.get_model_response_iterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        assert isinstance(handler, SnowflakeStreamingHandler)
