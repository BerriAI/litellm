"""
Unit tests for Snowflake chat transformation
Tests tool calling request/response transformations
"""

import os
import copy
import json

from unittest.mock import patch
from unittest.mock import MagicMock

import httpx

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
        assert (
            snowflake_tool["tool_spec"]["description"]
            == "Get the current weather in a given location"
        )
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
        assert transformed_request["tool_choice"]["name"] == [
            "get_weather"
        ]  # Array format

    def test_transform_request_with_string_tool_choice(self):
        """
        Test that string tool_choice values pass through unchanged.
        """
        config = SnowflakeConfig()

        for value in ["auto", "required", "none"]:
            optional_params = {"tool_choice": value}

            transformed_request = config.transform_request(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Test"}],
                optional_params=optional_params,
                litellm_params={},
                headers={},
            )

            assert transformed_request["tool_choice"] == value

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
                                    "input": {
                                        "location": "Paris, France",
                                        "unit": "celsius",
                                    },
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
                            {
                                "type": "text",
                                "text": "Let me check the weather for you. ",
                            },
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
        assert (
            result.choices[0].message.content
            == "Hello! I'm doing well, thank you for asking."
        )

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


class TestSnowFlakeCompletion:
    model_name = "mistral"

    messages = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "the capital of France"},
    ]

    response = {
        "choices": [
            {
                "message": {
                    "content": "Paris",
                    "content_list": [{"type": "text", "text": "Paris"}],
                }
            }
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 18, "total_tokens": 34},
    }

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_snowflake_jwt_account_id(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response)

        response = litellm.completion(
            f"snowflake/{self.model_name}",
            messages=self.messages,
            api_key="00000",
            account_id="AAAA-BBBB",
        )
        assert len(response.choices) == 1
        assert response.choices[0]["message"].content == "Paris"

        # check request
        post_kwargs = mock_post.call_args_list[-1][1]
        body = json.loads(post_kwargs["data"])
        assert body["model"] == self.model_name
        assert "the capital of France" in str(body["messages"])

        # JWT key was used
        assert "00000" in post_kwargs["headers"]["Authorization"]
        # account id was used
        assert "AAAA-BBBB" in post_kwargs["url"]
        # is completion
        assert post_kwargs["url"].endswith("cortex/inference:complete")

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_snowflake_pat_key_account_id(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response)

        response = litellm.completion(
            f"snowflake/{self.model_name}",
            messages=self.messages,
            api_key="pat/xxxxx",
            account_id="AAAA-BBBB",
        )
        assert len(response.choices) == 1
        assert response.choices[0]["message"].content == "Paris"

        # PAT key was used
        post_kwargs = mock_post.call_args_list[-1][1]
        assert "xxxxx" in post_kwargs["headers"]["Authorization"]
        assert (
            post_kwargs["headers"]["X-Snowflake-Authorization-Token-Type"]
            == "PROGRAMMATIC_ACCESS_TOKEN"
        )

        # account id was used
        assert "AAAA-BBBB" in post_kwargs["url"]

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_snowflake_env(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response)

        os.environ["SNOWFLAKE_ACCOUNT_ID"] = "AAAA-BBBB"
        os.environ["SNOWFLAKE_JWT"] = "00000"

        response = litellm.completion(
            f"snowflake/{self.model_name}",
            messages=self.messages,
        )

        assert len(response.choices) == 1
        assert response.choices[0]["message"].content == "Paris"

        # JWT key was used
        post_kwargs = mock_post.call_args_list[-1][1]
        assert "00000" in post_kwargs["headers"]["Authorization"]
        # account id was used
        assert "AAAA-BBBB" in post_kwargs["url"]

        os.environ.pop("SNOWFLAKE_ACCOUNT_ID", None)
        os.environ.pop("SNOWFLAKE_JWT", None)
