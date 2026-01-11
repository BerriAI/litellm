import inspect
import os
import sys
from typing import cast

import pytest
from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import get_optional_params


class TestEvent(BaseModel):
    name: str
    value: int


class TestOllamaChatConfigResponseFormat:
    def test_get_optional_params_with_pydantic_model(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=TestEvent,
            custom_llm_provider="ollama_chat",
        )
        print(f"optional_params: {optional_params}")

        assert "format" in optional_params
        transformed_format = optional_params["format"]

        expected_schema_structure = TestEvent.model_json_schema()
        transformed_format.pop("additionalProperties")

        assert (
            transformed_format == expected_schema_structure
        ), f"Transformed schema does not match expected. Got: {transformed_format}, Expected: {expected_schema_structure}"

    def test_map_openai_params_with_dict_json_schema(self):
        config = OllamaChatConfig()

        direct_schema = TestEvent.model_json_schema()
        response_format_dict = {
            "type": "json_schema",
            "json_schema": {"schema": direct_schema},
        }

        non_default_params = {"response_format": response_format_dict}

        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=response_format_dict,
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert (
            optional_params["format"] == direct_schema
        ), f"Schema from dict did not pass through correctly. Got: {optional_params['format']}, Expected: {direct_schema}"

    def test_map_openai_params_with_json_object(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format={"type": "json_object"},
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert (
            optional_params["format"] == "json"
        ), f"Expected 'json' for type 'json_object', got: {optional_params['format']}"

    def test_transform_request_loads_config_parameters(self):
        """Test that transform_request loads config parameters without overriding existing optional_params"""
        # Set config parameters on the class
        import litellm

        litellm.OllamaChatConfig(num_ctx=8000, temperature=0.0)

        try:
            config = OllamaChatConfig()

            # Initial optional_params with existing temperature (should not be overridden)
            optional_params = {"temperature": 0.3}

            # Transform request
            result = config.transform_request(
                model="llama2",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params=optional_params,
                litellm_params={},
                headers={},
            )

            # Verify config values were loaded but existing optional_params were preserved
            assert result["options"]["temperature"] == 0.3  # Should keep existing value
            assert result["options"]["num_ctx"] == 8000  # Should load from config

        finally:
            # Clean up class attributes
            delattr(litellm.OllamaChatConfig, "num_ctx")
            delattr(litellm.OllamaChatConfig, "temperature")

    def test_transform_request_content_list_to_string(self):
        """Test that content list is properly converted to string in transform_request"""
        config = OllamaChatConfig()

        # Test message with content as list containing text
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world!"},
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify content was converted to string
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello world!"
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_content_string_passthrough(self):
        """Test that string content passes through unchanged in transform_request"""
        config = OllamaChatConfig()

        # Test message with content as string
        messages = cast(
            list[AllMessageValues], [{"role": "user", "content": "Hello world!"}]
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify string content passes through
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello world!"
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_empty_content_list(self):
        """Test handling of empty content list in transform_request"""
        config = OllamaChatConfig()

        # Test message with empty content list
        messages = cast(list[AllMessageValues], [{"role": "user", "content": []}])

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify empty content becomes empty string
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == ""
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_image_extraction(self):
        """Test that images are properly extracted from messages in transform_request"""
        config = OllamaChatConfig()

        # Test message with images in content list
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
                            },
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify text content was extracted
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "What's in this image?"
        assert result["messages"][0]["role"] == "user"

        # Verify image was extracted to images list
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 1
        # Ollama expects pure base64 data without the data URL prefix
        assert result["messages"][0]["images"][0] == "/9j/4AAQSkZJRgABAQAAAQ..."

    def test_transform_request_multiple_images_extraction(self):
        """Test extraction of multiple images from a single message"""
        config = OllamaChatConfig()

        # Test message with multiple images
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these images:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,image1data..."
                            },
                        },
                        {"type": "text", "text": " and "},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,image2data..."},
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify text content was combined
        assert result["messages"][0]["content"] == "Compare these images: and "

        # Verify both images were extracted
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 2
        # Ollama expects pure base64 data without the data URL prefix
        assert result["messages"][0]["images"][0] == "image1data..."
        assert result["messages"][0]["images"][1] == "image2data..."

    def test_transform_request_image_url_as_string(self):
        """Test handling of image_url as direct string (edge case)"""
        config = OllamaChatConfig()

        # Test message with image_url as string (edge case from extract_images_from_message)
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this:"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify image URL was extracted
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 1
        assert result["messages"][0]["images"][0] == "https://example.com/image.jpg"

    def test_transform_request_no_images_no_images_key(self):
        """Test that messages without images don't have images key"""
        config = OllamaChatConfig()

        # Test message with no images
        messages = cast(
            list[AllMessageValues],
            [{"role": "user", "content": [{"type": "text", "text": "Just text here"}]}],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify no images key when no images present
        assert result["messages"][0]["content"] == "Just text here"
        # Since extract_images_from_message returns empty list [] when no images found,
        # and the code checks "if images is not None", an empty list will still be set
        assert "images" in result["messages"][0]
        assert result["messages"][0]["images"] == []


class TestOllamaChatTransformResponse:
    """Tests for transform_response method, especially for qwen3 tool_calls handling.

    Issue: https://github.com/BerriAI/litellm/issues/18922
    Qwen3 includes a 'thinking' field in responses that was causing tool_calls to be dropped.
    """

    def _create_mock_response(self, json_data: dict):
        """Create a mock httpx Response object."""
        import json
        from unittest.mock import MagicMock, PropertyMock

        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.text = json.dumps(json_data)
        return mock_response

    def _create_mock_logging_obj(self):
        """Create a mock logging object."""
        from unittest.mock import MagicMock

        mock_logging = MagicMock()
        mock_logging.post_call = MagicMock()
        return mock_logging

    def _create_model_response(self):
        """Create a base ModelResponse object."""
        import litellm
        from litellm.types.utils import Choices, Message, ModelResponse

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]
        return model_response

    def test_transform_response_with_qwen3_thinking_and_tool_calls(self):
        """Test that qwen3 responses with 'thinking' field preserve tool_calls.

        This is the core bug fix test - qwen3 returns both 'thinking' and 'tool_calls'
        and the tool_calls were being dropped.
        """
        import json

        config = OllamaChatConfig()

        # Simulated qwen3 response with thinking and tool_calls
        ollama_response = {
            "model": "qwen3:14b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Let me analyze this request and call the appropriate function...",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo", "units": "celsius"},
                        }
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        mock_response = self._create_mock_response(ollama_response)
        mock_logging = self._create_mock_logging_obj()
        model_response = self._create_model_response()

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            optional_params={},
            litellm_params={"api_base": "http://localhost:11434"},
            encoding="utf-8",
        )

        # Verify tool_calls are preserved
        assert result.choices[0].message.tool_calls is not None, (
            "tool_calls should not be None - this is the core qwen3 bug!"
        )
        assert len(result.choices[0].message.tool_calls) == 1

        # Verify tool_call structure matches OpenAI format
        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"

        # Verify arguments are stringified (OpenAI expects JSON string, not dict)
        assert isinstance(tool_call.function.arguments, str)
        parsed_args = json.loads(tool_call.function.arguments)
        assert parsed_args["location"] == "Tokyo"
        assert parsed_args["units"] == "celsius"

        # Verify id and type are set (auto-generated for Ollama responses)
        assert tool_call.id is not None
        assert tool_call.type == "function"

        # Verify thinking was remapped to reasoning_content
        assert result.choices[0].message.reasoning_content is not None
        assert "analyze this request" in result.choices[0].message.reasoning_content

        # Verify finish_reason is set to "tool_calls" (critical for clients to process tool calls)
        assert result.choices[0].finish_reason == "tool_calls", (
            "finish_reason should be 'tool_calls' when tool_calls are present!"
        )

    def test_transform_response_with_tool_calls_no_thinking(self):
        """Test that tool_calls work without thinking field (standard Ollama models)."""
        import json

        config = OllamaChatConfig()

        # Standard Ollama response with tool_calls but no thinking
        ollama_response = {
            "model": "llama3.1:8b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_database",
                            "arguments": {"query": "test"},
                        }
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 50,
            "eval_count": 25,
        }

        mock_response = self._create_mock_response(ollama_response)
        mock_logging = self._create_mock_logging_obj()
        model_response = self._create_model_response()

        result = config.transform_response(
            model="llama3.1:8b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Search for test"}],
            optional_params={},
            litellm_params={"api_base": "http://localhost:11434"},
            encoding="utf-8",
        )

        # Verify tool_calls are preserved
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].function.name == "search_database"

        # Verify finish_reason is "tool_calls"
        assert result.choices[0].finish_reason == "tool_calls"

    def test_transform_response_multiple_tool_calls(self):
        """Test handling of multiple tool_calls in a single response."""
        import json

        config = OllamaChatConfig()

        ollama_response = {
            "model": "qwen3:14b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "I need to get weather for both cities...",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo"},
                        }
                    },
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "New York"},
                        }
                    },
                ],
            },
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 75,
        }

        mock_response = self._create_mock_response(ollama_response)
        mock_logging = self._create_mock_logging_obj()
        model_response = self._create_model_response()

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Weather in Tokyo and New York?"}],
            optional_params={},
            litellm_params={"api_base": "http://localhost:11434"},
            encoding="utf-8",
        )

        # Verify both tool_calls are preserved
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 2

        # Verify each tool_call
        tool_call_1 = result.choices[0].message.tool_calls[0]
        tool_call_2 = result.choices[0].message.tool_calls[1]

        assert tool_call_1.function.name == "get_weather"
        assert json.loads(tool_call_1.function.arguments)["location"] == "Tokyo"

        assert tool_call_2.function.name == "get_weather"
        assert json.loads(tool_call_2.function.arguments)["location"] == "New York"

        # Each tool_call should have unique id
        assert tool_call_1.id != tool_call_2.id

        # Verify finish_reason is "tool_calls"
        assert result.choices[0].finish_reason == "tool_calls"

    def test_transform_response_content_with_tool_calls(self):
        """Test that content and tool_calls can coexist."""
        config = OllamaChatConfig()

        ollama_response = {
            "model": "qwen3:14b",
            "message": {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "thinking": "User wants weather info...",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo"},
                        }
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 50,
            "eval_count": 30,
        }

        mock_response = self._create_mock_response(ollama_response)
        mock_logging = self._create_mock_logging_obj()
        model_response = self._create_model_response()

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Weather?"}],
            optional_params={},
            litellm_params={"api_base": "http://localhost:11434"},
            encoding="utf-8",
        )

        # Both content and tool_calls should be present
        assert result.choices[0].message.content == "I'll check the weather for you."
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1

        # Verify finish_reason is "tool_calls"
        assert result.choices[0].finish_reason == "tool_calls"

    def test_transform_response_empty_content_string(self):
        """Test that empty content string with tool_calls works correctly."""
        config = OllamaChatConfig()

        # This is what qwen3 often returns - empty content with tool_calls
        ollama_response = {
            "model": "qwen3:14b",
            "message": {
                "role": "assistant",
                "content": "",  # Empty string, not None
                "thinking": "Calling the function...",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": {"expression": "2+2"},
                        }
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 30,
            "eval_count": 20,
        }

        mock_response = self._create_mock_response(ollama_response)
        mock_logging = self._create_mock_logging_obj()
        model_response = self._create_model_response()

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Calculate 2+2"}],
            optional_params={},
            litellm_params={"api_base": "http://localhost:11434"},
            encoding="utf-8",
        )

        # Content should be empty string (or None), but tool_calls should be present
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].function.name == "calculate"

        # Verify finish_reason is "tool_calls"
        assert result.choices[0].finish_reason == "tool_calls"
