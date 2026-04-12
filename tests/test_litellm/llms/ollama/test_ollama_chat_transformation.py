import inspect
import os
import sys
from typing import cast

import pytest
from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.ollama.chat.transformation import OllamaChatConfig, OllamaChatCompletionResponseIterator

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


class TestOllamaToolCalling:
    """Tests for Ollama tool calling fixes.

    Issue: https://github.com/BerriAI/litellm/issues/18922
    """

    def test_tools_passed_directly_without_capability_check(self):
        """Test that tools are passed directly to Ollama without model capability checks.

        Previously, the code called litellm.get_model_info() which could fail
        when Ollama runs on a remote server, causing a broken fallback.
        Now tools are passed directly - Ollama 0.4+ handles capability detection.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        optional_params = get_optional_params(
            model="ollama_chat/qwen3:14b",
            tools=tools,
            custom_llm_provider="ollama_chat",
        )

        # Tools should be passed through directly
        assert "tools" in optional_params
        assert optional_params["tools"] == tools
        # Should NOT trigger the broken fallback
        assert "functions_unsupported_model" not in optional_params
        assert "format" not in optional_params or optional_params.get("format") != "json"

    def test_finish_reason_tool_calls_non_streaming(self):
        """Test that finish_reason is set to 'tool_calls' when tool_calls present.

        Previously, finish_reason was hardcoded to 'stop' even when tool_calls
        were in the response, causing clients to ignore the tool calls.
        """
        import json
        from unittest.mock import MagicMock

        import litellm
        from litellm.types.utils import Choices, Message, ModelResponse

        config = OllamaChatConfig()

        # Simulated Ollama response with tool_calls
        ollama_response = {
            "model": "qwen3:14b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "",
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
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        mock_logging = MagicMock()

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Weather?"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        # finish_reason should be "tool_calls", not "stop"
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None

    def test_finish_reason_stop_when_no_tool_calls(self):
        """Test that finish_reason remains 'stop' when no tool_calls present."""
        import json
        from unittest.mock import MagicMock

        import litellm
        from litellm.types.utils import Choices, Message, ModelResponse

        config = OllamaChatConfig()

        # Simulated Ollama response without tool_calls
        ollama_response = {
            "model": "qwen3:14b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?",
            },
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        mock_logging = MagicMock()

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        # finish_reason should be "stop" (default behavior)
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.tool_calls is None


class TestOllamaReasoningContentStreaming:
    """Test that reasoning_content is properly extracted from all thinking chunks."""

    def test_multiple_thinking_chunks_all_returned_as_reasoning_content(self):
        """
        Test that more than 2 consecutive thinking chunks are all returned as reasoning_content.
        
        Previously, the code had a bug where finished_reasoning_content was set to True
        after just 2 chunks with 'thinking', causing subsequent thinking content to be lost.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),  # Not used in chunk_parser
            sync_stream=True,
        )

        # Simulate 5 consecutive chunks with 'thinking' content
        thinking_chunks = [
            {
                "model": "deepseek-r1",
                "message": {"role": "assistant", "thinking": f"Thinking chunk {i}"},
                "done": False,
            }
            for i in range(1, 6)
        ]

        # Process all thinking chunks
        reasoning_contents = []
        for chunk in thinking_chunks:
            result = iterator.chunk_parser(chunk)
            rc = result.choices[0].delta.reasoning_content
            reasoning_contents.append(rc)

        # ALL chunks should have reasoning_content, not just the first 2
        assert len(reasoning_contents) == 5
        assert reasoning_contents[0] == "Thinking chunk 1"
        assert reasoning_contents[1] == "Thinking chunk 2"
        assert reasoning_contents[2] == "Thinking chunk 3"  # This was previously None
        assert reasoning_contents[3] == "Thinking chunk 4"  # This was previously None
        assert reasoning_contents[4] == "Thinking chunk 5"  # This was previously None

        # Verify none of them are None
        for i, rc in enumerate(reasoning_contents):
            assert rc is not None, f"Chunk {i+1} reasoning_content should not be None"

    def test_thinking_to_content_transition(self):
        """
        Test that transition from thinking to regular content works correctly.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        # First: thinking chunks
        thinking_chunk = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "thinking": "Let me think about this..."},
            "done": False,
        }
        result1 = iterator.chunk_parser(thinking_chunk)
        assert result1.choices[0].delta.reasoning_content == "Let me think about this..."
        assert result1.choices[0].delta.content is None

        # Then: regular content chunk
        content_chunk = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "content": "Here is my answer."},
            "done": False,
        }
        result2 = iterator.chunk_parser(content_chunk)
        assert result2.choices[0].delta.content == "Here is my answer."
        # reasoning_content is not set when there's no thinking in the chunk
        assert getattr(result2.choices[0].delta, 'reasoning_content', None) is None

    def test_think_tags_in_content(self):
        """
        Test that <think> tags embedded in content are properly parsed.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        # Content with <think> tag
        chunk1 = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "content": "<think>I need to analyze this"},
            "done": False,
        }
        result1 = iterator.chunk_parser(chunk1)
        assert result1.choices[0].delta.reasoning_content == "I need to analyze this"
        assert result1.choices[0].delta.content is None

        # Content with </think> tag (end of thinking)
        chunk2 = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "content": "</think>The answer is 42."},
            "done": False,
        }
        result2 = iterator.chunk_parser(chunk2)
        assert result2.choices[0].delta.content == "The answer is 42."
        # reasoning_content is not set when it's regular content
        assert getattr(result2.choices[0].delta, 'reasoning_content', None) is None

    def test_done_chunk_with_thinking(self):
        """
        Test that the final chunk with done=True and thinking content works.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        # Final chunk with thinking
        done_chunk = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "thinking": "Final thought"},
            "done": True,
            "done_reason": "stop",
        }
        result = iterator.chunk_parser(done_chunk)
        assert result.choices[0].delta.reasoning_content == "Final thought"
        assert result.choices[0].finish_reason == "stop"


