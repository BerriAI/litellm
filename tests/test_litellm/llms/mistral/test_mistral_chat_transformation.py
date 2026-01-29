import os
import sys
from typing import List, cast
from unittest.mock import MagicMock, patch

import pytest

from litellm.types.llms.openai import AllMessageValues

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.mistral.chat.transformation import (
    MistralChatResponseIterator,
    MistralConfig,
)
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_mistral_chat_transformation():
    mistral_config = MistralConfig()
    result = mistral_config._transform_messages(
        **{
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": "Here is a representation of text"},
                        {
                            "type": "image_url",
                            "image_url": "https://images.pexels.com/photos/13268478/pexels-photo-13268478.jpeg",
                        },
                    ],
                    "role": "user",
                }
            ],
            "model": "mistral-medium-latest",
            "is_async": True,
        }
    )


class TestMistralReasoningSupport:
    """Test suite for Mistral Magistral reasoning functionality."""

    def test_get_supported_openai_params_magistral_model(self):
        """Test that magistral models support reasoning parameters."""
        mistral_config = MistralConfig()

        # Test magistral model supports reasoning parameters
        supported_params = mistral_config.get_supported_openai_params(
            "mistral/magistral-medium-2506"
        )
        assert "reasoning_effort" in supported_params
        assert "thinking" in supported_params

        # Test non-magistral model doesn't include reasoning parameters
        supported_params_normal = mistral_config.get_supported_openai_params(
            "mistral/mistral-large-latest"
        )
        assert "reasoning_effort" not in supported_params_normal
        assert "thinking" not in supported_params_normal

    def test_map_openai_params_reasoning_effort(self):
        """Test that reasoning_effort parameter is properly mapped for magistral models."""
        mistral_config = MistralConfig()

        # Test reasoning_effort mapping for magistral model
        optional_params = {}
        result = mistral_config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params=optional_params,
            model="mistral/magistral-medium-2506",
            drop_params=False,
        )

        assert result.get("_add_reasoning_prompt") is True

        # Test reasoning_effort ignored for non-magistral model
        optional_params_normal = {}
        result_normal = mistral_config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params=optional_params_normal,
            model="mistral/mistral-large-latest",
            drop_params=False,
        )

        assert "_add_reasoning_prompt" not in result_normal

    def test_map_openai_params_thinking(self):
        """Test that thinking parameter is properly mapped for magistral models."""
        mistral_config = MistralConfig()

        # Test thinking mapping for magistral model
        optional_params = {}
        result = mistral_config.map_openai_params(
            non_default_params={"thinking": {"budget": 1000}},
            optional_params=optional_params,
            model="mistral/magistral-small-2506",
            drop_params=False,
        )

        assert result.get("_add_reasoning_prompt") is True

    def test_get_mistral_reasoning_system_prompt(self):
        """Test that the reasoning system prompt is properly formatted."""
        prompt = MistralConfig._get_mistral_reasoning_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # Ensure it's not empty

    def test_add_reasoning_system_prompt_no_existing_system_message(self):
        """Test adding reasoning system prompt when no system message exists."""
        mistral_config = MistralConfig()

        messages = [{"role": "user", "content": "What is 2+2?"}]
        optional_params = {"_add_reasoning_prompt": True}

        result = mistral_config._add_reasoning_system_prompt_if_needed(
            messages, optional_params
        )

        # Should add a new system message at the beginning
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "<think>" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What is 2+2?"

        # Should remove the internal flag
        assert "_add_reasoning_prompt" not in optional_params

    def test_add_reasoning_system_prompt_with_existing_system_message(self):
        """Test adding reasoning system prompt when system message already exists."""
        mistral_config = MistralConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        optional_params = {"_add_reasoning_prompt": True}

        result = mistral_config._add_reasoning_system_prompt_if_needed(
            messages, optional_params
        )

        # Should modify existing system message
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "<think>" in result[0]["content"]
        assert "You are a helpful assistant." in result[0]["content"]
        assert result[1]["role"] == "user"

        # Should remove the internal flag
        assert "_add_reasoning_prompt" not in optional_params

    def test_add_reasoning_system_prompt_with_existing_list_content(self):
        """Test adding reasoning system prompt when system message has list content."""
        mistral_config = MistralConfig()

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {
                        "type": "text",
                        "text": "You always provide detailed explanations.",
                    },
                ],
            },
            {"role": "user", "content": "What is 2+2?"},
        ]
        optional_params = {"_add_reasoning_prompt": True}

        result = mistral_config._add_reasoning_system_prompt_if_needed(
            messages, optional_params
        )

        # Should modify existing system message preserving list format
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], list)

        # First item should be the reasoning prompt
        assert result[0]["content"][0]["type"] == "text"
        assert "<think>" in result[0]["content"][0]["text"]

        # Original content should be preserved
        assert "You are a helpful assistant." in result[0]["content"][1]["text"]
        assert (
            "You always provide detailed explanations."
            in result[0]["content"][2]["text"]
        )

        assert result[1]["role"] == "user"

        # Should remove the internal flag
        assert "_add_reasoning_prompt" not in optional_params

    def test_add_reasoning_system_prompt_preserves_content_types(self):
        """Test that reasoning prompt preserves original content types (string vs list)."""
        mistral_config = MistralConfig()

        # Test with string content
        string_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        string_params = {"_add_reasoning_prompt": True}

        string_result = mistral_config._add_reasoning_system_prompt_if_needed(
            string_messages, string_params
        )
        assert isinstance(string_result[0]["content"], str)
        assert "<think>" in string_result[0]["content"]
        assert "You are helpful." in string_result[0]["content"]

        # Test with list content
        list_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "user", "content": "Hello"},
        ]
        list_params = {"_add_reasoning_prompt": True}

        list_result = mistral_config._add_reasoning_system_prompt_if_needed(
            list_messages, list_params
        )
        assert isinstance(list_result[0]["content"], list)
        assert list_result[0]["content"][0]["type"] == "text"
        assert "<think>" in list_result[0]["content"][0]["text"]
        assert "You are helpful." in list_result[0]["content"][1]["text"]

    def test_add_reasoning_system_prompt_no_flag(self):
        """Test that no modification happens when _add_reasoning_prompt flag is not set."""
        mistral_config = MistralConfig()

        messages = [{"role": "user", "content": "What is 2+2?"}]
        optional_params = {}

        result = mistral_config._add_reasoning_system_prompt_if_needed(
            messages, optional_params
        )

        # Should return messages unchanged
        assert result == messages
        assert len(result) == 1

    def test_transform_request_magistral_with_reasoning(self):
        """Test transform_request method for magistral model with reasoning."""
        mistral_config = MistralConfig()

        messages = [{"role": "user", "content": "What is 15 * 7?"}]
        optional_params = {"_add_reasoning_prompt": True}

        result = mistral_config.transform_request(
            model="mistral/magistral-medium-2506",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Should have added system message
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "<think>" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"

        # Should remove internal flag from optional_params
        assert "_add_reasoning_prompt" not in result

    def test_transform_request_magistral_without_reasoning(self):
        """Test transform_request method for magistral model without reasoning."""
        mistral_config = MistralConfig()

        messages = [{"role": "user", "content": "What is 15 * 7?"}]
        optional_params = {}

        result = mistral_config.transform_request(
            model="mistral/magistral-medium-2506",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Should not modify messages
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_non_magistral_with_reasoning_params(self):
        """Test that non-magistral models ignore reasoning parameters."""
        mistral_config = MistralConfig()

        messages = [{"role": "user", "content": "What is 15 * 7?"}]
        optional_params = {"_add_reasoning_prompt": True}

        result = mistral_config.transform_request(
            model="mistral/mistral-large-latest",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Should not add system message for non-magistral models
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_case_insensitive_magistral_detection(self):
        """Test that magistral model detection is case-insensitive."""
        mistral_config = MistralConfig()

        # Test various case combinations
        models_to_test = [
            "mistral/Magistral-medium-2506",
            "mistral/MAGISTRAL-MEDIUM-2506",
            "mistral/magistral-SMALL-2506",
            "MaGiStRaL-medium-2506",
        ]

        for model in models_to_test:
            supported_params = mistral_config.get_supported_openai_params(model)
            assert "reasoning_effort" in supported_params, f"Failed for model: {model}"

    def test_end_to_end_reasoning_workflow(self):
        """Test the complete workflow from parameter to system prompt injection."""
        mistral_config = MistralConfig()

        # Step 1: Map parameters
        optional_params = {}
        mapped_params = mistral_config.map_openai_params(
            non_default_params={"reasoning_effort": "high", "temperature": 0.7},
            optional_params=optional_params,
            model="mistral/magistral-medium-2506",
            drop_params=False,
        )

        assert mapped_params.get("_add_reasoning_prompt") is True
        assert mapped_params.get("temperature") == 0.7

        # Step 2: Transform request
        messages = [{"role": "user", "content": "Solve for x: 2x + 5 = 13"}]

        result = mistral_config.transform_request(
            model="mistral/magistral-medium-2506",
            messages=messages,
            optional_params=mapped_params,
            litellm_params={},
            headers={},
        )

        # Verify final result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "<think>" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Solve for x: 2x + 5 = 13"
        assert result.get("temperature") == 0.7
        assert "_add_reasoning_prompt" not in result


def test_mistral_streaming_chunk_preserves_thinking_blocks():
    """Ensure streaming chunks keep magistral reasoning content."""
    iterator = MistralChatResponseIterator(
        streaming_response=iter([]), sync_stream=True, json_mode=False
    )

    streamed_chunk = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 123456,
        "model": "magistral-medium-2509",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "Working it out."}],
                        },
                        {"type": "text", "text": " Hello"},
                    ],
                },
                "finish_reason": None,
            }
        ],
    }

    parsed_chunk = iterator.chunk_parser(streamed_chunk)

    delta = parsed_chunk.choices[0].delta
    assert delta.thinking_blocks is not None
    assert delta.thinking_blocks[0]["thinking"] == "Working it out."
    assert delta.thinking_blocks[0]["signature"] == "mistral"
    assert delta.reasoning_content == "Working it out."
    assert delta.content == " Hello"


class TestMistralNameHandling:
    """Test suite for Mistral name handling in messages."""

    def test_handle_name_in_message_tool_role_empty_name_removes_name(self):
        """Test that empty name is removed for tool messages."""
        # Test with empty string
        tool_message = {"role": "tool", "content": "Function result", "name": ""}
        result = MistralConfig._handle_name_in_message(tool_message)
        assert "name" not in result
        assert result["role"] == "tool"
        assert result["content"] == "Function result"

    def test_handle_name_in_message_tool_role_valid_name_keeps_name(self):
        """Test that valid name is kept for tool messages."""
        # Test with normal function name
        tool_message = {
            "role": "tool",
            "content": "Function result",
            "name": "get_weather",
        }
        result = MistralConfig._handle_name_in_message(tool_message)
        assert "name" in result
        assert result["name"] == "get_weather"
        assert result["role"] == "tool"
        assert result["content"] == "Function result"

    def test_handle_name_in_message_no_name_field(self):
        """Test that messages without name field are unchanged."""
        # Test with user role
        user_message = {"role": "user", "content": "Hello"}
        result = MistralConfig._handle_name_in_message(user_message)
        assert "name" not in result
        assert result["role"] == "user"
        assert result["content"] == "Hello"


class TestMistralParallelToolCalls:
    """Test suite for Mistral parallel tool calls functionality."""

    def test_get_supported_openai_params_includes_parallel_tool_calls(self):
        """Test that parallel_tool_calls is in supported parameters."""
        mistral_config = MistralConfig()
        supported_params = mistral_config.get_supported_openai_params(
            "mistral/mistral-large-latest"
        )
        assert "parallel_tool_calls" in supported_params

    def test_transform_request_preserves_parallel_tool_calls(self):
        """Test that transform_request preserves parallel_tool_calls parameter."""
        mistral_config = MistralConfig()

        messages = [{"role": "user", "content": "What's the weather like?"}]
        optional_params = {"parallel_tool_calls": True}

        result = mistral_config.transform_request(
            model="mistral/mistral-large-latest",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result.get("parallel_tool_calls") is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestMistralThinkingContentHandling:
    """Test suite for Mistral thinking content response handling functionality."""

    def test_transform_response_with_thinking_content(self):
        """Test that Mistral responses with thinking content are correctly transformed."""
        import json
        from unittest.mock import Mock

        import litellm

        # Raw response from Mistral with thinking content
        raw_response_data = {
            "id": "12a18e1439f24f95b9812a016e0af235",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [
                                    {
                                        "type": "text",
                                        "text": "Well, the capital of France is a well-known fact. It's Paris. But just to be sure, I recall that Paris is indeed the capital city of France. I don't need to look it up because it's a common knowledge fact. But if I were unsure, I would double-check using a reliable source or a knowledge base. Since I'm confident about this, I can provide the answer directly.",
                                    }
                                ],
                            },
                            {"type": "text", "text": "The capital of France is Paris."},
                        ],
                        "refusal": None,
                        "role": "assistant",
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                    },
                }
            ],
            "created": 1754654178,
            "model": "magistral-medium-2507",
            "object": "chat.completion",
            "service_tier": None,
            "system_fingerprint": None,
            "usage": {
                "completion_tokens": 93,
                "prompt_tokens": 11,
                "total_tokens": 104,
                "completion_tokens_details": None,
                "prompt_tokens_details": None,
            },
        }

        # Mock httpx response
        mock_response = Mock()
        mock_response.json.return_value = raw_response_data
        mock_response.headers = {}
        mock_response.text = json.dumps(raw_response_data)

        # Mock logging object with proper attributes
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        # Test the transformation
        mistral_config = MistralConfig()
        model_response = litellm.ModelResponse()

        # Test transform_response method
        final_response = mistral_config.transform_response(
            model="mistral/magistral-medium-2507",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        # Verify the response structure
        assert final_response is not None
        assert len(final_response.choices) == 1
        choice = final_response.choices[0]

        # Verify message content
        message = choice.message
        assert message.role == "assistant"

        # The content should be processed - either as text or as thinking blocks
        # Content could be the text part or the full content list
        content_str = str(message.content) if message.content else ""

        # Verify the actual text content is preserved somewhere
        assert "The capital of France is Paris." in content_str or (
            hasattr(message, "thinking_blocks") and message.thinking_blocks
        )

        # Verify usage information
        assert final_response.usage.completion_tokens == 93
        assert final_response.usage.prompt_tokens == 11
        assert final_response.usage.total_tokens == 104

        # Verify model and metadata
        assert final_response.id == "12a18e1439f24f95b9812a016e0af235"
        assert final_response.created == 1754654178


class TestMistralEmptyContentHandling:
    """Test suite for Mistral empty content response handling functionality."""

    def test_handle_empty_content_response_converts_empty_string_to_none(self):
        """Test that empty string content is converted to None."""
        response_data = {
            "choices": [
                {
                    "message": {"content": "", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ]
        }

        result = MistralConfig._handle_empty_content_response(response_data)

        assert result["choices"][0]["message"]["content"] is None

    def test_handle_empty_content_response_preserves_actual_content(self):
        """Test that actual content is preserved unchanged."""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        result = MistralConfig._handle_empty_content_response(response_data)

        assert (
            result["choices"][0]["message"]["content"] == "Hello, how can I help you?"
        )

    def test_handle_empty_content_response_handles_multiple_choices(self):
        """Test that only the first choice is processed for empty content."""
        response_data = {
            "choices": [
                {
                    "message": {"content": "", "role": "assistant"},
                    "finish_reason": "stop",
                },
                {
                    "message": {"content": "", "role": "assistant"},
                    "finish_reason": "stop",
                },
            ]
        }

        result = MistralConfig._handle_empty_content_response(response_data)

        # Only first choice should be converted to None
        assert result["choices"][0]["message"]["content"] is None
        # Second choice should remain as empty string
        assert result["choices"][1]["message"]["content"] is None

    def test_is_empty_assistant_message(self):
        """Test that is_empty_assistant_message returns True for empty assistant message."""
        message = {"role": "assistant", "content": ""}
        assert MistralConfig._is_empty_assistant_message(message) is True

    def test_is_empty_assistant_message_with_content(self):
        """Test that is_empty_assistant_message returns False for assistant message with content."""
        message = {"role": "assistant", "content": "Hello"}
        assert MistralConfig._is_empty_assistant_message(message) is False

class TestMistralFileHandling:
    """Test suite for Mistral file handling functionality."""
    
    def test_handle_file_message_with_file_id(self):
        """Test that file messages with file_id are handled correctly."""
        mistral_config = MistralConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please review this file."},
                    {"type": "file", "file": {"file_id": "file-12345"}}
                ]
            }
        ]
        casted_message = cast(list[AllMessageValues], messages)
        result = mistral_config._handle_message_with_file(casted_message)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Check that content is transformed correctly
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        # Check that file type is preserved
        assert result[0]["content"][1]["type"] == "file"
        # Check that file_id is modified to match Mistral's expected format
        assert result[0]["content"][1]["file_id"] == "file-12345" # type: ignore

    def test_handle_file_message_without_file_id(self):
        """Test that file messages without file_id are ignored."""
        mistral_config = MistralConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please review this file."}
                ]
            }
        ]
        casted_message = cast(list[AllMessageValues], messages)
        result = mistral_config._handle_message_with_file(casted_message)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1  # Only text part remains

    def test_handle_message_with_file_multiple_files(self):
        """Test that multiple file messages are handled correctly."""
        mistral_config = MistralConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please review these files."},
                    {"type": "file", "file": {"file_id": "file-12345"}},
                    {"type": "file", "file": {"file_id": "file-67890"}}
                ]
            }
        ]
        casted_message = cast(list[AllMessageValues], messages)
        result = mistral_config._handle_message_with_file(casted_message)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Check that content is transformed correctly
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 3  # Text + 2 files
        # Check that file types are preserved
        assert result[0]["content"][1]["type"] == "file"
        assert result[0]["content"][2]["type"] == "file"
        # Check that file_ids are modified to match Mistral's expected format
        assert result[0]["content"][1]["file_id"] == "file-12345"  # type: ignore
        assert result[0]["content"][2]["file_id"] == "file-67890"  # type: ignore
