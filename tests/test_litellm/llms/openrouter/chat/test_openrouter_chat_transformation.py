import os
import sys

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openrouter.chat.transformation import (
    OpenRouterChatCompletionStreamingHandler,
    OpenrouterConfig,
    OpenRouterException,
)


class TestOpenRouterChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = OpenRouterChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test input chunk
        chunk = {
            "id": "test_id",
            "created": 1234567890,
            "model": "test_model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [
                {"delta": {"content": "test content", "reasoning": "test reasoning"}}
            ],
        }

        # Parse chunk
        result = handler.chunk_parser(chunk)

        # Verify response
        assert result.id == "test_id"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "test_model"
        assert result.usage.prompt_tokens == chunk["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == chunk["usage"]["completion_tokens"]
        assert result.usage.total_tokens == chunk["usage"]["total_tokens"]
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["reasoning_content"] == "test reasoning"

    def test_chunk_parser_error_response(self):
        handler = OpenRouterChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test error chunk
        error_chunk = {
            "error": {
                "message": "test error",
                "code": 400,
                "metadata": {"key": "value"},
                "user_id": "test_user",
            }
        }

        # Verify error handling
        with pytest.raises(OpenRouterException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert "Message: test error" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_chunk_parser_key_error(self):
        handler = OpenRouterChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test invalid chunk missing required fields
        invalid_chunk = {"incomplete": "data"}

        # Verify KeyError handling
        with pytest.raises(OpenRouterException) as exc_info:
            handler.chunk_parser(invalid_chunk)

        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


def test_openrouter_extra_body_transformation():
    transformed_request = OpenrouterConfig().transform_request(
        model="openrouter/deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "Hello, world!"}],
        optional_params={"extra_body": {"provider": {"order": ["DeepSeek"]}}},
        litellm_params={},
        headers={},
    )

    # https://github.com/BerriAI/litellm/issues/8425, validate its not contained in extra_body still
    assert transformed_request["provider"]["order"] == ["DeepSeek"]
    assert transformed_request["messages"] == [
        {"role": "user", "content": "Hello, world!"}
    ]


def test_openrouter_cache_control_flag_removal():
    transformed_request = OpenrouterConfig().transform_request(
        model="openrouter/deepseek/deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "Hello, world!",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert transformed_request["messages"][0].get("cache_control") is None



def test_openrouter_transform_request_with_cache_control():
    """
    Test transform_request moves cache_control from message level to content blocks (string content).
    
    Input:
    {
        "role": "user",
        "content": "what are the key terms...",
        "cache_control": {"type": "ephemeral"}
    }
    
    Expected Output:
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "what are the key terms...",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
    """
    import json
    config = OpenrouterConfig()
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing legal documents."
                },
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement"
                }
            ]
        },
        {
            "role": "user",
            "content": "what are the key terms and conditions in this agreement?",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-3-5-sonnet-20240620",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    
    print("\n=== Transformed Request ===")
    print(json.dumps(transformed_request, indent=4, default=str))
    
    assert "messages" in transformed_request
    assert len(transformed_request["messages"]) == 2
    
    user_message = transformed_request["messages"][1]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_transform_request_with_cache_control_list_content():
    """
    Test transform_request moves cache_control only to the last content block when content is already a list.
    This prevents exceeding Anthropic's limit of 4 cache breakpoints.
    
    Input:
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a historian..."},
            {"type": "text", "text": "HUGE TEXT BODY"}
        ],
        "cache_control": {"type": "ephemeral"}
    }
    
    Expected Output:
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a historian..."
            },
            {
                "type": "text",
                "text": "HUGE TEXT BODY",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
    """
    import json
    config = OpenrouterConfig()
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a historian studying the fall of the Roman Empire."
                },
                {
                    "type": "text",
                    "text": "HUGE TEXT BODY"
                }
            ],
            "cache_control": {"type": "ephemeral"}
        },
        {
            "role": "user",
            "content": "What triggered the collapse?"
        }
    ]
    
    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-3-5-sonnet-20240620",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    
    print("\n=== Transformed Request (List Content) ===")
    print(json.dumps(transformed_request, indent=4, default=str))
    
    assert "messages" in transformed_request
    assert len(transformed_request["messages"]) == 2
    
    system_message = transformed_request["messages"][0]
    assert system_message["role"] == "system"
    assert isinstance(system_message["content"], list)
    assert len(system_message["content"]) == 2
    # Only the last content block should have cache_control
    assert "cache_control" not in system_message["content"][0]
    assert system_message["content"][1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_message


def test_openrouter_transform_request_with_cache_control_gemini():
    """
    Test transform_request moves cache_control to content blocks for Gemini models.
    
    Input:
    {
        "role": "user",
        "content": "Analyze this data",
        "cache_control": {"type": "ephemeral"}
    }
    
    Expected Output:
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Analyze this data",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
    """
    import json
    config = OpenrouterConfig()
    
    messages = [
        {
            "role": "user",
            "content": "Analyze this data",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    transformed_request = config.transform_request(
        model="openrouter/google/gemini-2.0-flash-exp:free",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    
    print("\n=== Transformed Request (Gemini) ===")
    print(json.dumps(transformed_request, indent=4, default=str))
    
    assert "messages" in transformed_request
    assert len(transformed_request["messages"]) == 1
    
    user_message = transformed_request["messages"][0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_transform_request_multiple_cache_controls():
    """
    Test that cache_control is only added to the last content block per message.
    This prevents exceeding Anthropic's limit of 4 cache breakpoints.
    
    When a message has 5 content blocks with cache_control at message level,
    only the 5th block should have cache_control, not all 5 blocks.
    """
    import json
    config = OpenrouterConfig()
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
                {"type": "text", "text": "Block 3"},
                {"type": "text", "text": "Block 4"},
                {"type": "text", "text": "Block 5"}
            ],
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-3-5-sonnet-20240620",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    
    print("\n=== Transformed Request (Multiple Blocks) ===")
    print(json.dumps(transformed_request, indent=4, default=str))
    
    system_message = transformed_request["messages"][0]
    assert len(system_message["content"]) == 5
    
    # Only the last block should have cache_control
    for i in range(4):
        assert "cache_control" not in system_message["content"][i], f"Block {i} should not have cache_control"
    
    assert system_message["content"][4]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_message


def test_openrouter_cost_tracking_non_streaming():
    """
    Test OpenRouter cost tracking for non-streaming completions.

    Verifies:
    1. Request includes usage.include=true to get cost data
    2. Response extracts cost from usage.cost and stores in _hidden_params
    """
    from unittest.mock import Mock, patch
    from litellm.types.utils import ModelResponse, Choices, Message, Usage

    config = OpenrouterConfig()

    # Test request adds usage parameter
    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-sonnet-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert "usage" in transformed_request
    assert transformed_request["usage"] == {"include": True}

    # Test response extracts cost
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "gen-123",
        "model": "openrouter/anthropic/claude-sonnet-4.5",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.00015}
    }
    mock_response.headers = {}

    model_response = ModelResponse(
        id="gen-123",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="Hello!", role="assistant"))],
        created=1234567890,
        model="openrouter/anthropic/claude-sonnet-4.5",
        object="chat.completion",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )

    with patch.object(OpenAIGPTConfig, 'transform_response', return_value=model_response):
        result = config.transform_response(
            model="openrouter/anthropic/claude-sonnet-4.5",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert hasattr(result, "_hidden_params")
    assert "llm_provider-x-litellm-response-cost" in result._hidden_params["additional_headers"]
    assert result._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.00015


def test_openrouter_cost_tracking_streaming():
    """
    Test OpenRouter cost tracking for streaming completions.

    Verifies:
    1. Request includes usage.include=true (same as non-streaming)
    2. Streaming chunks preserve usage/cost data in the final chunk
    3. Cost field is accessible in the usage object
    """
    config = OpenrouterConfig()

    # Test request adds usage parameter for streaming
    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-sonnet-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert "usage" in transformed_request
    assert transformed_request["usage"] == {"include": True}

    # Test streaming chunks preserve cost data
    handler = OpenRouterChatCompletionStreamingHandler(
        streaming_response=None, sync_stream=True
    )

    # First chunk - content only
    chunk1 = {
        "id": "gen-stream-456",
        "created": 1234567890,
        "model": "openrouter/anthropic/claude-sonnet-4.5",
        "choices": [{"delta": {"content": "Hello", "reasoning": None}, "index": 0}],
    }

    # Final chunk - usage and cost
    chunk2 = {
        "id": "gen-stream-456",
        "created": 1234567890,
        "model": "openrouter/anthropic/claude-sonnet-4.5",
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15, "cost": 0.0001},
        "choices": [{"delta": {"content": "", "reasoning": None}, "finish_reason": "stop", "index": 0}],
    }

    result1 = handler.chunk_parser(chunk1)
    result2 = handler.chunk_parser(chunk2)

    # First chunk has content, no usage
    assert result1.choices[0]["delta"]["content"] == "Hello"
    assert result1.usage is None

    # Final chunk has usage with cost preserved
    assert result2.choices[0]["finish_reason"] == "stop"
    assert result2.usage is not None
    assert result2.usage.prompt_tokens == 5
    assert result2.usage.completion_tokens == 10
    assert result2.usage.total_tokens == 15
    # Verify cost field is preserved in the Usage object - this is the key data for cost tracking
    # The chunk_parser converts the dict to a Usage Pydantic model which includes the cost field
    assert result2.usage.cost == 0.0001


def test_openrouter_reasoning_models_allow_reasoning_effort_param():
    """
    OpenRouter reasoning-capable models should accept the reasoning_effort param.
    """
    config = OpenrouterConfig()

    supported_params = config.get_supported_openai_params(
        model="openrouter/deepseek/deepseek-v3.2"
    )

    assert "reasoning_effort" in supported_params
    assert supported_params.count("reasoning_effort") == 1


def test_openrouter_non_reasoning_models_do_not_add_reasoning_effort():
    """
    Models without reasoning support should not gain reasoning-specific params.
    """
    config = OpenrouterConfig()

    supported_params = config.get_supported_openai_params(
        model="openrouter/anthropic/claude-3-5-haiku"
    )

    assert "reasoning_effort" not in supported_params
