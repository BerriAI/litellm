import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

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
    Test transform_request moves cache_control to all content blocks when content is already a list.
    
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
                "text": "You are a historian...",
                "cache_control": {"type": "ephemeral"}
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
    assert system_message["content"][0]["cache_control"] == {"type": "ephemeral"}
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
    