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
