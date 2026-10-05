"""
Unit tests for OVHCloud AI Endpoints chat integration.
"""


import pytest

from litellm.llms.ovhcloud.utils import OVHCloudException
from litellm.utils import get_optional_params


from litellm.llms.ovhcloud.chat.transformation import (
    OVHCloudChatCompletionStreamingHandler,
    OVHCloudChatConfig,
)

config = OVHCloudChatConfig()
model = "ovhcloud/Mistral-7B-Instruct-v0.3"


class TestOvhCloudChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        chunk = {
            "id": "test_id",
            "created": 1234567890,
            "model": "gpt-oss-20b",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [
                {"delta": {"content": "test content", "reasoning": "test reasoning"}}
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "test_id"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "gpt-oss-20b"
        assert result.usage.prompt_tokens == chunk["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == chunk["usage"]["completion_tokens"]
        assert result.usage.total_tokens == chunk["usage"]["total_tokens"]
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["reasoning_content"] == "test reasoning"

    def test_chunk_parser_error_response(self):
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        error_chunk = {
            "error": {
                "message": "test error",
                "code": 400,
            }
        }

        with pytest.raises(OVHCloudException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert "OVHCloud Error: test error" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_chunk_parser_key_error(self):
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        invalid_chunk = {"incomplete": "data"}

        with pytest.raises(OVHCloudException) as exc_info:
            handler.chunk_parser(invalid_chunk)

        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


class TestOVHCloudConfig:
    def test_transform_request_basic(self):
        """Test basic request transformation"""
        transformed_request = config.transform_request(
            model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert transformed_request["model"] == model
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_transform_request_with_extra_body(self):
        """Test request transformation with extra_body parameters"""
        transformed_request = config.transform_request(
            model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={"extra_body": {"custom_param": "custom_value"}},
            litellm_params={},
            headers={},
        )

        assert transformed_request["custom_param"] == "custom_value"
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping"""
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }

        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )

        assert mapped_params["temperature"] == 0.7
        assert mapped_params["max_tokens"] == 100
        assert mapped_params["top_p"] == 0.9

    def test_get_error_class(self):
        """Test error class creation"""
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, OVHCloudException)
        assert error.message == "Test error"
        assert error.status_code == 400

    @pytest.mark.parametrize(
        "model",
        [
            "Meta-Llama-3_3-70B-Instruct",
            "Meta-Llama-3_1-70B-Instruct",
            "Mixtral-8x7B-Instruct-v0.1",
            "gpt-oss-120b",
            "some-model-not-in-the-cost-map",
        ],
    )
    def test_tools_not_filtered_by_static_model_map(self, model):
        """
        OVHCloud AI Endpoints are OpenAI-compatible; tools/tool_choice must pass
        through for any model. The server is responsible for rejecting unsupported
        tool calls — LiteLLM must not strip them based on a stale static catalog.
        """

        params = get_optional_params(
            model=model,
            custom_llm_provider="ovhcloud",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "x", "parameters": {}},
                }
            ],
            tool_choice="auto",
        )

        assert "tools" in params
        assert "tool_choice" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestOVHCloudReasoningFieldMigration:
    """Tests for OVHCloud reasoning_content -> reasoning field migration."""

    def test_streaming_new_reasoning_field(self):
        """New `reasoning` field should be mapped to `reasoning_content`."""
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=iter([]),
                        sync_stream=True,
        )
        chunk = {
            "id": "test-id",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "reasoning": "Let me think...",
                    },
                    "index": 0,
                }
            ],
        }
        result = handler.chunk_parser(chunk)
        assert result.choices[0]["delta"]["reasoning_content"] == "Let me think..."

    def test_streaming_legacy_reasoning_content_unchanged(self):
        """Legacy `reasoning_content` field should pass through untouched."""
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=iter([]),
                        sync_stream=True,
        )
        chunk = {
            "id": "test-id",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "Already correct field.",
                    },
                    "index": 0,
                }
            ],
        }
        result = handler.chunk_parser(chunk)
        assert result.choices[0]["delta"]["reasoning_content"] == "Already correct field."

    def test_streaming_both_fields_legacy_wins(self):
        """When both fields present, existing `reasoning_content` is not overwritten."""
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=iter([]),
                        sync_stream=True,
        )
        chunk = {
            "id": "test-id",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "delta": {
                        "reasoning": "new field",
                        "reasoning_content": "legacy field",
                    },
                    "index": 0,
                }
            ],
        }
        result = handler.chunk_parser(chunk)
        assert result.choices[0]["delta"]["reasoning_content"] == "legacy field"
