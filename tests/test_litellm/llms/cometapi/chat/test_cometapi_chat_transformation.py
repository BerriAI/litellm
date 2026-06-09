"""
Unit tests for CometAPI Chat Configuration

Tests the CometAPIChatConfig class methods using mocks
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.cometapi.chat.transformation import (
    CometAPIChatCompletionStreamingHandler,
    CometAPIConfig,
)
from litellm.llms.cometapi.common_utils import CometAPIException


class TestCometAPIChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = CometAPIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test input chunk
        chunk = {
            "id": "test_id",
            "created": 1234567890,
            "model": "gpt-5.5",
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
        assert result.model == "gpt-5.5"
        assert result.usage.prompt_tokens == chunk["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == chunk["usage"]["completion_tokens"]
        assert result.usage.total_tokens == chunk["usage"]["total_tokens"]
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["reasoning_content"] == "test reasoning"

    def test_chunk_parser_error_response(self):
        handler = CometAPIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test error chunk
        error_chunk = {
            "error": {
                "message": "test error",
                "code": 400,
            }
        }

        # Verify error handling
        with pytest.raises(CometAPIException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert "CometAPI Error: test error" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_chunk_parser_key_error(self):
        handler = CometAPIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test invalid chunk missing required fields
        invalid_chunk = {"incomplete": "data"}

        # Verify KeyError handling
        with pytest.raises(CometAPIException) as exc_info:
            handler.chunk_parser(invalid_chunk)

        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


class TestCometAPIConfig:
    def test_transform_request_basic(self):
        """Test basic request transformation"""
        config = CometAPIConfig()

        transformed_request = config.transform_request(
            model="cometapi/gpt-5.5",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert transformed_request["model"] == "cometapi/gpt-5.5"
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_transform_request_with_extra_body(self):
        """Test request transformation with extra_body parameters"""
        config = CometAPIConfig()

        transformed_request = config.transform_request(
            model="cometapi/gpt-5.5",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={"extra_body": {"custom_param": "custom_value"}},
            litellm_params={},
            headers={},
        )

        # Validate that extra_body parameters are merged into the request
        assert transformed_request["custom_param"] == "custom_value"
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_transform_request_allows_empty_extra_body(self):
        config = CometAPIConfig()

        transformed_request = config.transform_request(
            model="cometapi/gpt-5.5",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={"extra_body": None},
            litellm_params={},
            headers={},
        )

        assert transformed_request["model"] == "cometapi/gpt-5.5"
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_transform_request_extra_body_can_override_request_fields(self):
        """Test extra_body preserves LiteLLM's existing override behavior"""
        config = CometAPIConfig()

        transformed_request = config.transform_request(
            model="cometapi/gpt-5.5",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={"extra_body": {"model": "cometapi/gpt-5.5-all"}},
            litellm_params={},
            headers={},
        )

        assert transformed_request["model"] == "cometapi/gpt-5.5-all"

    def test_cache_control_flag_removal(self):
        """Test cache control flag removal from messages"""
        config = CometAPIConfig()

        transformed_request = config.transform_request(
            model="cometapi/gpt-5.5",
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

        # CometAPI should remove cache_control flags by default
        assert transformed_request["messages"][0].get("cache_control") is None

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping"""
        config = CometAPIConfig()

        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }

        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="cometapi/gpt-5.5",
            drop_params=False,
        )

        assert mapped_params["temperature"] == 0.7
        assert mapped_params["max_tokens"] == 100
        assert mapped_params["top_p"] == 0.9

    def test_get_error_class(self):
        """Test error class creation"""
        config = CometAPIConfig()

        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, CometAPIException)
        assert error.message == "Test error"
        assert error.status_code == 400

    def test_get_complete_url(self):
        """Test CometAPI chat endpoint URL normalization"""
        config = CometAPIConfig()

        assert (
            config.get_complete_url(
                api_base="https://api.cometapi.com/v1",
                api_key="comet-explicit-key",
                model="gpt-5.5",
                optional_params={},
                litellm_params={},
            )
            == "https://api.cometapi.com/v1/chat/completions"
        )
