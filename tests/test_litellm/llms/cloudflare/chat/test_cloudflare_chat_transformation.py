"""
Unit tests for Cloudflare Workers AI chat transformation.

Tests the CloudflareChatConfig including parameter handling, URL construction,
header generation, and response transformation.
"""

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.cloudflare.chat.transformation import (
    CloudflareChatConfig,
    CloudflareChatResponseIterator,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function, ModelResponse, Usage


class TestCloudflareChatConfig:
    """Test CloudflareChatConfig parameter handling and transformation."""

    def setup_method(self):
        self.config = CloudflareChatConfig()
        self.model = "@cf/meta/llama-3.1-8b-instruct"

    def test_get_supported_openai_params(self):
        """Test that all expected params are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "stream" in params
        assert "max_tokens" in params
        assert "max_completion_tokens" in params
        assert "temperature" in params
        assert "top_p" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params
        assert "tools" in params
        assert "tool_choice" in params
        assert "response_format" in params
        assert "stop" in params
        assert "seed" in params

    def test_map_openai_params_temperature(self):
        """Test that temperature is passed through correctly."""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["temperature"] == 0.7

    def test_map_openai_params_max_completion_tokens(self):
        """Test that max_completion_tokens maps to max_tokens."""
        result = self.config.map_openai_params(
            non_default_params={"max_completion_tokens": 512},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["max_tokens"] == 512
        assert "max_completion_tokens" not in result

    def test_map_openai_params_tools(self):
        """Test that tools param is passed through."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = self.config.map_openai_params(
            non_default_params={"tools": tools},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["tools"] == tools

    def test_map_openai_params_response_format(self):
        """Test that response_format is passed through."""
        response_format = {"type": "json_object"}
        result = self.config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["response_format"] == response_format

    def test_transform_request(self):
        """Test that transform_request builds correct data structure."""
        messages = [{"role": "user", "content": "hello"}]
        optional_params = {"temperature": 0.5, "max_tokens": 100}
        data = self.config.transform_request(
            model=self.model,
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert data["messages"] == messages
        assert data["temperature"] == 0.5
        assert data["max_tokens"] == 100

    def test_validate_environment(self):
        """Test that validate_environment sets correct headers."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key-123",
        )
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "application/json"

    def test_validate_environment_missing_key(self):
        """Test that validate_environment raises error when key is missing."""
        with pytest.raises(ValueError, match="Missing CloudflareError API Key"):
            self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_content_type_header_no_typo(self):
        """Test that content-type header does not have the 'apbplication' typo."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        assert headers["content-type"] == "application/json"
        assert "apbplication" not in headers["content-type"]

    @patch("litellm.llms.cloudflare.chat.transformation.get_secret_str")
    def test_get_complete_url_with_api_base(self, mock_get_secret):
        """Test URL construction with explicit api_base."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/ai/run/",
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert url == f"https://custom.api.com/ai/run/{self.model}"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.cloudflare.chat.transformation.get_secret_str")
    def test_get_complete_url_default(self, mock_get_secret):
        """Test URL construction with default api_base (uses account ID)."""
        mock_get_secret.return_value = "test-account-id"
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert (
            url
            == f"https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/{self.model}"
        )

    def test_transform_response_native_format(self):
        """Test transform_response handles native Cloudflare format."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": {"response": "Hello! How can I help you?"},
            "success": True,
        }
        model_response = ModelResponse()

        # Mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]

        with patch("litellm.utils.get_token_count", return_value=10):
            result = self.config.transform_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=MagicMock(),
                request_data={},
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
                encoding=mock_encoding,
            )

        assert result.choices[0].message.content == "Hello! How can I help you?"
        assert "cloudflare/" in result.model

    def test_transform_response_with_usage(self):
        """Test transform_response uses usage from result when available."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": {
                "response": "Hi!",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                },
            },
            "success": True,
        }
        model_response = ModelResponse()
        mock_encoding = MagicMock()

        result = self.config.transform_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3
        assert result.usage.total_tokens == 8


    def test_transform_response_tool_calls(self):
        """Test transform_response converts tool calls to proper types."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"location": "London"}',
                    }
                ]
            },
            "success": True,
        }
        model_response = ModelResponse()
        mock_encoding = MagicMock()

        result = self.config.transform_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "weather?"}],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert result.choices[0].finish_reason == "tool_calls"
        tool_call = result.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ChatCompletionMessageToolCall)
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == '{"location": "London"}'

    def test_transform_response_null_result(self):
        """Test transform_response handles null result without crashing."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": None,
            "success": False,
        }
        model_response = ModelResponse()
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = []

        with patch("litellm.utils.get_token_count", return_value=0):
            result = self.config.transform_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=MagicMock(),
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=mock_encoding,
            )

        assert result.choices[0].message.content == ""


class TestCloudflareChatResponseIterator:
    """Test streaming response chunk parsing."""

    def test_chunk_parser_text(self):
        """Test that response text is extracted from chunk."""
        iterator = CloudflareChatResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )
        chunk = {"response": "Hello"}
        result = iterator.chunk_parser(chunk)
        assert result["text"] == "Hello"
        assert result["is_finished"] is False

    def test_chunk_parser_finish_reason(self):
        """Test that finish_reason is detected from chunk."""
        iterator = CloudflareChatResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )
        chunk = {"response": "", "finish_reason": "stop"}
        result = iterator.chunk_parser(chunk)
        assert result["is_finished"] is True
        assert result["finish_reason"] == "stop"

    def test_chunk_parser_index(self):
        """Test that index is extracted from chunk."""
        iterator = CloudflareChatResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )
        chunk = {"response": "hi", "index": 2}
        result = iterator.chunk_parser(chunk)
        assert result["index"] == 2
