"""
Test transformation logic for Docker Model Runner chat completions.

This test verifies that the DockerModelRunnerChatConfig correctly transforms
requests to the Docker Model Runner API format, including URL generation,
parameter mapping, and request/response transformation.
"""

import json
import os
import sys
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.docker_model_runner.chat.transformation import (
    DockerModelRunnerChatConfig,
)
from litellm.types.llms.openai import AllMessageValues


def _make_mock_http_response(response_body: dict, status_code: int = 200) -> MagicMock:
    """
    Create a mock httpx response with the standard pattern used across litellm tests.

    The mock response includes both .json() and .text attributes, as litellm
    may read either depending on the code path.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    return mock_response


class TestDockerModelRunnerChatUrlGeneration:
    """Tests for URL generation with various api_base configurations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerChatConfig()

    def test_get_complete_url_default_api_base(self):
        """Test URL generation with no api_base provided (uses default)."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="ai/smollm2",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/chat/completions"

    def test_get_complete_url_with_auto_select_engine(self):
        """Test URL generation with auto-select engine path."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/v1",
            api_key=None,
            model="ai/smollm2",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/chat/completions"

    def test_get_complete_url_with_llama_cpp_engine(self):
        """Test URL generation with explicit llama.cpp engine."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/llama.cpp/v1",
            api_key=None,
            model="ai/qwen2.5",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/llama.cpp/v1/chat/completions"

    def test_get_complete_url_with_vllm_engine(self):
        """Test URL generation with explicit vLLM engine."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/vllm/v1",
            api_key=None,
            model="ai/smollm2-vllm",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/vllm/v1/chat/completions"

    def test_get_complete_url_container_host(self):
        """Test URL generation from within a Docker container."""
        url = self.config.get_complete_url(
            api_base="http://model-runner.docker.internal/engines/v1",
            api_key=None,
            model="ai/smollm2",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://model-runner.docker.internal/engines/v1/chat/completions"
        assert "model-runner.docker.internal" in url

    def test_get_complete_url_removes_trailing_slash(self):
        """Test that trailing slashes are properly removed from api_base."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/v1/",
            api_key=None,
            model="ai/smollm2",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/chat/completions"
        assert "//v1" not in url

    def test_get_complete_url_custom_port(self):
        """Test URL generation with custom TCP port."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12435/engines/v1",
            api_key=None,
            model="ai/smollm2",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12435/engines/v1/chat/completions"


class TestDockerModelRunnerChatParameterMapping:
    """Tests for OpenAI parameter mapping and transformation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerChatConfig()

    def test_map_openai_params_basic(self):
        """Test basic parameter mapping preserves standard OpenAI params."""
        non_default_params = {"temperature": 0.5, "max_tokens": 200, "top_p": 0.9}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="ai/smollm2",
            drop_params=False,
        )
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9

    def test_map_max_completion_tokens_to_max_tokens(self):
        """Test that max_completion_tokens is mapped to max_tokens for DMR."""
        non_default_params = {"max_completion_tokens": 150}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="ai/smollm2",
            drop_params=False,
        )
        assert result["max_tokens"] == 150
        assert "max_completion_tokens" not in result

    def test_transform_request_body_basic(self):
        """Test basic request body transformation with messages."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        optional_params = {"temperature": 0.7, "max_tokens": 100}

        request_data = self.config.transform_request(
            model="ai/smollm2",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert request_data["model"] == "ai/smollm2"
        assert request_data["messages"] == messages
        assert request_data["temperature"] == 0.7
        assert request_data["max_tokens"] == 100

    def test_transform_request_body_with_system_message(self):
        """Test request body with system message included."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]

        request_data = self.config.transform_request(
            model="ai/llama3.2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert len(request_data["messages"]) == 2
        assert request_data["messages"][0]["role"] == "system"
        assert request_data["messages"][1]["role"] == "user"

    def test_validate_environment_returns_headers(self):
        """Test that validate_environment returns proper headers."""
        headers = self.config.validate_environment(
            headers={},
            model="ai/smollm2",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base="http://localhost:12434/engines/v1",
        )
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"

    def test_validate_environment_default_api_key(self):
        """Test that validate_environment uses dummy-key when no API key provided."""
        headers = self.config.validate_environment(
            headers={},
            model="ai/smollm2",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="http://localhost:12434/engines/v1",
        )
        assert headers["Authorization"] == "Bearer dummy-key"

    def test_get_supported_openai_params_includes_standard_params(self):
        """Test that supported params include standard OpenAI params."""
        supported = self.config.get_supported_openai_params(model="ai/smollm2")
        assert "temperature" in supported
        assert "top_p" in supported
        assert "max_tokens" in supported
        assert "stream" in supported
        assert "stop" in supported


class TestDockerModelRunnerChatResponseTransformation:
    """Tests for response transformation from DMR API format."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerChatConfig()

    def test_transform_response_standard(self):
        """Test standard response transformation."""
        from litellm.types.utils import ModelResponse

        dmr_response = {
            "id": "chatcmpl-test-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "ai/smollm2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }
        mock_response = _make_mock_http_response(dmr_response)

        model_response = ModelResponse()

        result = self.config.transform_response(
            model="ai/smollm2",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        assert result.choices[0].message.content == "Hello! How can I help you?"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.total_tokens == 25


class TestDockerModelRunnerVisionHandling:
    """Tests for vision/multimodal message handling in Docker Model Runner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerChatConfig()

    def test_multimodal_message_preserves_image_url_block(self):
        """
        Messages containing image_url blocks must NOT be converted to a plain string.
        The image_url dict must survive _transform_messages intact so the backend
        receives the actual image data.
        """
        messages: List[AllMessageValues] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

        result = self.config._transform_messages(messages=messages, model="ai/qwen3.6-27b", is_async=False)

        user_msg = result[0]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        assert isinstance(content, list), "content must remain a list when image_url is present"
        types = [c.get("type") for c in content]
        assert "image_url" in types, "image_url block must be preserved"
        image_url_block = [c for c in content if c.get("type") == "image_url"][0]
        assert "url" in image_url_block["image_url"]
        assert image_url_block["image_url"]["url"].startswith("data:image/")

    def test_text_only_message_still_converted_to_string(self):
        """
        Messages that contain only text blocks (or a plain string content)
        must still be converted to a string content, preserving backward
        compatibility with older DMR backends.
        """
        messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        result = self.config._transform_messages(messages=messages, model="ai/smollm2", is_async=False)

        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["content"] == "Hello, how are you?"

    def test_text_block_only_list_converted_to_string(self):
        """
        A content list containing only {type: 'text'} blocks should be
        collapsed into a single string, matching prior behaviour.
        """
        messages: List[AllMessageValues] = [
            {"role": "user", "content": [{"type": "text", "text": "Part 1 "}, {"type": "text", "text": "Part 2"}]}
        ]

        result = self.config._transform_messages(messages=messages, model="ai/smollm2", is_async=False)

        assert result[0]["content"] == "Part 1 Part 2"

    def test_mixed_messages_preserves_image_while_converting_text(self):
        """
        When a message batch contains both a text-only message and a
        multimodal message, the text-only one gets converted to a string
        and the multimodal one keeps its image_url block.
        """
        messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123", "detail": "high"},
                    },
                ],
            },
        ]

        result = self.config._transform_messages(messages=messages, model="ai/qwen3.6-27b", is_async=False)

        assert result[0]["content"] == "You are a helpful assistant."
        user_content = result[1]["content"]
        assert isinstance(user_content, list)
        assert any(c.get("type") == "image_url" for c in user_content)
        assert any(c.get("type") == "text" and c.get("text") == "Describe this image:" for c in user_content)

    def test_multimodal_message_in_the_middle_keeps_conversation_order(self):
        """
        A multimodal message surrounded by text-only messages must stay at
        its original position, otherwise the conversation order changes.
        """
        messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
                ],
            },
            {"role": "assistant", "content": "The image shows a cat."},
            {"role": "user", "content": "What color is it?"},
        ]

        result = self.config._transform_messages(messages=messages, model="ai/qwen3.6-27b", is_async=False)

        assert [m["role"] for m in result] == ["system", "user", "assistant", "user"]
        assert result[2]["content"] == "The image shows a cat."
        assert result[3]["content"] == "What color is it?"
        user_content = result[1]["content"]
        assert isinstance(user_content, list)
        assert any(c.get("type") == "image_url" for c in user_content)

    def test_input_audio_block_preserved(self):
        """
        input_audio blocks (and other non-text block types) should also
        be preserved so they reach the backend.
        """
        messages: List[AllMessageValues] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}},
                    {"type": "text", "text": "Transcribe this audio"},
                ],
            }
        ]

        result = self.config._transform_messages(messages=messages, model="ai/qwen3.6-27b", is_async=False)

        user_content = result[0]["content"]
        assert isinstance(user_content, list)
        types = [c.get("type") for c in user_content]
        assert "input_audio" in types

    def test_has_multimodal_content_detection(self):
        """
        Unit test for the _has_multimodal_content helper.
        """
        assert DockerModelRunnerChatConfig._has_multimodal_content({"role": "user", "content": "just text"}) is False
        assert (
            DockerModelRunnerChatConfig._has_multimodal_content(
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            )
            is False
        )
        assert (
            DockerModelRunnerChatConfig._has_multimodal_content(
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {}}]}
            )
            is True
        )
        assert (
            DockerModelRunnerChatConfig._has_multimodal_content(
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}
            )
            is True
        )
        assert DockerModelRunnerChatConfig._has_multimodal_content({"role": "user", "content": []}) is False
        assert DockerModelRunnerChatConfig._has_multimodal_content({"role": "user"}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    """Integration tests that verify the full transformation pipeline via HTTP mock."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerChatConfig()

    def test_url_generation_matches_endpoint_format(self):
        """Test that URL generation produces correct endpoint format."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/v1",
            api_key=None,
            model="ai/smollm2",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/chat/completions"
        assert "/v1/chat/completions" in url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
