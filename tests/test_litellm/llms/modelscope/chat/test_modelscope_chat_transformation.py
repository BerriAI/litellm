"""
Unit tests for ModelScope configuration.

These tests validate the ModelScopeChatConfig class which extends OpenAIGPTConfig.
ModelScope is an OpenAI-compatible provider with minor customizations.
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm import completion
from litellm.llms.modelscope.chat.transformation import ModelScopeChatConfig

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"


class TestModelScopeConfig:
    """Test class for ModelScope functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = ModelScopeChatConfig()
        headers = {}
        api_key = "fake-modelscope-key"

        result = config.validate_environment(
            headers=headers,
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    @pytest.mark.respx()
    def test_modelscope_completion_mock(self, respx_mock):
        """Mock test for basic ModelScope completion."""

        litellm.disable_aiohttp_transport = True

        api_key = "fake-modelscope-key"
        api_base = "https://api-inference.modelscope.cn/v1"

        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": DEFAULT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```python\nprint("Hey from LiteLLM!")\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        response = completion(
            model=f"modelscope/{DEFAULT_MODEL}",
            messages=[
                {"role": "user", "content": "write code for saying hey from LiteLLM"}
            ],
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert response.choices[0].message.content is not None
        assert "```python" in response.choices[0].message.content

    # ── _transform_messages tests ──────────────────────────────────────

    def test_transform_messages_flattens_text_content_list(self):
        """Content lists containing only text items should be flattened to a string."""
        config = ModelScopeChatConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"},
                ],
            }
        ]

        result = config._transform_messages(messages=messages, model=DEFAULT_MODEL)

        assert result[0]["content"] == "Hello world"

    def test_transform_messages_preserves_multimodal_content_list(self):
        """Content lists with image_url should be preserved as lists for vision models."""
        config = ModelScopeChatConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]

        result = config._transform_messages(messages=messages, model=DEFAULT_MODEL)

        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"

    def test_transform_messages_string_content_unchanged(self):
        """Messages with string content should pass through unchanged."""
        config = ModelScopeChatConfig()
        messages = [{"role": "user", "content": "Hello"}]

        result = config._transform_messages(messages=messages, model=DEFAULT_MODEL)

        assert result[0]["content"] == "Hello"

    def test_transform_messages_multi_turn(self):
        """Multi-turn conversations should be handled correctly."""
        config = ModelScopeChatConfig()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me more"},
                ],
            },
        ]

        result = config._transform_messages(messages=messages, model=DEFAULT_MODEL)

        assert result[0]["content"] == "Hi"
        assert result[1]["content"] == "Hello!"
        assert result[2]["content"] == "Tell me more"

    def test_transform_messages_multimodal_multi_turn(self):
        """Multi-turn with mixed text-only and multimodal messages."""
        config = ModelScopeChatConfig()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
                ],
            },
        ]

        result = config._transform_messages(messages=messages, model=DEFAULT_MODEL)

        assert result[0]["content"] == "Hi"
        assert result[1]["content"] == "Hello!"
        # Multimodal message should keep list format
        assert isinstance(result[2]["content"], list)
        assert result[2]["content"][1]["type"] == "image_url"

    # ── get_complete_url tests ─────────────────────────────────────────

    def test_get_complete_url_default(self):
        """Default api_base should append /chat/completions."""
        config = ModelScopeChatConfig()

        url = config.get_complete_url(
            api_base=None,
            api_key="fake-key",
            model=DEFAULT_MODEL,
            optional_params={},
            litellm_params={},
        )

        assert url == "https://api-inference.modelscope.cn/v1/chat/completions"

    def test_get_complete_url_custom_base(self):
        """Custom api_base should append /chat/completions."""
        config = ModelScopeChatConfig()

        url = config.get_complete_url(
            api_base="https://custom.modelscope.cn/v1",
            api_key="fake-key",
            model=DEFAULT_MODEL,
            optional_params={},
            litellm_params={},
        )

        assert url == "https://custom.modelscope.cn/v1/chat/completions"

    def test_get_complete_url_already_has_endpoint(self):
        """api_base already ending in /chat/completions should not be doubled."""
        config = ModelScopeChatConfig()

        url = config.get_complete_url(
            api_base="https://api-inference.modelscope.cn/v1/chat/completions",
            api_key="fake-key",
            model=DEFAULT_MODEL,
            optional_params={},
            litellm_params={},
        )

        assert url == "https://api-inference.modelscope.cn/v1/chat/completions"
        assert url.count("/chat/completions") == 1

    # ── _get_openai_compatible_provider_info tests ─────────────────────

    def test_get_provider_info_with_explicit_api_base(self):
        """Explicit api_base and api_key should be returned as-is."""
        config = ModelScopeChatConfig()

        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base="https://custom.example.com/v1",
            api_key="my-key",
        )

        assert api_base == "https://custom.example.com/v1"
        assert api_key == "my-key"

    def test_get_provider_info_default_fallback(self):
        """When no api_base or env var is set, DEFAULT_BASE_URL should be used."""
        config = ModelScopeChatConfig()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODELSCOPE_API_BASE", None)
            os.environ.pop("MODELSCOPE_API_KEY", None)

            api_base, api_key = config._get_openai_compatible_provider_info(
                api_base=None,
                api_key=None,
            )

        assert api_base == "https://api-inference.modelscope.cn/v1"
        assert api_key is None

    def test_get_provider_info_env_var_fallback(self):
        """MODELSCOPE_API_BASE env var should be used when api_base is not provided."""
        config = ModelScopeChatConfig()

        with patch.dict(
            os.environ,
            {"MODELSCOPE_API_BASE": "https://env.modelscope.cn/v1"},
        ):
            api_base, _ = config._get_openai_compatible_provider_info(
                api_base=None,
                api_key=None,
            )

        assert api_base == "https://env.modelscope.cn/v1"

    # ── Mock HTTP tests ────────────────────────────────────────────────

    @pytest.mark.respx()
    def test_completion_with_text_content_list(self, respx_mock):
        """Verify that text-only content list messages are flattened before sending."""
        litellm.disable_aiohttp_transport = True

        api_key = "fake-modelscope-key"
        api_base = "https://api-inference.modelscope.cn/v1"
        captured_request = {}

        def capture_request(request):
            captured_request["body"] = request.content
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-456",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": DEFAULT_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Sure!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            )

        respx_mock.post(f"{api_base}/chat/completions").mock(side_effect=capture_request)

        response = completion(
            model=f"modelscope/{DEFAULT_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " world"},
                    ],
                }
            ],
            api_key=api_key,
            api_base=api_base,
        )

        assert response.choices[0].message.content == "Sure!"

        body = json.loads(captured_request["body"])
        assert isinstance(body["messages"][0]["content"], str)
        assert body["messages"][0]["content"] == "Hello world"

    @pytest.mark.respx()
    def test_completion_with_multimodal_messages(self, respx_mock):
        """Verify that multimodal messages (text + image_url) are sent as content lists."""
        litellm.disable_aiohttp_transport = True

        api_key = "fake-modelscope-key"
        api_base = "https://api-inference.modelscope.cn/v1"
        captured_request = {}

        def capture_request(request):
            captured_request["body"] = request.content
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-789",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": DEFAULT_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "A cat sitting on a couch.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108},
                },
            )

        respx_mock.post(f"{api_base}/chat/completions").mock(side_effect=capture_request)

        response = completion(
            model=f"modelscope/{DEFAULT_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.jpg"},
                        },
                    ],
                }
            ],
            api_key=api_key,
            api_base=api_base,
        )

        assert response.choices[0].message.content == "A cat sitting on a couch."

        body = json.loads(captured_request["body"])
        msg = body["messages"][0]
        # Multimodal content should remain as a list
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "What is in this image?"}
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"] == "https://example.com/cat.jpg"
