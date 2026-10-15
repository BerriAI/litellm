"""
Unit tests for PublicAI configuration.

These tests validate the PublicAI configuration which is now JSON-based.
PublicAI is an OpenAI-compatible provider with minor customizations.
"""

import json
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.openai_like.dynamic_config import create_config_class


PUBLICAI_COMPLETION_RESPONSE: Final = {
    "id": "publicai-completion",
    "object": "chat.completion",
    "created": 11111,
    "model": "swiss-ai/apertus-8b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from PublicAI"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
}
PUBLICAI_STREAMING_RESPONSE: Final = b"".join(
    (
        b'data: {"id":"publicai-stream-1","object":"chat.completion.chunk","created":11111,'
        b'"model":"swiss-ai/apertus-8b-instruct","choices":[{"index":0,"delta":{"content":"Hello "},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"id":"publicai-stream-2","object":"chat.completion.chunk","created":11111,'
        b'"model":"swiss-ai/apertus-8b-instruct","choices":[{"index":0,"delta":{"content":"there"},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"id":"publicai-stream-3","object":"chat.completion.chunk","created":11111,'
        b'"model":"swiss-ai/apertus-8b-instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    )
)


class TestPublicAIConfig:
    """Test class for PublicAI functionality"""

    @pytest.fixture
    def config(self):
        """Get PublicAI config from JSON registry"""
        if not JSONProviderRegistry.exists("publicai"):
            pytest.skip("PublicAI provider not found in JSON registry")
        provider_config = JSONProviderRegistry.get("publicai")
        if provider_config is None:
            pytest.skip("PublicAI provider not found in JSON registry")
        return create_config_class(provider_config)()

    def test_default_api_base(self, config):
        """
        Test that default API base is used when none is provided
        """
        headers = {}
        api_key = "fake-publicai-key"

        result = config.validate_environment(
            headers=headers,
            model="swiss-ai-apertus",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    @patch("litellm.utils.supports_function_calling", return_value=True)
    def test_get_supported_openai_params(self, mock_supports_fc, config):
        """
        Test that get_supported_openai_params returns correct params.
        We mock supports_function_calling because the test model name
        'swiss-ai-apertus' is not in the model registry; this test validates
        config behaviour, not registry lookups.
        """
        supported_params = config.get_supported_openai_params(model="swiss-ai-apertus")

        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params

        # Note: JSON-based configs inherit from OpenAIGPTConfig which includes functions
        # This is expected behavior for JSON-based providers

    @patch("litellm.utils.supports_function_calling", return_value=True)
    def test_map_openai_params_includes_functions(self, mock_supports_fc, config):
        """
        Test that functions parameter is mapped (JSON-based configs don't exclude functions).
        We mock supports_function_calling because the test model name
        'swiss-ai-apertus' is not in the model registry.
        """
        non_default_params = {
            "functions": [{"name": "test_function", "description": "Test function"}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="swiss-ai-apertus",
            drop_params=False,
        )

        # JSON-based configs inherit from OpenAIGPTConfig which includes functions
        assert "functions" in result
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 1000

    def test_map_openai_params_max_completion_tokens_mapping(self, config):
        """
        Test that max_completion_tokens is mapped to max_tokens
        """
        non_default_params = {"max_completion_tokens": 1000, "temperature": 0.7}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="swiss-ai-apertus",
            drop_params=False,
        )

        assert result.get("max_tokens") == 1000
        assert "max_completion_tokens" not in result
        assert result.get("temperature") == 0.7

    def test_get_complete_url(self, config):
        """
        Test that get_complete_url constructs the correct endpoint URL
        """
        url = config.get_complete_url(
            api_base=None,
            api_key="fake-key",
            model="swiss-ai-apertus",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://api.publicai.co/v1/chat/completions"

    def test_get_complete_url_with_custom_base(self, config):
        """
        Test that get_complete_url works with custom api_base
        """
        url = config.get_complete_url(
            api_base="https://custom.publicai.co/v1",
            api_key="fake-key",
            model="swiss-ai-apertus",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://custom.publicai.co/v1/chat/completions"


@pytest.mark.respx(assert_all_called=True)
def test_publicai_content_list_conversion_reaches_provider_as_text(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.publicai.co/v1/chat/completions")
    route.return_value = httpx.Response(200, json=PUBLICAI_COMPLETION_RESPONSE)

    response: Final = litellm.completion(
        model="publicai/swiss-ai/apertus-8b-instruct",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Say hello"}]}],
        max_tokens=10,
        api_key="test-api-key",
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "swiss-ai/apertus-8b-instruct",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
    }
    assert response.choices[0].message.content == "Hello from PublicAI"
    assert response.usage is not None
    assert response.usage.total_tokens == 7


@pytest.mark.respx(assert_all_called=True)
def test_publicai_parameter_mapping_reaches_provider_as_max_tokens(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.publicai.co/v1/chat/completions")
    route.return_value = httpx.Response(200, json=PUBLICAI_COMPLETION_RESPONSE)

    response: Final = litellm.completion(
        model="publicai/swiss-ai/apertus-8b-instruct",
        messages=[{"role": "user", "content": "Hi"}],
        max_completion_tokens=5,
        api_key="test-api-key",
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "swiss-ai/apertus-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
    }
    assert response.choices[0].message.content == "Hello from PublicAI"


@pytest.mark.respx(assert_all_called=True)
def test_publicai_completion_basic_uses_provider_request(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.publicai.co/v1/chat/completions")
    route.return_value = httpx.Response(200, json=PUBLICAI_COMPLETION_RESPONSE)
    messages: Final = [{"role": "user", "content": "Say hello"}]

    response: Final = litellm.completion(
        model="publicai/swiss-ai/apertus-8b-instruct",
        messages=messages,
        max_tokens=10,
        api_key="test-api-key",
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "swiss-ai/apertus-8b-instruct",
        "messages": messages,
        "max_tokens": 10,
    }
    assert response.choices[0].message.content == "Hello from PublicAI"


@pytest.mark.respx(assert_all_called=True)
def test_publicai_completion_with_streaming_uses_provider_request(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.publicai.co/v1/chat/completions")
    route.return_value = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=PUBLICAI_STREAMING_RESPONSE,
    )
    messages: Final = [{"role": "user", "content": "Say hello"}]

    chunks: Final = tuple(
        litellm.completion(
            model="publicai/swiss-ai/apertus-8b-instruct",
            messages=messages,
            max_tokens=10,
            stream=True,
            api_key="test-api-key",
        )
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "swiss-ai/apertus-8b-instruct",
        "messages": messages,
        "max_tokens": 10,
        "stream": True,
    }
    assert tuple(chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content) == (
        "Hello ",
        "there",
    )
