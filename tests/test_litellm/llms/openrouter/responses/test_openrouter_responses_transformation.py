"""
Tests for OpenRouter Responses API configuration.

Validates that OpenRouter is registered as a native Responses API provider,
routing requests directly to https://openrouter.ai/api/v1/responses instead
of falling back to the chat completion bridge. This is required to preserve
reasoning.encrypted_content for multi-turn stateless workflows.

Related issue: https://github.com/BerriAI/litellm/issues/22189
"""

import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.openrouter.responses.transformation import (
    OpenRouterResponsesAPIConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestOpenRouterResponsesAPIConfig:
    """Test OpenRouter Responses API configuration."""

    def test_custom_llm_provider(self):
        """custom_llm_provider should return OPENROUTER."""
        config = OpenRouterResponsesAPIConfig()
        assert config.custom_llm_provider == LlmProviders.OPENROUTER

    def test_get_complete_url_default(self):
        """Default URL should point to OpenRouter's Responses API endpoint."""
        config = OpenRouterResponsesAPIConfig()
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://openrouter.ai/api/v1/responses"

    def test_get_complete_url_custom_base(self):
        """Custom api_base should be respected."""
        config = OpenRouterResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://custom.openrouter.ai/api/v1",
            litellm_params={},
        )
        assert url == "https://custom.openrouter.ai/api/v1/responses"

    def test_get_complete_url_strips_trailing_slash(self):
        """Trailing slashes on api_base should be stripped."""
        config = OpenRouterResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://openrouter.ai/api/v1/",
            litellm_params={},
        )
        assert url == "https://openrouter.ai/api/v1/responses"

    def test_validate_environment_sets_auth_header(self):
        """validate_environment should set the Authorization header."""
        config = OpenRouterResponsesAPIConfig()
        from litellm.types.router import GenericLiteLLMParams

        params = GenericLiteLLMParams(api_key="sk-or-test-key")
        headers = config.validate_environment(
            headers={}, model="openai/o4-mini", litellm_params=params
        )
        assert headers["Authorization"] == "Bearer sk-or-test-key"

    def test_validate_environment_raises_without_key(self, monkeypatch):
        """validate_environment should raise when no API key is available."""
        config = OpenRouterResponsesAPIConfig()
        from litellm.types.router import GenericLiteLLMParams

        # Clear any globally set API keys so the validation correctly raises
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OR_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenRouter API key is required") as exc_info:
            config.validate_environment(
                headers={},
                model="openai/o4-mini",
                litellm_params=GenericLiteLLMParams(),
            )
        e = exc_info.value
        assert "OpenRouter API key is required" in str(e)


class TestOpenRouterResponsesAPICostTracking:
    """
    Regression tests: OpenRouter's Responses API must request and extract cost
    data the same way the chat completions path already does. Without this,
    every request routed through the Responses API (e.g. Codex-style clients)
    gets logged with $0 spend despite real token usage.
    """

    def test_transform_request_adds_usage_include(self):
        """Request should always ask OpenRouter to include usage.cost in the response."""
        config = OpenRouterResponsesAPIConfig()
        from litellm.types.router import GenericLiteLLMParams

        request = config.transform_responses_api_request(
            model="openai/gpt-5-mini",
            input="Hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request.get("usage") == {"include": True}

    def test_transform_request_preserves_existing_usage_param(self):
        """An explicitly-set usage param should not be clobbered."""
        config = OpenRouterResponsesAPIConfig()
        from litellm.types.router import GenericLiteLLMParams

        request = config.transform_responses_api_request(
            model="openai/gpt-5-mini",
            input="Hello",
            response_api_optional_request_params={"usage": {"include": False}},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request.get("usage") == {"include": False}

    def test_transform_response_extracts_cost(self):
        """Response should pull usage.cost into hidden params for the cost calculator."""
        config = OpenRouterResponsesAPIConfig()

        body = {
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "openai/gpt-5-mini",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.01234,
            },
        }
        raw_response = Mock(spec=httpx.Response)
        raw_response.text = json.dumps(body)
        raw_response.json.return_value = body
        raw_response.headers = {}

        result = config.transform_response_api_response(
            model="openai/gpt-5-mini",
            raw_response=raw_response,
            logging_obj=Mock(),
        )

        assert (
            result._hidden_params["additional_headers"][
                "llm_provider-x-litellm-response-cost"
            ]
            == 0.01234
        )

    def test_transform_response_without_cost_does_not_error(self):
        """Missing usage.cost should not raise or set the header."""
        config = OpenRouterResponsesAPIConfig()

        body = {
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "openai/gpt-5-mini",
            "output": [],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
        raw_response = Mock(spec=httpx.Response)
        raw_response.text = json.dumps(body)
        raw_response.json.return_value = body
        raw_response.headers = {}

        result = config.transform_response_api_response(
            model="openai/gpt-5-mini",
            raw_response=raw_response,
            logging_obj=Mock(),
        )

        assert "llm_provider-x-litellm-response-cost" not in result._hidden_params.get(
            "additional_headers", {}
        )


class TestOpenRouterResponsesAPIRegistration:
    """Test that OpenRouter is properly registered as a native Responses API provider."""

    def test_provider_config_manager_returns_openrouter_config(self):
        """
        ProviderConfigManager.get_provider_responses_api_config should return
        OpenRouterResponsesAPIConfig for the OPENROUTER provider, NOT None.

        When it returns None, requests fall through to the completion bridge,
        which loses encrypted_content (the bug in issue #22189).
        """
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENROUTER,
        )
        assert config is not None, (
            "OpenRouter must be registered as a native Responses API provider "
            "to preserve reasoning.encrypted_content"
        )
        assert isinstance(config, OpenRouterResponsesAPIConfig)

    def test_openrouter_not_using_completion_bridge(self):
        """
        Verify that OpenRouter does NOT fall through to the completion bridge.
        The completion bridge drops encrypted_content because chat completions
        use a different format (reasoning_details) than the Responses API.
        """
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENROUTER,
        )
        # If config is not None, the native Responses API path is used
        assert config is not None
        # The URL should point to OpenRouter's responses endpoint
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert "/responses" in url
