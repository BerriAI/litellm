"""
Tests for OpenRouter Responses API configuration.

Validates that OpenRouter is registered as a native Responses API provider,
routing requests directly to https://openrouter.ai/api/v1/responses instead
of falling back to the chat completion bridge. This is required to preserve
reasoning.encrypted_content for multi-turn stateless workflows.

Related issue: https://github.com/BerriAI/litellm/issues/22189
"""

from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.openrouter.responses.transformation import (
    OpenRouterResponsesAPIConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _successful_response_body(cost: float | None = None) -> dict:
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "openai/gpt-4o-mini",
        "output": [],
        "usage": usage,
    }


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
        headers = config.validate_environment(headers={}, model="openai/o4-mini", litellm_params=params)
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

    def test_transform_request_includes_usage(self):
        """OpenRouter must return usage details for response cost accounting."""
        from litellm.types.router import GenericLiteLLMParams

        request = OpenRouterResponsesAPIConfig().transform_responses_api_request(
            model="openai/gpt-4o-mini",
            input="hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["usage"] == {"include": True}

    def test_transform_response_stores_usage_cost(self):
        """OpenRouter usage.cost must reach the LiteLLM cost calculator."""
        raw_response = httpx.Response(200, json=_successful_response_body(cost=0.0125))

        response = OpenRouterResponsesAPIConfig().transform_response_api_response(
            model="openai/gpt-4o-mini",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )

        assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.0125

    def test_transform_response_allows_missing_usage_cost(self):
        """A response without cost data must remain usable."""
        raw_response = httpx.Response(200, json=_successful_response_body())

        response = OpenRouterResponsesAPIConfig().transform_response_api_response(
            model="openai/gpt-4o-mini",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )

        assert "llm_provider-x-litellm-response-cost" not in response._hidden_params["additional_headers"]

    def test_extra_body_usage_cannot_disable_openrouter_cost_tracking(self):
        """The final request merge must force usage.include to true."""
        from litellm.llms.custom_httpx.llm_http_handler import _ensure_openrouter_responses_usage

        request = {"usage": {"include": False, "detail": "all"}}
        _ensure_openrouter_responses_usage(request, OpenRouterResponsesAPIConfig())

        assert request["usage"] == {"include": True, "detail": "all"}


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
            "OpenRouter must be registered as a native Responses API provider to preserve reasoning.encrypted_content"
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
