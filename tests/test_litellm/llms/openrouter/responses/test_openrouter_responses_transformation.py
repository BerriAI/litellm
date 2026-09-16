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
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.openrouter.responses.transformation import (
    OpenRouterResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
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


_RESPONSE_BODY = {
    "id": "resp_123",
    "object": "response",
    "created_at": 1234567890,
    "status": "completed",
    "model": "openrouter/anthropic/claude-3.5-sonnet",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
}


def _raw_response(body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/responses"),
    )


class TestOpenRouterResponsesAPICostTracking:
    """
    Regression for https://github.com/BerriAI/litellm/issues/38507.

    OpenRouter's Responses API path never requested `usage.include=true` and
    never read OpenRouter's returned `usage.cost`, so every `aresponses`
    request against an OpenRouter model missing from litellm's bundled
    pricing JSON logged `spend = 0` despite real token usage.
    """

    def test_transform_request_adds_usage_include(self):
        """usage.include=true is added so OpenRouter returns real cost."""
        config = OpenRouterResponsesAPIConfig()
        body = config.transform_responses_api_request(
            model="anthropic/claude-3.5-sonnet",
            input="hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["usage"] == {"include": True}

    def test_transform_request_preserves_caller_usage(self):
        """A usage value the caller already set is left untouched."""
        config = OpenRouterResponsesAPIConfig()
        body = config.transform_responses_api_request(
            model="anthropic/claude-3.5-sonnet",
            input="hello",
            response_api_optional_request_params={"usage": {"include": False}},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["usage"] == {"include": False}

    def test_transform_response_extracts_openrouter_cost(self):
        """usage.cost from the response body reaches the cost calculator."""
        config = OpenRouterResponsesAPIConfig()
        body = {**_RESPONSE_BODY, "usage": {**_RESPONSE_BODY["usage"], "cost": 0.001234}}

        response = config.transform_response_api_response(
            model="anthropic/claude-3.5-sonnet",
            raw_response=_raw_response(body),
            logging_obj=MagicMock(),
        )

        additional_headers = response._hidden_params["additional_headers"]
        assert additional_headers["llm_provider-x-litellm-response-cost"] == 0.001234
        assert get_response_cost_from_hidden_params(response._hidden_params) == 0.001234

    def test_transform_response_no_cost_in_body(self):
        """No usage.cost -> no cost header, and nothing raised."""
        config = OpenRouterResponsesAPIConfig()

        response = config.transform_response_api_response(
            model="anthropic/claude-3.5-sonnet",
            raw_response=_raw_response(_RESPONSE_BODY),
            logging_obj=MagicMock(),
        )

        additional_headers = response._hidden_params.get("additional_headers", {})
        assert "llm_provider-x-litellm-response-cost" not in additional_headers
        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    def test_transform_response_no_usage_object(self):
        """A response with no usage object at all -> no cost header, no crash."""
        config = OpenRouterResponsesAPIConfig()
        body = {k: v for k, v in _RESPONSE_BODY.items() if k != "usage"}

        response = config.transform_response_api_response(
            model="anthropic/claude-3.5-sonnet",
            raw_response=_raw_response(body),
            logging_obj=MagicMock(),
        )

        additional_headers = response._hidden_params.get("additional_headers", {})
        assert "llm_provider-x-litellm-response-cost" not in additional_headers

    def test_transform_response_preserves_response_headers(self):
        """The cost key is merged in, not written over the parent's headers."""
        config = OpenRouterResponsesAPIConfig()
        body = {**_RESPONSE_BODY, "usage": {**_RESPONSE_BODY["usage"], "cost": 0.5}}
        raw = _raw_response(body)
        raw.headers["x-ratelimit-remaining"] = "42"

        response = config.transform_response_api_response(
            model="anthropic/claude-3.5-sonnet",
            raw_response=raw,
            logging_obj=MagicMock(),
        )

        additional_headers = response._hidden_params["additional_headers"]
        assert additional_headers["llm_provider-x-litellm-response-cost"] == 0.5
        assert additional_headers["llm_provider-x-ratelimit-remaining"] == "42"
