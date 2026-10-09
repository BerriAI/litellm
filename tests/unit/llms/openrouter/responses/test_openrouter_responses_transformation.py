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
from litellm.litellm_core_utils.core_helpers import RESPONSE_COST_HEADER
from litellm.llms.openrouter.responses.transformation import (
    OpenRouterResponsesAPIConfig,
)
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
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


@pytest.mark.parametrize("cost", [1.23e-5, None])
def test_openrouter_responses_returns_provider_reported_cost(cost):
    usage: dict[str, object] = {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    if cost is not None:
        usage["cost"] = cost
    response = OpenRouterResponsesAPIConfig().transform_response_api_response(
        model="openrouter/unmapped-model",
        raw_response=httpx.Response(
            200,
            json={
                "id": "resp_test",
                "created_at": 1,
                "model": "unmapped-model",
                "object": "response",
                "output": [],
                "status": "completed",
                "usage": usage,
            },
        ),
        logging_obj=MagicMock(),
    )

    assert get_response_cost_from_hidden_params(response.hidden_params) == cost


@pytest.mark.parametrize(
    "event_type",
    ["response.completed", "response.incomplete", "response.failed"],
)
def test_openrouter_terminal_stream_events_return_provider_reported_cost(event_type):
    event = OpenRouterResponsesAPIConfig().transform_streaming_response(
        model="openrouter/unmapped-model",
        parsed_chunk={
            "type": event_type,
            "response": {
                "id": "resp_test",
                "created_at": 1,
                "model": "unmapped-model",
                "object": "response",
                "output": [],
                "status": "completed",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "cost": 1.23e-5,
                },
            },
        },
        logging_obj=MagicMock(),
    )
    assert isinstance(
        event,
        (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent),
    )

    assert event.response.hidden_params["additional_headers"][RESPONSE_COST_HEADER] == 1.23e-5
