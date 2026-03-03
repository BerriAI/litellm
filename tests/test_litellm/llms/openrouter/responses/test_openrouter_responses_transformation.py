"""
Tests for OpenRouter Responses API configuration.

Validates that OpenRouter is registered as a native Responses API provider,
routing requests directly to https://openrouter.ai/api/v1/responses instead
of falling back to the chat completion bridge. This is required to preserve
reasoning.encrypted_content for multi-turn stateless workflows.

Related issue: https://github.com/BerriAI/litellm/issues/22189
"""

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

    def test_validate_environment_raises_without_key(self):
        """validate_environment should raise when no API key is available."""
        config = OpenRouterResponsesAPIConfig()
        from litellm.types.router import GenericLiteLLMParams

        try:
            config.validate_environment(
                headers={},
                model="openai/o4-mini",
                litellm_params=GenericLiteLLMParams(),
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
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
