"""
Tests for OpenAI-like Responses API support in the JSON provider system.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)


class TestSimpleProviderConfigSupportedEndpoints:
    """Test the supported_endpoints field on SimpleProviderConfig."""

    def test_default_supported_endpoints(self):
        """supported_endpoints defaults to [] (chat always enabled, nothing else)"""
        from litellm.llms.openai_like.json_loader import SimpleProviderConfig

        config = SimpleProviderConfig("test", {"base_url": "https://example.com", "api_key_env": "TEST_KEY"})
        assert config.supported_endpoints == []

    def test_custom_supported_endpoints(self):
        """supported_endpoints can be set explicitly"""
        from litellm.llms.openai_like.json_loader import SimpleProviderConfig

        config = SimpleProviderConfig(
            "test",
            {
                "base_url": "https://example.com",
                "api_key_env": "TEST_KEY",
                "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
            },
        )
        assert "/v1/responses" in config.supported_endpoints
        assert "/v1/chat/completions" in config.supported_endpoints

    def test_responses_only_endpoint(self):
        """A provider can support only responses"""
        from litellm.llms.openai_like.json_loader import SimpleProviderConfig

        config = SimpleProviderConfig(
            "test",
            {
                "base_url": "https://example.com",
                "api_key_env": "TEST_KEY",
                "supported_endpoints": ["/v1/responses"],
            },
        )
        assert config.supported_endpoints == ["/v1/responses"]


class TestJSONProviderRegistryResponsesAPI:
    """Test supports_responses_api on JSONProviderRegistry."""

    def test_existing_provider_no_responses(self):
        """Existing providers without supported_endpoints don't support responses"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # publicai has no supported_endpoints in JSON, defaults to []
        assert JSONProviderRegistry.supports_responses_api("publicai") is False

    def test_nonexistent_provider(self):
        """Non-existent provider returns False"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("nonexistent_provider_xyz") is False

    def test_provider_with_responses_endpoint(self):
        """A provider with /v1/responses in supported_endpoints returns True"""
        from litellm.llms.openai_like.json_loader import (
            JSONProviderRegistry,
            SimpleProviderConfig,
        )

        # Temporarily inject a test provider
        test_config = SimpleProviderConfig(
            "test_responses_provider",
            {
                "base_url": "https://test.example.com",
                "api_key_env": "TEST_API_KEY",
                "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
            },
        )
        JSONProviderRegistry._providers["test_responses_provider"] = test_config
        try:
            assert JSONProviderRegistry.supports_responses_api("test_responses_provider") is True
        finally:
            del JSONProviderRegistry._providers["test_responses_provider"]


class TestCreateResponsesConfigClass:
    """Test dynamic responses config class generation."""

    def _make_test_provider(self):
        from litellm.llms.openai_like.json_loader import SimpleProviderConfig

        return SimpleProviderConfig(
            "test_resp",
            {
                "base_url": "https://api.testresp.com/v1",
                "api_key_env": "TEST_RESP_API_KEY",
                "api_base_env": "TEST_RESP_API_BASE",
                "supported_endpoints": ["/v1/responses"],
            },
        )

    def test_generated_class_custom_llm_provider(self):
        """Generated class returns the provider slug"""
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()
        assert config.custom_llm_provider == "test_resp"

    def test_generated_class_get_complete_url(self):
        """Generated class builds correct responses URL"""
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://api.testresp.com/v1/responses"

    def test_generated_class_get_complete_url_with_override(self):
        """api_base override takes precedence"""
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()

        url = config.get_complete_url(api_base="https://custom.api.com/v1", litellm_params={})
        assert url == "https://custom.api.com/v1/responses"

    def test_generated_class_get_complete_url_strips_trailing_slash(self):
        """Trailing slashes are stripped from base URL"""
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()

        url = config.get_complete_url(api_base="https://custom.api.com/v1/", litellm_params={})
        assert url == "https://custom.api.com/v1/responses"

    def test_generated_class_validate_environment(self):
        """validate_environment sets Authorization header from env"""
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()

        with patch(
            "litellm.llms.openai_like.dynamic_config.get_secret_str",
            return_value="sk-test-key-123",
        ):
            headers = config.validate_environment(headers={}, model="test-model", litellm_params=None)
        assert headers["Authorization"] == "Bearer sk-test-key-123"

    def test_generated_class_validate_environment_litellm_params_override(self):
        """api_key from litellm_params takes precedence over env"""
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )
        from litellm.types.router import GenericLiteLLMParams

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()

        litellm_params = GenericLiteLLMParams(api_key="sk-override-key")
        headers = config.validate_environment(
            headers={}, model="test-model", litellm_params=litellm_params
        )
        assert headers["Authorization"] == "Bearer sk-override-key"

    def test_generated_class_inherits_openai_responses_methods(self):
        """Generated class inherits OpenAI Responses API transformation methods"""
        from litellm.llms.openai.responses.transformation import (
            OpenAIResponsesAPIConfig,
        )
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()

        # Should have inherited methods from OpenAIResponsesAPIConfig
        assert hasattr(config, "get_supported_openai_params")
        assert hasattr(config, "map_openai_params")
        assert hasattr(config, "transform_responses_api_request")
        assert hasattr(config, "transform_response_api_response")
        assert hasattr(config, "transform_streaming_response")

        # Verify inheritance chain
        assert isinstance(config, OpenAIResponsesAPIConfig)

    def test_generated_class_get_complete_url_uses_api_base_env(self):
        """get_complete_url falls back to api_base_env when api_base is None"""
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )

        provider = self._make_test_provider()
        config_cls = create_responses_config_class(provider)
        config = config_cls()

        with patch(
            "litellm.llms.openai_like.dynamic_config.get_secret_str",
            return_value="https://env-override.example.com/v1",
        ):
            url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://env-override.example.com/v1/responses"


class TestProviderConfigManagerResponsesAPI:
    """Test that ProviderConfigManager integrates JSON responses providers."""

    def test_json_provider_with_responses_returns_config(self):
        """A JSON provider with /v1/responses returns a responses config"""
        from litellm.llms.openai_like.json_loader import (
            JSONProviderRegistry,
            SimpleProviderConfig,
        )
        from litellm.utils import ProviderConfigManager

        test_config = SimpleProviderConfig(
            "test_pcm_resp",
            {
                "base_url": "https://api.testpcm.com/v1",
                "api_key_env": "TEST_PCM_KEY",
                "supported_endpoints": ["/v1/responses"],
            },
        )
        JSONProviderRegistry._providers["test_pcm_resp"] = test_config
        try:
            config = ProviderConfigManager.get_provider_responses_api_config(
                provider="test_pcm_resp",
                model="some-model",
            )
            assert config is not None
            assert config.custom_llm_provider == "test_pcm_resp"
        finally:
            del JSONProviderRegistry._providers["test_pcm_resp"]

    def test_json_provider_without_responses_returns_none(self):
        """A JSON provider without /v1/responses returns None"""
        from litellm.utils import ProviderConfigManager

        # publicai only supports chat completions
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="publicai",
            model="some-model",
        )
        assert config is None

    def test_unknown_provider_returns_none(self):
        """A completely unknown provider returns None"""
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="totally_unknown_provider_xyz",
            model="some-model",
        )
        assert config is None

    def test_standard_providers_still_work(self):
        """Existing enum-based providers still resolve correctly"""
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENAI,
            model="gpt-4o",
        )
        assert config is not None

    def test_standard_provider_as_string_still_works(self):
        """Passing 'openai' as a string also works"""
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="openai",
            model="gpt-4o",
        )
        assert config is not None

    def test_python_class_takes_priority_over_json(self):
        """If a provider has both a Python class and JSON config, Python wins"""
        from litellm.llms.openai_like.json_loader import (
            JSONProviderRegistry,
            SimpleProviderConfig,
        )
        from litellm.llms.perplexity.responses.transformation import (
            PerplexityResponsesConfig,
        )
        from litellm.utils import ProviderConfigManager

        # Inject perplexity into JSON registry with responses support
        test_config = SimpleProviderConfig(
            "perplexity",
            {
                "base_url": "https://api.perplexity.ai",
                "api_key_env": "PERPLEXITY_API_KEY",
                "supported_endpoints": ["/v1/responses"],
            },
        )
        JSONProviderRegistry._providers["perplexity"] = test_config
        try:
            config = ProviderConfigManager.get_provider_responses_api_config(
                provider="perplexity",
                model="some-model",
            )
            # Should be the Python class, not the JSON-generated one
            assert isinstance(config, PerplexityResponsesConfig)
        finally:
            del JSONProviderRegistry._providers["perplexity"]
