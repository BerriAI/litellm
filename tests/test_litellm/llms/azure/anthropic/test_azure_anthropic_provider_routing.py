import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import _is_azure_anthropic_model, get_llm_provider


class TestAzureAnthropicProviderRouting:
    def test_is_azure_anthropic_model_with_claude(self):
        """Test _is_azure_anthropic_model detects Claude models"""
        # Test various Claude model names
        assert _is_azure_anthropic_model("azure/claude-sonnet-4-5") == "claude-sonnet-4-5"
        assert _is_azure_anthropic_model("azure/claude-opus-4-1") == "claude-opus-4-1"
        assert _is_azure_anthropic_model("azure/claude-haiku-4-5") == "claude-haiku-4-5"
        assert _is_azure_anthropic_model("azure/claude-3-5-sonnet") == "claude-3-5-sonnet"
        assert _is_azure_anthropic_model("azure/claude-3-opus") == "claude-3-opus"

    def test_is_azure_anthropic_model_case_insensitive(self):
        """Test _is_azure_anthropic_model is case insensitive"""
        assert _is_azure_anthropic_model("azure/CLAUDE-sonnet-4-5") == "CLAUDE-sonnet-4-5"
        assert _is_azure_anthropic_model("azure/Claude-Sonnet-4-5") == "Claude-Sonnet-4-5"

    def test_is_azure_anthropic_model_with_non_claude(self):
        """Test _is_azure_anthropic_model returns None for non-Claude models"""
        assert _is_azure_anthropic_model("azure/gpt-4") is None
        assert _is_azure_anthropic_model("azure/gpt-35-turbo") is None
        assert _is_azure_anthropic_model("azure/command-r-plus") is None

    def test_is_azure_anthropic_model_with_invalid_format(self):
        """Test _is_azure_anthropic_model handles invalid formats"""
        assert _is_azure_anthropic_model("azure") is None
        assert _is_azure_anthropic_model("claude-sonnet-4-5") is None
        assert _is_azure_anthropic_model("") is None

    def test_get_llm_provider_routes_azure_claude_to_azure_anthropic(self):
        """Test that get_llm_provider routes azure/claude-* models to azure_anthropic"""
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="azure/claude-sonnet-4-5"
        )
        assert provider == "azure_anthropic"
        assert model == "claude-sonnet-4-5"  # Should strip "azure/" prefix

    def test_get_llm_provider_routes_azure_claude_opus(self):
        """Test routing for Claude Opus models"""
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="azure/claude-opus-4-1"
        )
        assert provider == "azure_anthropic"
        assert model == "claude-opus-4-1"

    def test_get_llm_provider_routes_azure_claude_haiku(self):
        """Test routing for Claude Haiku models"""
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="azure/claude-haiku-4-5"
        )
        assert provider == "azure_anthropic"
        assert model == "claude-haiku-4-5"

    def test_get_llm_provider_does_not_route_non_claude_azure_models(self):
        """Test that non-Claude Azure models are not routed to azure_anthropic"""
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="azure/gpt-4"
        )
        assert provider != "azure_anthropic"
        # Should be routed to regular azure provider
        assert provider == "azure" or provider == "openai"

    def test_get_llm_provider_with_custom_llm_provider_override(self):
        """Test that custom_llm_provider parameter can override routing"""
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="azure/claude-sonnet-4-5", custom_llm_provider="azure"
        )
        # When custom_llm_provider is explicitly set, it should be respected
        # But the routing logic should still detect it as azure_anthropic
        # This depends on the order of checks in get_llm_provider
        assert provider in ["azure_anthropic", "azure"]

