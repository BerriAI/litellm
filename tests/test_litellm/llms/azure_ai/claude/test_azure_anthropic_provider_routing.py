import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import (
    _is_azure_claude_model,
    get_llm_provider,
)


class TestAzureAnthropicProviderRouting:
    def test_is_azure_claude_model_with_claude(self):
        """Test _is_azure_claude_model detects Claude models"""
        # Test various Claude model names
        assert _is_azure_claude_model("claude-sonnet-4-5") is True
        assert _is_azure_claude_model("claude-opus-4-1") is True
        assert _is_azure_claude_model("claude-haiku-4-5") is True
        assert _is_azure_claude_model("claude-3-5-sonnet") is True
        assert _is_azure_claude_model("claude-3-opus") is True

    def test_is_azure_claude_model_case_insensitive(self):
        """Test _is_azure_claude_model is case insensitive"""
        assert _is_azure_claude_model("CLAUDE-sonnet-4-5") is True
        assert _is_azure_claude_model("Claude-Sonnet-4-5") is True

    def test_is_azure_claude_model_with_non_claude(self):
        """Test _is_azure_claude_model returns False for non-Claude models"""
        assert _is_azure_claude_model("gpt-4") is False
        assert _is_azure_claude_model("gpt-35-turbo") is False
        assert _is_azure_claude_model("command-r-plus") is False

    def test_is_azure_claude_model_with_invalid_format(self):
        """Test _is_azure_claude_model handles invalid formats"""
        assert _is_azure_claude_model("") is False

    def test_get_llm_provider_routes_azure_ai_claude_to_azure_ai(self):
        """Test that azure_ai/claude-* models route through azure_ai"""
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="azure_ai/claude-sonnet-4-5"
        )
        assert provider == "azure_ai"
        assert model == "claude-sonnet-4-5"

    def test_get_llm_provider_does_not_route_non_claude_azure_models(self):
        """Test that non-Claude Azure models are not routed to azure_ai"""
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="azure/gpt-4"
        )
        # Should be routed to regular azure provider
        assert provider == "azure" or provider == "openai"

