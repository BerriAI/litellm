import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from unittest.mock import patch

import pytest

import litellm
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestAzureAnthropicProviderConfig:
    def test_get_provider_anthropic_messages_config_returns_azure_config(self):
        """Test that get_provider_anthropic_messages_config returns AzureAnthropicMessagesConfig for azure_anthropic provider"""
        from litellm.llms.azure.anthropic.messages_transformation import (
            AzureAnthropicMessagesConfig,
        )
        
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-sonnet-4-5",
            provider=LlmProviders.AZURE_ANTHROPIC,
        )
        
        assert config is not None
        assert isinstance(config, AzureAnthropicMessagesConfig)

    def test_get_provider_anthropic_messages_config_returns_anthropic_config_for_anthropic_provider(self):
        """Test that get_provider_anthropic_messages_config returns AnthropicMessagesConfig for anthropic provider"""
        from litellm.llms.azure.anthropic.messages_transformation import (
            AzureAnthropicMessagesConfig,
        )
        
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-sonnet-4-5",
            provider=LlmProviders.ANTHROPIC,
        )
        
        # Should return AnthropicMessagesConfig, not AzureAnthropicMessagesConfig
        assert config is not None
        assert not isinstance(config, AzureAnthropicMessagesConfig)
        assert isinstance(config, litellm.AnthropicMessagesConfig)

    def test_get_provider_chat_config_returns_azure_anthropic_config(self):
        """Test that get_provider_chat_config returns AzureAnthropicConfig for azure_anthropic provider"""
        from litellm.llms.azure.anthropic.transformation import AzureAnthropicConfig
        
        config = ProviderConfigManager.get_provider_chat_config(
            model="claude-sonnet-4-5",
            provider=LlmProviders.AZURE_ANTHROPIC,
        )
        
        assert config is not None
        assert isinstance(config, AzureAnthropicConfig)

