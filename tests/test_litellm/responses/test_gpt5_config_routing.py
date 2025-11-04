"""Tests for GPT-5 Responses API config routing."""

import pytest

import litellm
from litellm import LlmProviders
from litellm.llms.azure.responses.gpt_5_transformation import (
    AzureOpenAIGPT5ResponsesAPIConfig,
)
from litellm.llms.azure.responses.o_series_transformation import (
    AzureOpenAIOSeriesResponsesAPIConfig,
)
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.openai.responses.gpt_5_transformation import (
    OpenAIGPT5ResponsesAPIConfig,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.utils import ProviderConfigManager


class TestOpenAIConfigRouting:
    """Test that OpenAI GPT-5 models route to the correct config."""

    def test_gpt5_routes_to_gpt5_config(self):
        """Test that gpt-5 routes to GPT-5 config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENAI, model="gpt-5"
        )
        assert isinstance(config, OpenAIGPT5ResponsesAPIConfig)

    def test_gpt5_mini_routes_to_gpt5_config(self):
        """Test that gpt-5-mini routes to GPT-5 config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENAI, model="gpt-5-mini"
        )
        assert isinstance(config, OpenAIGPT5ResponsesAPIConfig)

    def test_gpt5_codex_routes_to_gpt5_config(self):
        """Test that gpt-5-codex routes to GPT-5 config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENAI, model="gpt-5-codex"
        )
        assert isinstance(config, OpenAIGPT5ResponsesAPIConfig)

    def test_gpt4_routes_to_base_config(self):
        """Test that gpt-4 routes to base OpenAI config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENAI, model="gpt-4"
        )
        assert isinstance(config, OpenAIResponsesAPIConfig)
        assert not isinstance(config, OpenAIGPT5ResponsesAPIConfig)

    def test_gpt4o_routes_to_base_config(self):
        """Test that gpt-4o routes to base OpenAI config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENAI, model="gpt-4o"
        )
        assert isinstance(config, OpenAIResponsesAPIConfig)
        assert not isinstance(config, OpenAIGPT5ResponsesAPIConfig)


class TestAzureConfigRouting:
    """Test that Azure GPT-5 models route to the correct config."""

    def test_azure_gpt5_routes_to_gpt5_config(self):
        """Test that Azure gpt-5 routes to GPT-5 config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="gpt-5"
        )
        assert isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)

    def test_azure_gpt5_mini_routes_to_gpt5_config(self):
        """Test that Azure gpt-5-mini routes to GPT-5 config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="gpt-5-mini"
        )
        assert isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)

    def test_azure_gpt5_codex_routes_to_gpt5_config(self):
        """Test that Azure gpt-5-codex routes to GPT-5 config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="gpt-5-codex"
        )
        assert isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)

    def test_azure_gpt4_routes_to_base_config(self):
        """Test that Azure gpt-4 routes to base Azure config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="gpt-4"
        )
        assert isinstance(config, AzureOpenAIResponsesAPIConfig)
        assert not isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)
        assert not isinstance(config, AzureOpenAIOSeriesResponsesAPIConfig)

    def test_azure_gpt4o_routes_to_base_config(self):
        """Test that Azure gpt-4o routes to base Azure config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="gpt-4o"
        )
        assert isinstance(config, AzureOpenAIResponsesAPIConfig)
        assert not isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)
        assert not isinstance(config, AzureOpenAIOSeriesResponsesAPIConfig)

    def test_azure_o1_routes_correctly(self):
        """Test that Azure o1 models route correctly and not to GPT-5 config."""
        # Note: o1 models might route to O-series config if supports_reasoning
        # detects them, otherwise they route to base config. The key thing is
        # they should NOT route to GPT-5 config.
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="o1"
        )
        assert not isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)

    def test_azure_o1_mini_routes_correctly(self):
        """Test that Azure o1-mini models route correctly and not to GPT-5 config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="o1-mini"
        )
        assert not isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)


class TestGPT5ConfigPriorityOverOSeries:
    """Test that GPT-5 config takes precedence over O-series detection."""

    def test_gpt5_not_detected_as_o_series(self):
        """Test that gpt-5 is not incorrectly detected as O-series.
        
        This was the original bug - GPT-5 was being detected as O-series
        due to substring matching on 'o' in 'gpt-5'.
        """
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="gpt-5"
        )
        # Should be GPT-5 config, not O-series
        assert isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)
        assert not isinstance(config, AzureOpenAIOSeriesResponsesAPIConfig)

    def test_gpt5_mini_not_detected_as_o_series(self):
        """Test that gpt-5-mini is not incorrectly detected as O-series."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE, model="gpt-5-mini"
        )
        assert isinstance(config, AzureOpenAIGPT5ResponsesAPIConfig)
        assert not isinstance(config, AzureOpenAIOSeriesResponsesAPIConfig)
