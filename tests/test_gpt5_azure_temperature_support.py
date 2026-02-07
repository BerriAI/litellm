"""
Test that Azure GPT-5 models support temperature parameter in Responses API.
"""

import pytest
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders


def test_azure_gpt5_supports_temperature():
    """Test that Azure GPT-5 uses the correct config that supports temperature."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        provider=LlmProviders.AZURE,
        model="gpt-5"
    )
    
    # Should use AzureOpenAIResponsesAPIConfig, not AzureOpenAIOSeriesResponsesAPIConfig
    assert type(config).__name__ == "AzureOpenAIResponsesAPIConfig"
    
    # Should support temperature parameter
    supported_params = config.get_supported_openai_params("gpt-5")
    assert "temperature" in supported_params, "Azure GPT-5 should support temperature parameter"


def test_azure_o_series_does_not_support_temperature():
    """Test that Azure O-series models still use the correct O-series config."""
    test_models = ["o1", "o1-preview", "o3"]
    
    for model in test_models:
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE,
            model=model
        )
        
        # Should use AzureOpenAIOSeriesResponsesAPIConfig
        assert type(config).__name__ == "AzureOpenAIOSeriesResponsesAPIConfig", \
            f"Azure {model} should use O-series config"
        
        # Should NOT support temperature parameter
        supported_params = config.get_supported_openai_params(model)
        assert "temperature" not in supported_params, \
            f"Azure {model} should NOT support temperature parameter"


def test_openai_gpt5_supports_temperature():
    """Test that OpenAI GPT-5 supports temperature parameter."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        provider=LlmProviders.OPENAI,
        model="gpt-5"
    )
    
    # Should use OpenAIResponsesAPIConfig
    assert type(config).__name__ == "OpenAIResponsesAPIConfig"
    
    # Should support temperature parameter
    supported_params = config.get_supported_openai_params("gpt-5")
    assert "temperature" in supported_params, "OpenAI GPT-5 should support temperature parameter"


def test_azure_gpt5_variants_support_temperature():
    """Test that various GPT-5 model name variants support temperature."""
    gpt5_variants = ["gpt-5", "gpt-5-turbo", "GPT-5", "azure/gpt-5"]
    
    for model in gpt5_variants:
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE,
            model=model
        )
        
        # All GPT-5 variants should use the base config, not O-series config
        assert type(config).__name__ == "AzureOpenAIResponsesAPIConfig", \
            f"Model '{model}' should not use O-series config"
        
        # All should support temperature
        supported_params = config.get_supported_openai_params(model)
        assert "temperature" in supported_params, \
            f"Model '{model}' should support temperature parameter"


def test_azure_gpt_models_support_temperature():
    """Test that all GPT models (gpt-3.5, gpt-4, gpt-5, etc.) support temperature."""
    gpt_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-5"]
    
    for model in gpt_models:
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.AZURE,
            model=model
        )
        
        # All GPT models should use the base config, not O-series config
        assert type(config).__name__ == "AzureOpenAIResponsesAPIConfig", \
            f"Model '{model}' should not use O-series config"
        
        # All should support temperature
        supported_params = config.get_supported_openai_params(model)
        assert "temperature" in supported_params, \
            f"Model '{model}' should support temperature parameter"
