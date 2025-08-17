import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure.responses.o_series_transformation import AzureOpenAIOSeriesResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


@pytest.mark.serial
def test_validate_environment_api_key_within_litellm_params():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    litellm_params = GenericLiteLLMParams(api_key="test-api-key")

    result = azure_openai_responses_apiconfig.validate_environment(
        headers={}, model="", litellm_params=litellm_params
    )

    expected = {"api-key": "test-api-key"}

    assert result == expected

@pytest.mark.serial
def test_validate_environment_api_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.api_key", "test-api-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-api-key"}

        assert result == expected

@pytest.mark.serial
def test_validate_environment_azure_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.azure_key", "test-azure-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-azure-key"}

        assert result == expected

@pytest.mark.serial
def test_validate_environment_azure_key_within_headers():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    headers = {"api-key": "test-api-key-from-headers"}
    litellm_params = GenericLiteLLMParams()

    result = azure_openai_responses_apiconfig.validate_environment(
        headers=headers, model="", litellm_params=litellm_params
    )

    expected = {"api-key": "test-api-key-from-headers"}

    assert result == expected


@pytest.mark.serial
def test_get_complete_url():
    """
    Test the get_complete_url function
    """
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    api_base = "https://litellm8397336933.openai.azure.com"
    litellm_params = {"api_version": "2024-05-01-preview"}

    result = azure_openai_responses_apiconfig.get_complete_url(
        api_base=api_base, litellm_params=litellm_params
    )

    expected = "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

    assert result == expected


@pytest.mark.serial
def test_azure_o_series_responses_api_supported_params():
    """Test that Azure OpenAI O-series responses API excludes temperature from supported parameters."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()
    supported_params = config.get_supported_openai_params("o_series/gpt-o1")
    
    # Temperature should not be in supported params for O-series models
    assert "temperature" not in supported_params
    
    # Other parameters should still be supported
    assert "input" in supported_params
    assert "max_output_tokens" in supported_params
    assert "stream" in supported_params
    assert "top_p" in supported_params


@pytest.mark.serial
def test_azure_o_series_responses_api_drop_temperature_param():
    """Test that temperature parameter is dropped when drop_params is True for O-series models."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()
    
    # Create request params with temperature
    request_params = ResponsesAPIOptionalRequestParams(
        temperature=0.7,
        max_output_tokens=1000,
        stream=False,
        top_p=0.9
    )
    
    # Test with drop_params=True
    mapped_params_with_drop = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=True
    )
    
    # Temperature should be dropped
    assert "temperature" not in mapped_params_with_drop
    # Other params should remain
    assert mapped_params_with_drop["max_output_tokens"] == 1000
    assert mapped_params_with_drop["top_p"] == 0.9
    
    # Test with drop_params=False
    mapped_params_without_drop = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=False
    )
    
    # Temperature should still be present when drop_params=False
    assert mapped_params_without_drop["temperature"] == 0.7
    assert mapped_params_without_drop["max_output_tokens"] == 1000
    assert mapped_params_without_drop["top_p"] == 0.9


@pytest.mark.serial
def test_azure_o_series_responses_api_drop_params_no_temperature():
    """Test that map_openai_params works correctly when temperature is not present for O-series models."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()
    
    # Create request params without temperature
    request_params = ResponsesAPIOptionalRequestParams(
        max_output_tokens=1000,
        stream=False,
        top_p=0.9
    )
    
    # Should work fine even with drop_params=True
    mapped_params = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=True
    )
    
    assert "temperature" not in mapped_params
    assert mapped_params["max_output_tokens"] == 1000
    assert mapped_params["top_p"] == 0.9


@pytest.mark.serial
def test_azure_regular_responses_api_supports_temperature():
    """Test that regular Azure OpenAI responses API (non-O-series) supports temperature parameter."""
    config = AzureOpenAIResponsesAPIConfig()
    supported_params = config.get_supported_openai_params("gpt-4o")
    
    # Regular Azure models should support temperature
    assert "temperature" in supported_params
    
    # Other parameters should still be supported
    assert "input" in supported_params
    assert "max_output_tokens" in supported_params
    assert "stream" in supported_params
    assert "top_p" in supported_params


@pytest.mark.serial
def test_o_series_model_detection():
    """Test that the O-series configuration correctly identifies O-series models."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()
    
    # Test explicit o_series naming
    assert config.is_o_series_model("o_series/gpt-o1") == True
    assert config.is_o_series_model("azure/o_series/gpt-o3") == True
    
    # Test regular models
    assert config.is_o_series_model("gpt-4o") == False
    assert config.is_o_series_model("gpt-3.5-turbo") == False


@pytest.mark.serial
def test_provider_config_manager_o_series_selection():
    """Test that ProviderConfigManager returns the correct config for O-series vs regular models."""
    from litellm.utils import ProviderConfigManager
    import litellm
    
    # Test O-series model selection
    o_series_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE,
        model="o_series/gpt-o1"
    )
    assert isinstance(o_series_config, AzureOpenAIOSeriesResponsesAPIConfig)
    
    # Test regular model selection
    regular_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE,
        model="gpt-4o"
    )
    assert isinstance(regular_config, AzureOpenAIResponsesAPIConfig)
    assert not isinstance(regular_config, AzureOpenAIOSeriesResponsesAPIConfig)
    
    # Test with no model specified (should default to regular)
    default_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE,
        model=None
    )
    assert isinstance(default_config, AzureOpenAIResponsesAPIConfig)
    assert not isinstance(default_config, AzureOpenAIOSeriesResponsesAPIConfig)
