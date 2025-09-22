import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock

from litellm.llms.azure.responses.o_series_transformation import (
    AzureOpenAIOSeriesResponsesAPIConfig,
)
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams


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
        temperature=0.7, max_output_tokens=1000, stream=False, top_p=0.9
    )

    # Test with drop_params=True
    mapped_params_with_drop = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=True,
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
        drop_params=False,
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
        max_output_tokens=1000, stream=False, top_p=0.9
    )

    # Should work fine even with drop_params=True
    mapped_params = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=True,
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
    import litellm
    from litellm.utils import ProviderConfigManager

    # Test O-series model selection
    o_series_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE, model="o_series/gpt-o1"
    )
    assert isinstance(o_series_config, AzureOpenAIOSeriesResponsesAPIConfig)

    # Test regular model selection
    regular_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE, model="gpt-4o"
    )
    assert isinstance(regular_config, AzureOpenAIResponsesAPIConfig)
    assert not isinstance(regular_config, AzureOpenAIOSeriesResponsesAPIConfig)

    # Test with no model specified (should default to regular)
    default_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE, model=None
    )
    assert isinstance(default_config, AzureOpenAIResponsesAPIConfig)
    assert not isinstance(default_config, AzureOpenAIOSeriesResponsesAPIConfig)


class TestAzureResponsesAPIConfig:
    def setup_method(self):
        self.config = AzureOpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_azure_get_complete_url_with_version_types(self):
        """Test Azure get_complete_url with different API version types"""
        base_url = "https://litellm8397336933.openai.azure.com"

        # Test with preview version - should use openai/v1/responses
        result_preview = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "preview"},
        )
        assert (
            result_preview
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=preview"
        )

        # Test with latest version - should use openai/v1/responses
        result_latest = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "latest"},
        )
        assert (
            result_latest
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=latest"
        )

        # Test with date-based version - should use openai/responses
        result_date = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "2025-01-01"},
        )
        assert (
            result_date
            == "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2025-01-01"
        )

    def test_azure_get_complete_url_with_default_api_version(self):
        """Test Azure get_complete_url uses default API version when none is provided"""
        from litellm.constants import AZURE_DEFAULT_RESPONSES_API_VERSION

        base_url = "https://litellm8397336933.openai.azure.com"

        # Test with no api_version provided - should use default
        result_no_version = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={},
        )
        expected_url = f"https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version={AZURE_DEFAULT_RESPONSES_API_VERSION}"
        assert result_no_version == expected_url

        # Test with empty litellm_params - should use default
        result_empty_params = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={},
        )
        assert result_empty_params == expected_url

        # Test with None api_version - should use default
        result_none_version = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": None},
        )
        assert result_none_version == expected_url

    def test_azure_cancel_response_api_request(self):
        """Test Azure cancel response API request transformation"""
        from litellm.types.router import GenericLiteLLMParams
        
        response_id = "resp_test123"
        api_base = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"
        litellm_params = GenericLiteLLMParams(api_version="2024-05-01-preview")
        headers = {"Authorization": "Bearer test-key"}
        
        url, data = self.config.transform_cancel_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        expected_url = "https://test.openai.azure.com/openai/responses/resp_test123/cancel?api-version=2024-05-01-preview"
        assert url == expected_url
        assert data == {}

    def test_azure_cancel_response_api_response(self):
        """Test Azure cancel response API response transformation"""
        from unittest.mock import Mock
        from litellm.types.llms.openai import ResponsesAPIResponse
        
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "resp_test123",
            "object": "response",
            "created_at": 1234567890,
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "status": "cancelled"
        }
        mock_response.text = "test response"
        mock_response.status_code = 200
        
        # Mock logging object
        mock_logging_obj = Mock()
        
        result = self.config.transform_cancel_response_api_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
        )
        
        assert isinstance(result, ResponsesAPIResponse)
        assert result.id == "resp_test123"