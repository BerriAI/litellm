import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
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
def test_validate_environment_azure_openai_api_key_within_secret_str():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.api_key", None), \
         patch("litellm.azure_key", None), \
         patch("litellm.llms.azure.common_utils.get_secret_str") as mock_get_secret_str:
        # Configure the mock to return "test-api-key" when called with "AZURE_OPENAI_API_KEY"
        mock_get_secret_str.side_effect = (
            lambda key: "test-api-key" if key == "AZURE_OPENAI_API_KEY" else None
        )

        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )
        expected = {"api-key": "test-api-key"}

        assert result == expected

@pytest.mark.serial
def test_validate_environment_azure_api_key_within_secret_str():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.api_key", None), \
         patch("litellm.azure_key", None), \
         patch("litellm.llms.azure.common_utils.get_secret_str") as mock_get_secret_str:
        # Configure the mock to return None for "AZURE_OPENAI_API_KEY" and "test-api-key" for "AZURE_API_KEY"
        def mock_side_effect(key):
            if key == "AZURE_OPENAI_API_KEY":
                return None
            elif key == "AZURE_API_KEY":
                return "test-api-key"
            else:
                return None
        
        mock_get_secret_str.side_effect = mock_side_effect

        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )
        expected = {"api-key": "test-api-key"}

        assert result == expected
