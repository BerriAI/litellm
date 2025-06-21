from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from unittest.mock import AsyncMock, MagicMock, patch


def test_validate_environment_api_key_within_litellm_params():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    litellm_params = GenericLiteLLMParams(api_key="test-api-key")

    result = azure_openai_responses_apiconfig.validate_environment(
        headers={}, model="", litellm_params=litellm_params
    )

    expected = {"api-key": "test-api-key"}

    assert result == expected


def test_validate_environment_api_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.api_key", "test-api-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-api-key"}

        assert result == expected


def test_validate_environment_azure_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.azure_key", "test-azure-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-azure-key"}

        assert result == expected


def test_validate_environment_azure_openai_api_key_within_secret_str():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch(
        "litellm.llms.azure.responses.transformation.get_secret_str"
    ) as mock_get_secret_str:
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


def test_validate_environment_azure_api_key_within_secret_str():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch(
        "litellm.llms.azure.responses.transformation.get_secret_str"
    ) as mock_get_secret_str:
        # Configure the mock to return "test-api-key" when called with "AZURE_API_KEY"
        mock_get_secret_str.side_effect = (
            lambda key: "test-api-key" if key == "AZURE_API_KEY" else None
        )

        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )
        expected = {"api-key": "test-api-key"}

        assert result == expected


def test_validate_environment_get_azure_ad_token():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch(
        "litellm.llms.azure.responses.transformation.get_azure_ad_token"
    ) as mock_get_azure_ad_token:
        mock_get_azure_ad_token.side_effect = lambda key: "test-azure-ad-token"

        litellm_params = GenericLiteLLMParams(azure_ad_token="test-azure-ad-token")
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )
        expected = {"Authorization": "Bearer test-azure-ad-token"}

        assert result == expected
