import json
import os
import sys
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.common_utils import initialize_azure_sdk_client


# Mock the necessary dependencies
@pytest.fixture
def setup_mocks():
    with patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_entrata_id"
    ) as mock_entrata_token, patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_username_password"
    ) as mock_username_password_token, patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_oidc"
    ) as mock_oidc_token, patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_provider"
    ) as mock_token_provider, patch(
        "litellm.llms.azure.common_utils.litellm"
    ) as mock_litellm, patch(
        "litellm.llms.azure.common_utils.verbose_logger"
    ) as mock_logger, patch(
        "litellm.llms.azure.common_utils.select_azure_base_url_or_endpoint"
    ) as mock_select_url:

        # Configure mocks
        mock_litellm.AZURE_DEFAULT_API_VERSION = "2023-05-15"
        mock_litellm.enable_azure_ad_token_refresh = False

        mock_entrata_token.return_value = lambda: "mock-entrata-token"
        mock_username_password_token.return_value = (
            lambda: "mock-username-password-token"
        )
        mock_oidc_token.return_value = "mock-oidc-token"
        mock_token_provider.return_value = lambda: "mock-default-token"

        mock_select_url.side_effect = lambda params: params

        yield {
            "entrata_token": mock_entrata_token,
            "username_password_token": mock_username_password_token,
            "oidc_token": mock_oidc_token,
            "token_provider": mock_token_provider,
            "litellm": mock_litellm,
            "logger": mock_logger,
            "select_url": mock_select_url,
        }


def test_initialize_with_api_key(setup_mocks):
    # Test with api_key provided
    result = initialize_azure_sdk_client(
        litellm_params={},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
    )

    # Verify expected result
    assert result["api_key"] == "test-api-key"
    assert result["azure_endpoint"] == "https://test.openai.azure.com"
    assert result["api_version"] == "2023-06-01"
    assert "azure_ad_token" in result
    assert result["azure_ad_token"] is None


def test_initialize_with_tenant_credentials(setup_mocks):
    # Test with tenant_id, client_id, and client_secret provided
    result = initialize_azure_sdk_client(
        litellm_params={
            "tenant_id": "test-tenant-id",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_from_entrata_id was called
    setup_mocks["entrata_token"].assert_called_once_with(
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
    )

    # Verify expected result
    assert result["api_key"] is None
    assert result["azure_endpoint"] == "https://test.openai.azure.com"
    assert "azure_ad_token_provider" in result


def test_initialize_with_username_password(setup_mocks):
    # Test with azure_username, azure_password, and client_id provided
    result = initialize_azure_sdk_client(
        litellm_params={
            "azure_username": "test-username",
            "azure_password": "test-password",
            "client_id": "test-client-id",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_from_username_password was called
    setup_mocks["username_password_token"].assert_called_once_with(
        azure_username="test-username",
        azure_password="test-password",
        client_id="test-client-id",
    )

    # Verify expected result
    assert "azure_ad_token_provider" in result


def test_initialize_with_oidc_token(setup_mocks):
    # Test with azure_ad_token that starts with "oidc/"
    result = initialize_azure_sdk_client(
        litellm_params={"azure_ad_token": "oidc/test-token"},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_from_oidc was called
    setup_mocks["oidc_token"].assert_called_once_with("oidc/test-token")

    # Verify expected result
    assert result["azure_ad_token"] == "mock-oidc-token"


def test_initialize_with_enable_token_refresh(setup_mocks):
    # Enable token refresh
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True

    # Test with token refresh enabled
    result = initialize_azure_sdk_client(
        litellm_params={},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_provider was called
    setup_mocks["token_provider"].assert_called_once()

    # Verify expected result
    assert "azure_ad_token_provider" in result


def test_initialize_with_token_refresh_error(setup_mocks):
    # Enable token refresh but make it raise an error
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True
    setup_mocks["token_provider"].side_effect = ValueError("Token provider error")

    # Test with token refresh enabled but raising error
    result = initialize_azure_sdk_client(
        litellm_params={},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify error was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Azure AD Token Provider could not be used."
    )


def test_api_version_from_env_var(setup_mocks):
    # Test api_version from environment variable
    with patch.dict(os.environ, {"AZURE_API_VERSION": "2023-07-01"}):
        result = initialize_azure_sdk_client(
            litellm_params={},
            api_key="test-api-key",
            api_base="https://test.openai.azure.com",
            model_name="gpt-4",
            api_version=None,
        )

    # Verify expected result
    assert result["api_version"] == "2023-07-01"


def test_select_azure_base_url_called(setup_mocks):
    # Test that select_azure_base_url_or_endpoint is called
    result = initialize_azure_sdk_client(
        litellm_params={},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
    )

    # Verify that select_azure_base_url_or_endpoint was called
    setup_mocks["select_url"].assert_called_once()
