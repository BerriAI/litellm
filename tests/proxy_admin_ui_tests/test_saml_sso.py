"""
Tests for SAML 2.0 SSO authentication
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)

from fastapi import Request

import litellm
from litellm.proxy.auth.saml_handler import SAMLAuthenticationHandler
from litellm.proxy.management_endpoints.types import CustomOpenID


@pytest.fixture
def mock_saml_env_vars(monkeypatch):
    """Set up mock SAML environment variables"""
    monkeypatch.setenv("SAML_IDP_ENTITY_ID", "https://idp.example.com")
    monkeypatch.setenv("SAML_IDP_SSO_URL", "https://idp.example.com/sso")
    monkeypatch.setenv(
        "SAML_IDP_X509_CERT",
        "MIICXDCCAcWgAwIBAgIBADANBgkqhkiG9w0BAQ0FADBLMQswCQYDVQQGEwJ1czELMAkGA1UECAwCQ0ExFjAUBgNVBAoMDU9uZUxvZ2luIEluYy4xFzAVBgNVBAMMDnNwLmV4YW1wbGUuY29tMB4XDTIzMDEwMTAwMDAwMFoXDTI4MDEwMTAwMDAwMFowSzELMAkGA1UEBhMCdXMxCzAJBgNVBAgMAkNBMRYwFAYDVQQKDA1PbmVMb2dpbiBJbmMuMRcwFQYDVQQDDA5zcC5leGFtcGxlLmNvbTCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAtlE7p7cPpKMDnDnF2suDmQN/iDFALTpBwlT4RmlqW8vLpv6H",
    )
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")
    monkeypatch.setenv("SAML_USER_ID_ATTRIBUTE", "email")
    monkeypatch.setenv("SAML_USER_EMAIL_ATTRIBUTE", "email")
    monkeypatch.setenv("SAML_USER_FIRST_NAME_ATTRIBUTE", "firstName")
    monkeypatch.setenv("SAML_USER_LAST_NAME_ATTRIBUTE", "lastName")


def test_should_use_saml_handler_enabled(mock_saml_env_vars):
    """Test that SAML handler is enabled when environment variables are set"""
    assert SAMLAuthenticationHandler.should_use_saml_handler() is True


def test_should_use_saml_handler_disabled():
    """Test that SAML handler is disabled when environment variables are not set"""
    # Clear all SAML env vars
    for key in [
        "SAML_IDP_ENTITY_ID",
        "SAML_IDP_SSO_URL",
        "SAML_IDP_X509_CERT",
    ]:
        if key in os.environ:
            del os.environ[key]

    assert SAMLAuthenticationHandler.should_use_saml_handler() is False


@patch("litellm.proxy.auth.saml_handler.OneLogin_Saml2_Auth")
def test_get_login_url(mock_saml_auth, mock_saml_env_vars):
    """Test generating SAML login URL"""
    # Mock the SAML auth object
    mock_auth_instance = MagicMock()
    mock_auth_instance.login.return_value = "https://idp.example.com/sso?SAMLRequest=..."
    mock_saml_auth.return_value = mock_auth_instance

    # Create a mock request
    mock_request = MagicMock()
    mock_request.url.scheme = "https"
    mock_request.url.hostname = "proxy.example.com"
    mock_request.url.port = 443
    mock_request.url.path = "/sso/key/generate"
    mock_request.query_params = {}

    login_url = SAMLAuthenticationHandler.get_login_url(mock_request)

    # Verify that login was called and URL returned
    assert login_url is not None
    assert "https://idp.example.com/sso" in login_url
    mock_auth_instance.login.assert_called_once()


@pytest.mark.asyncio
@patch("litellm.proxy.auth.saml_handler.OneLogin_Saml2_Auth")
async def test_process_saml_response_success(mock_saml_auth, mock_saml_env_vars):
    """Test processing a successful SAML response"""
    # Mock the SAML auth object
    mock_auth_instance = MagicMock()
    mock_auth_instance.process_response = MagicMock()
    mock_auth_instance.get_errors.return_value = []
    mock_auth_instance.is_authenticated.return_value = True
    mock_auth_instance.get_nameid.return_value = "user@example.com"
    mock_auth_instance.get_session_index.return_value = "session123"
    mock_auth_instance.get_attributes.return_value = {
        "email": ["user@example.com"],
        "firstName": ["John"],
        "lastName": ["Doe"],
        "displayName": ["John Doe"],
    }
    mock_saml_auth.return_value = mock_auth_instance

    # Create a mock request with form data
    mock_request = MagicMock()
    mock_request.url.scheme = "https"
    mock_request.url.hostname = "proxy.example.com"
    mock_request.url.port = 443
    mock_request.url.path = "/sso/saml/acs"
    mock_request.query_params = {}

    # Mock form data
    async def mock_form():
        return {"SAMLResponse": "mock_saml_response"}

    mock_request.form = mock_form

    # Process the SAML response
    result = await SAMLAuthenticationHandler.process_saml_response(mock_request)

    # Verify the result
    assert isinstance(result, CustomOpenID)
    assert result.id == "user@example.com"
    assert result.email == "user@example.com"
    assert result.first_name == "John"
    assert result.last_name == "Doe"
    assert result.display_name == "John Doe"
    assert result.provider == "saml"


@pytest.mark.asyncio
@patch("litellm.proxy.auth.saml_handler.OneLogin_Saml2_Auth")
async def test_process_saml_response_auth_failed(mock_saml_auth, mock_saml_env_vars):
    """Test processing a SAML response when authentication fails"""
    from litellm.proxy._types import ProxyException

    # Mock the SAML auth object with authentication failure
    mock_auth_instance = MagicMock()
    mock_auth_instance.process_response = MagicMock()
    mock_auth_instance.get_errors.return_value = ["invalid_response"]
    mock_auth_instance.get_last_error_reason.return_value = "Invalid SAML response"
    mock_auth_instance.is_authenticated.return_value = False
    mock_saml_auth.return_value = mock_auth_instance

    # Create a mock request
    mock_request = MagicMock()
    mock_request.url.scheme = "https"
    mock_request.url.hostname = "proxy.example.com"
    mock_request.url.port = 443
    mock_request.url.path = "/sso/saml/acs"
    mock_request.query_params = {}

    async def mock_form():
        return {"SAMLResponse": "mock_invalid_response"}

    mock_request.form = mock_form

    # Verify that ProxyException is raised
    with pytest.raises(ProxyException) as exc_info:
        await SAMLAuthenticationHandler.process_saml_response(mock_request)

    assert "SAML authentication failed" in str(exc_info.value.message)


@patch("litellm.proxy.auth.saml_handler.OneLogin_Saml2_Settings")
def test_get_metadata(mock_saml_settings, mock_saml_env_vars):
    """Test generating SAML metadata XML"""
    # Mock the settings object
    mock_settings_instance = MagicMock()
    mock_settings_instance.get_sp_metadata.return_value = '<?xml version="1.0"?><md:EntityDescriptor>...</md:EntityDescriptor>'
    mock_settings_instance.validate_metadata.return_value = []
    mock_saml_settings.return_value = mock_settings_instance

    # Get the metadata
    metadata = SAMLAuthenticationHandler.get_metadata()

    # Verify metadata was generated
    assert metadata is not None
    assert "<?xml" in metadata
    assert "EntityDescriptor" in metadata
    mock_settings_instance.get_sp_metadata.assert_called_once()
    mock_settings_instance.validate_metadata.assert_called_once()


def test_get_saml_settings_missing_idp_entity_id(monkeypatch):
    """Test that missing SAML_IDP_ENTITY_ID raises an error"""
    from litellm.proxy._types import ProxyException

    # Set only some of the required variables
    monkeypatch.setenv("SAML_IDP_SSO_URL", "https://idp.example.com/sso")
    monkeypatch.setenv("SAML_IDP_X509_CERT", "mock_cert")
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")

    # Ensure SAML_IDP_ENTITY_ID is not set
    if "SAML_IDP_ENTITY_ID" in os.environ:
        del os.environ["SAML_IDP_ENTITY_ID"]

    request_data = {
        "https": "on",
        "http_host": "proxy.example.com",
        "server_port": 443,
        "script_name": "/",
        "get_data": {},
        "post_data": {},
    }

    # Verify that ProxyException is raised
    with pytest.raises(ProxyException) as exc_info:
        SAMLAuthenticationHandler._get_saml_settings(request_data)

    assert "SAML_IDP_ENTITY_ID not set" in str(exc_info.value.message)
