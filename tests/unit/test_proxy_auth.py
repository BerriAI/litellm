"""
Unit tests for litellm.proxy_auth module.

Tests the OAuth2/JWT token management for LiteLLM Proxy authentication.
"""

import time
from unittest.mock import Mock, patch

import pytest

from litellm.proxy_auth import (
    AccessToken,
    AzureADCredential,
    GenericOAuth2Credential,
    ProxyAuthHandler,
)


class TestAccessToken:
    """Tests for AccessToken dataclass."""

    def test_access_token_creation(self):
        """Test AccessToken can be created with required fields."""
        token = AccessToken(token="test-token", expires_on=1234567890)
        assert token.token == "test-token"
        assert token.expires_on == 1234567890

    def test_access_token_equality(self):
        """Test AccessToken equality comparison."""
        token1 = AccessToken(token="test", expires_on=123)
        token2 = AccessToken(token="test", expires_on=123)
        assert token1 == token2


class MockCredential:
    """Mock credential for testing."""

    def __init__(self, expires_in_seconds: int = 3600):
        self.call_count = 0
        self.expires_in = expires_in_seconds

    def get_token(self, scope: str) -> AccessToken:
        self.call_count += 1
        return AccessToken(
            token=f"mock-token-{self.call_count}",
            expires_on=int(time.time()) + self.expires_in,
        )


class TestProxyAuthHandler:
    """Tests for ProxyAuthHandler."""

    def test_get_auth_headers_returns_bearer_token(self):
        """Test that get_auth_headers returns correct Authorization header."""
        cred = MockCredential()
        handler = ProxyAuthHandler(credential=cred, scope="test-scope")

        headers = handler.get_auth_headers()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "mock-token-1" in headers["Authorization"]

    def test_token_caching(self):
        """Test that tokens are cached and not re-requested."""
        cred = MockCredential(expires_in_seconds=3600)  # Long expiry
        handler = ProxyAuthHandler(credential=cred, scope="test-scope")

        # Multiple calls should only request token once
        handler.get_auth_headers()
        handler.get_auth_headers()
        handler.get_auth_headers()

        assert cred.call_count == 1

    def test_token_refresh_when_about_to_expire(self):
        """Test that tokens are refreshed when about to expire (within 60s buffer)."""
        cred = MockCredential(expires_in_seconds=30)  # Expires in 30s (< 60s buffer)
        handler = ProxyAuthHandler(credential=cred, scope="test-scope")

        # First call gets token
        handler.get_auth_headers()
        # Second call should refresh because token expires within 60s buffer
        handler.get_auth_headers()

        assert cred.call_count == 2

    def test_get_token_method(self):
        """Test the get_token method returns AccessToken."""
        cred = MockCredential()
        handler = ProxyAuthHandler(credential=cred, scope="test-scope")

        token = handler.get_token()

        assert isinstance(token, AccessToken)
        assert token.token == "mock-token-1"


class TestAzureADCredential:
    """Tests for AzureADCredential."""

    def test_lazy_initialization(self):
        """Test that azure-identity is not imported until get_token is called."""
        # This should not raise ImportError even if azure-identity is not installed
        cred = AzureADCredential(credential=None)
        # _initialized should be False until get_token is called
        assert cred._initialized is False

    def test_wraps_azure_credential(self):
        """Test that AzureADCredential wraps an azure-identity credential."""
        # Mock Azure credential
        mock_azure_cred = Mock()
        mock_azure_cred.get_token.return_value = Mock(
            token="azure-token", expires_on=9999999999
        )

        cred = AzureADCredential(credential=mock_azure_cred)
        token = cred.get_token("https://graph.microsoft.com/.default")

        assert token.token == "azure-token"
        assert token.expires_on == 9999999999
        mock_azure_cred.get_token.assert_called_once_with(
            "https://graph.microsoft.com/.default"
        )


class TestGenericOAuth2Credential:
    """Tests for GenericOAuth2Credential."""

    def test_token_request(self):
        """Test that GenericOAuth2Credential makes correct OAuth2 request."""
        with patch("httpx.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "oauth2-token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            cred = GenericOAuth2Credential(
                client_id="test-client",
                client_secret="test-secret",
                token_url="https://example.com/oauth2/token",
            )
            token = cred.get_token("test-scope")

            assert token.token == "oauth2-token"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["data"]["grant_type"] == "client_credentials"
            assert call_kwargs[1]["data"]["client_id"] == "test-client"
            assert call_kwargs[1]["data"]["client_secret"] == "test-secret"
            assert call_kwargs[1]["data"]["scope"] == "test-scope"

    def test_token_caching(self):
        """Test that GenericOAuth2Credential caches tokens."""
        with patch("httpx.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "oauth2-token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            cred = GenericOAuth2Credential(
                client_id="test-client",
                client_secret="test-secret",
                token_url="https://example.com/oauth2/token",
            )

            # Multiple calls should only make one HTTP request
            cred.get_token("test-scope")
            cred.get_token("test-scope")
            cred.get_token("test-scope")

            assert mock_post.call_count == 1


class TestLiteLLMIntegration:
    """Tests for integration with litellm module."""

    def test_proxy_auth_variable_exists(self):
        """Test that litellm.proxy_auth variable exists."""
        import litellm

        # Should be None by default
        assert hasattr(litellm, "proxy_auth")

    def test_proxy_auth_can_be_set(self):
        """Test that litellm.proxy_auth can be set to a ProxyAuthHandler."""
        import litellm

        original_value = litellm.proxy_auth
        try:
            cred = MockCredential()
            handler = ProxyAuthHandler(credential=cred, scope="test")
            litellm.proxy_auth = handler

            assert litellm.proxy_auth is handler
        finally:
            litellm.proxy_auth = original_value
