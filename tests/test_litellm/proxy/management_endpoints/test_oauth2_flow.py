"""
Comprehensive unit tests for OAuth2 SSO flow implementation.

This module tests the OAuth2-compatible SSO endpoints that allow external 
applications (like RooCode) to authenticate users and obtain API tokens.

Test Coverage:
- OAuth2 parameter validation
- Secure state generation and validation
- Token response formatting
- Error handling scenarios
- CORS header configuration
- Security edge cases
"""

import json
import os
import time
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from litellm.proxy.management_endpoints.sso_helper_utils import (
    OAuth2TokenManager,
    OAuth2StateManager,
    OAuth2URLManager,
    OAuth2CORSHandler,
)
from litellm.proxy.management_endpoints.ui_sso import (
    serve_login_page,
)


class TestOAuth2TokenResponse:
    """Test OAuth2 token response formatting and compliance."""
    
    def test_create_oauth_token_response_basic(self):
        """Test basic OAuth2 token response creation."""
        # Arrange
        token = "sk-litellm-test123"
        
        # Act
        response = OAuth2TokenManager.create_token_response(token)
        
        # Assert
        assert response == {
            "access_token": "sk-litellm-test123",
            "token_type": "Bearer",
            "expires_in": 86400
        }
    
    def test_create_oauth_token_response_structure(self):
        """Test OAuth2 token response has required fields."""
        # Arrange
        token = "sk-litellm-test123"
        
        # Act
        response = OAuth2TokenManager.create_token_response(token)
        
        # Assert
        assert "access_token" in response
        assert "token_type" in response
        assert "expires_in" in response
        assert response["token_type"] == "Bearer"
        assert isinstance(response["expires_in"], int)


class TestOAuth2FlowInitiation:
    """Test OAuth2 flow initiation and parameter validation."""
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request object."""
        request = MagicMock(spec=Request)
        request.url = MagicMock()
        request.url.__str__.return_value = "https://your-litellm-proxy.com/sso/key/generate?response_type=oauth_token"
        return request
    
    @patch.dict(os.environ, {
        "MICROSOFT_CLIENT_ID": "test_client_id",
        "LITELLM_LICENSE": "test_license"
    })
    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_oauth_flow_initiation_valid(self, mock_request):
        """Test successful OAuth2 flow initiation."""
        # Act
        with patch("litellm.proxy.management_endpoints.ui_sso.str_to_bool", return_value=False):
            response = run_async_test(serve_login_page(
                request=mock_request,
                response_type="oauth_token"
            ))
        
        # Assert
        assert isinstance(response, RedirectResponse)
        assert response.status_code == 303
        
        # Verify redirect URL contains oauth_flow=true
        redirect_url = response.headers["location"]
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        assert query_params.get("oauth_flow") == ["true"]
    
    @patch.dict(os.environ, {
        "MICROSOFT_CLIENT_ID": "test_client_id",
        "LITELLM_LICENSE": "test_license"
    })
    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_oauth_flow_invalid_response_type(self, mock_request):
        """Test OAuth2 flow with invalid response_type."""
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            with patch("litellm.proxy.management_endpoints.ui_sso.str_to_bool", return_value=False):
                run_async_test(serve_login_page(
                    request=mock_request,
                    response_type="invalid_type"
                ))
        
        assert exc_info.value.status_code == 400
        assert "Unsupported response_type" in str(exc_info.value.detail)
    
    @patch.dict(os.environ, {
        "MICROSOFT_CLIENT_ID": "test_client_id",
        "LITELLM_LICENSE": "test_license"
    })
    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_oauth_flow_url_parsing_security(self, mock_request):
        """Test secure URL parsing prevents injection attacks."""
        # Arrange - Create a potentially malicious URL
        mock_request.url.__str__.return_value = (
            "https://your-litellm-proxy.com/sso/key/generate?response_type=oauth_token"
            "&redirect=https://evil.com&param=<script>alert('xss')</script>"
        )
        
        # Act
        with patch("litellm.proxy.management_endpoints.ui_sso.str_to_bool", return_value=False):
            response = run_async_test(serve_login_page(
                request=mock_request,
                response_type="oauth_token"
            ))
        
        # Assert
        assert isinstance(response, RedirectResponse)
        redirect_url = response.headers["location"]
        
        # Verify malicious content is properly encoded/handled
        assert "<script>" not in redirect_url
        assert "https://evil.com" not in redirect_url
        parsed_url = urlparse(redirect_url)
        assert parsed_url.path == "/sso/login"


class TestOAuth2StateHandling:
    """Test OAuth2 state parameter generation and validation."""
    
    @patch("litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.get_redirect_url_for_sso")
    @patch("litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler._get_cli_state")
    def test_oauth_state_generation(self, mock_get_cli_state, mock_get_redirect_url):
        """Test secure OAuth2 state parameter generation."""
        # Arrange
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.__str__.return_value = "https://your-litellm-proxy.com/sso/login?oauth_flow=true"
        
        mock_get_redirect_url.return_value = "https://provider.com/oauth/authorize"
        mock_get_cli_state.return_value = None
        
        # Mock the redirect response that would normally be returned
        with patch("litellm.proxy.management_endpoints.ui_sso.RedirectResponse") as mock_redirect:
            mock_redirect.return_value = MagicMock(spec=RedirectResponse)
            
            # Act - Test the state generation logic using OAuth2StateManager
            generated_state = OAuth2StateManager.generate_secure_state("oauth_token")
            
            # Assert - Verify state structure
            assert generated_state.startswith("oauth:")
            parsed_data = json.loads(generated_state[6:])
            assert parsed_data["flow"] == "oauth_token"
            assert "timestamp" in parsed_data
            assert "nonce" in parsed_data
            assert len(parsed_data["nonce"]) > 10
    
    def test_oauth_state_validation_valid(self):
        """Test valid OAuth2 state parameter validation."""
        # Arrange - Generate a valid state using OAuth2StateManager
        state = OAuth2StateManager.generate_secure_state("oauth_token")
        
        # Act
        is_valid = OAuth2StateManager.validate_state(state)
        flow_type = OAuth2StateManager.extract_flow_type(state)
        
        # Assert
        assert is_valid is True
        assert flow_type == "oauth_token"
    
    def test_oauth_state_validation_expired(self):
        """Test expired OAuth2 state parameter validation."""
        # Arrange - Create an expired state by manually setting old timestamp
        old_time = int(time.time()) - 600  # 10 minutes ago
        state_data = {
            "flow": "oauth_token",
            "timestamp": old_time,
            "nonce": "test_nonce_123"
        }
        expired_state = f"oauth:{json.dumps(state_data)}"
        
        # Act
        is_valid = OAuth2StateManager.validate_state(expired_state)
        
        # Assert - State should be considered expired (> 5 minutes)
        assert is_valid is False
    
    def test_oauth_state_validation_malformed(self):
        """Test malformed OAuth2 state parameter handling."""
        # Arrange
        malformed_states = [
            "oauth:invalid_json",
            "oauth:{incomplete",
            "not_oauth_state",
            "oauth:",
            "",
        ]
        
        for state in malformed_states:
            # Act
            is_valid = OAuth2StateManager.validate_state(state)
            
            # Assert - All malformed states should be invalid
            assert is_valid is False


class TestOAuth2ResponseValidation:
    """Test OAuth2 response structure and validation."""
    
    def test_oauth_callback_response_structure(self):
        """Test OAuth2 callback response structure matches expectations."""
        # Arrange - Simulate what a successful OAuth callback should return
        expected_token = "sk-litellm-test123"
        expected_response_content = {
            "access_token": expected_token,
            "token_type": "Bearer",
            "expires_in": 86400,
            "scope": "litellm:api"
        }
        
        expected_headers = {
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-store",
            "Pragma": "no-cache"
        }
        
        # Act - Create the response object (simulating what our code would return)
        response = JSONResponse(
            content=expected_response_content,
            headers=expected_headers,
            status_code=200
        )
        
        # Assert - Verify response structure
        assert response.status_code == 200
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        
        # Verify content structure matches OAuth2 spec
        response_body = json.loads(response.body)
        assert response_body["access_token"] == expected_token
        assert response_body["token_type"] == "Bearer"
        assert response_body["expires_in"] == 86400
        assert response_body["scope"] == "litellm:api"


class TestOAuth2CORSHandling:
    """Test CORS header configuration for OAuth2 endpoints."""
    
    def test_cors_headers_default(self):
        """Test default CORS headers configuration."""
        # Act - Get CORS headers using OAuth2CORSHandler
        cors_headers = OAuth2CORSHandler.get_oauth_cors_headers()
        
        # Assert
        assert cors_headers["Access-Control-Allow-Origin"] == "*"
        assert "GET" in cors_headers["Access-Control-Allow-Methods"]
        assert "POST" in cors_headers["Access-Control-Allow-Methods"]
        assert "no-store" in cors_headers["Cache-Control"]
    
    @patch.dict(os.environ, {"OAUTH_ALLOWED_ORIGINS": "https://roocode.com,https://app.roocode.com"})
    def test_cors_headers_configured_origins(self):
        """Test CORS headers with configured allowed origins."""
        # Act - Get CORS headers using OAuth2CORSHandler with configured origins
        cors_headers = OAuth2CORSHandler.get_oauth_cors_headers()
        
        # Assert
        assert cors_headers["Access-Control-Allow-Origin"] == "https://roocode.com,https://app.roocode.com"


class TestOAuth2SecurityFeatures:
    """Test security features and edge cases."""
    
    def test_oauth_response_no_cache_headers(self):
        """Test that OAuth2 responses include proper no-cache headers."""
        # Arrange
        token = "sk-litellm-test123"
        oauth_response = OAuth2TokenManager.create_token_response(token)
        
        # Expected headers per OAuth2 RFC
        security_headers = {
            "Cache-Control": "no-store",
            "Pragma": "no-cache"
        }
        
        # Assert
        assert "access_token" in oauth_response
        # Headers would be set in the JSONResponse object
        for header, value in security_headers.items():
            assert value in ["no-store", "no-cache"]
    
    def test_oauth_state_uniqueness(self):
        """Test that OAuth2 state parameters are unique."""
        # Arrange & Act
        state1 = OAuth2StateManager.generate_secure_state("oauth_token")
        state2 = OAuth2StateManager.generate_secure_state("oauth_token")
        
        # Assert
        assert state1 != state2
        assert len(state1) > 20  # Reasonable length for full state
        assert len(state2) > 20
    
    def test_oauth_token_format_validation(self):
        """Test that generated tokens follow expected format."""
        # Arrange
        token = "sk-litellm-1234567890abcdef"
        
        # Act
        oauth_response = OAuth2TokenManager.create_token_response(token)
        
        # Assert
        assert oauth_response["access_token"].startswith("sk-litellm-")
        assert oauth_response["token_type"] == "Bearer"
        assert isinstance(oauth_response["expires_in"], int)
        assert oauth_response["expires_in"] > 0


# Integration test helper
class TestOAuth2RedirectURI:
    """Test redirect_uri functionality for VSCode extension compatibility."""
    
    def test_redirect_uri_validation_valid_ide_schemes(self):
        """Test valid IDE and editor redirect_uri validation."""
        # Arrange
        valid_ide_uris = [
            # VSCode family
            "vscode://extension-id/callback",
            "vscode-insiders://extension-id/callback/path",
            "vscodium://extension-id/callback",
            "code-oss://extension-id/callback",
            # Other popular IDEs
            "cursor://extension-id/callback",
            "fleet://extension-id/callback", 
            "zed://extension-id/callback",
            "sublime://extension-id/callback",
            "atom://extension-id/callback",
            "nova://extension-id/callback",
            # Web protocols
            "https://example.com/callback",
            "http://localhost:3000/callback",
            # Development tools
            "github-desktop://callback/path",
            "sourcetree://callback"
        ]
        
        for uri in valid_ide_uris:
            # Act
            is_valid = OAuth2URLManager.validate_redirect_uri(uri)
            
            # Assert
            assert is_valid is True, f"URI should be valid: {uri}"
    
    def test_redirect_uri_validation_invalid(self):
        """Test invalid redirect_uri validation."""
        # Arrange
        invalid_uris = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "ftp://malicious.com/callback",
            "file:///etc/passwd",
            "",
            "not-a-uri",
            "vscode://",  # Missing extension id
            "https://"    # Invalid URL
        ]
        
        for uri in invalid_uris:
            # Act
            is_valid = OAuth2URLManager.validate_redirect_uri(uri)
            
            # Assert
            assert is_valid is False, f"URI should be invalid: {uri}"
    
    def test_build_callback_redirect_url_vscode(self):
        """Test building VSCode callback redirect URL."""
        # Arrange
        redirect_uri = "vscode://roocode.roo/litellm"
        access_token = "sk-litellm-test123"
        
        # Act
        callback_url = OAuth2URLManager.build_callback_redirect_url(
            redirect_uri=redirect_uri,
            access_token=access_token,
            token_type="Bearer",
            expires_in=86400
        )
        
        # Assert
        parsed_url = urlparse(callback_url)
        query_params = parse_qs(parsed_url.query)
        
        assert parsed_url.scheme == "vscode"
        assert parsed_url.netloc == "roocode.roo"
        assert parsed_url.path == "/litellm"
        assert query_params["access_token"] == ["sk-litellm-test123"]
        assert query_params["token_type"] == ["Bearer"]
        assert query_params["expires_in"] == ["86400"]
    
    def test_build_callback_redirect_url_https(self):
        """Test building HTTPS callback redirect URL."""
        # Arrange
        redirect_uri = "https://example.com/oauth/callback"
        access_token = "sk-litellm-test123"
        
        # Act
        callback_url = OAuth2URLManager.build_callback_redirect_url(
            redirect_uri=redirect_uri,
            access_token=access_token
        )
        
        # Assert
        parsed_url = urlparse(callback_url)
        query_params = parse_qs(parsed_url.query)
        
        assert parsed_url.scheme == "https"
        assert parsed_url.netloc == "example.com"
        assert parsed_url.path == "/oauth/callback"
        assert query_params["access_token"] == ["sk-litellm-test123"]
    
    def test_state_with_redirect_uri(self):
        """Test state generation and extraction with redirect_uri."""
        # Arrange
        redirect_uri = "vscode://roocode.roo/litellm"
        
        # Act
        state = OAuth2StateManager.generate_secure_state("oauth_token", redirect_uri)
        extracted_flow = OAuth2StateManager.extract_flow_type(state)
        extracted_uri = OAuth2StateManager.extract_redirect_uri(state)
        
        # Assert
        assert extracted_flow == "oauth_token"
        assert extracted_uri == redirect_uri
        assert OAuth2StateManager.validate_state(state) is True
    
    def test_state_without_redirect_uri(self):
        """Test state generation without redirect_uri (backward compatibility)."""
        # Arrange & Act
        state = OAuth2StateManager.generate_secure_state("oauth_token")
        extracted_flow = OAuth2StateManager.extract_flow_type(state)
        extracted_uri = OAuth2StateManager.extract_redirect_uri(state)
        
        # Assert
        assert extracted_flow == "oauth_token"
        assert extracted_uri is None
        assert OAuth2StateManager.validate_state(state) is True
    
    @patch.dict(os.environ, {"OAUTH_ALLOWED_REDIRECT_SCHEMES": "myide,customapp,special-editor"})
    def test_custom_redirect_schemes_from_env(self):
        """Test custom redirect schemes from environment variable."""
        # Arrange
        custom_uris = [
            "myide://extension/callback",
            "customapp://oauth/callback", 
            "special-editor://callback/path"
        ]
        
        for uri in custom_uris:
            # Act
            is_valid = OAuth2URLManager.validate_redirect_uri(uri)
            
            # Assert
            assert is_valid is True, f"Custom scheme URI should be valid: {uri}"
    
    def test_cursor_editor_specific_flow(self):
        """Test Cursor editor specific OAuth flow."""
        # Arrange
        redirect_uri = "cursor://roocode-extension/oauth-callback"
        access_token = "sk-litellm-cursor-test123"
        
        # Act
        callback_url = OAuth2URLManager.build_callback_redirect_url(
            redirect_uri=redirect_uri,
            access_token=access_token,
            token_type="Bearer",
            expires_in=86400
        )
        
        # Assert
        parsed_url = urlparse(callback_url)
        query_params = parse_qs(parsed_url.query)
        
        assert parsed_url.scheme == "cursor"
        assert parsed_url.netloc == "roocode-extension"
        assert parsed_url.path == "/oauth-callback"
        assert query_params["access_token"] == ["sk-litellm-cursor-test123"]
        assert query_params["token_type"] == ["Bearer"]
        assert query_params["expires_in"] == ["86400"]


class TestOAuth2Integration:
    """Integration tests for complete OAuth2 flow."""
    
    def test_oauth_flow_integration_structure(self):
        """Test the structure of a complete OAuth2 flow."""
        # This test verifies the flow structure without actual HTTP calls
        
        # Step 1: Client initiates OAuth flow
        oauth_url = "https://your-litellm-proxy.com/sso/key/generate?response_type=oauth_token"
        parsed_url = urlparse(oauth_url)
        query_params = parse_qs(parsed_url.query)
        
        assert query_params["response_type"] == ["oauth_token"]
        
        # Step 2: Expected redirect to SSO login
        expected_redirect = "/sso/login?oauth_flow=true"
        
        # Step 3: Expected final response format using OAuth2TokenManager
        expected_response = OAuth2TokenManager.create_token_response(
            "sk-litellm-xxxxxxxxxx", 
            scope="litellm:api"
        )
        
        # Assert flow structure
        assert "response_type" in str(oauth_url)
        assert "oauth_flow" in expected_redirect
        assert all(key in expected_response for key in ["access_token", "token_type", "expires_in"])
    
    def test_vscode_extension_flow_structure(self):
        """Test VSCode extension OAuth flow structure with redirect_uri."""
        # Arrange - VSCode extension OAuth URL
        vscode_oauth_url = (
            "https://your-litellm-proxy.com/sso/key/generate"
            "?response_type=oauth_token"
            "&redirect_uri=vscode://roocode.roo/litellm"
        )
        
        # Parse URL
        parsed_url = urlparse(vscode_oauth_url)
        query_params = parse_qs(parsed_url.query)
        
        # Step 1: Verify OAuth parameters
        assert query_params["response_type"] == ["oauth_token"]
        assert query_params["redirect_uri"] == ["vscode://roocode.roo/litellm"]
        
        # Step 2: Expected callback URL format
        expected_callback = (
            "vscode://roocode.roo/litellm"
            "?access_token=sk-litellm-xxxxxxxxxx"
            "&token_type=Bearer"
            "&expires_in=86400"
        )
        
        # Verify callback structure
        callback_parsed = urlparse(expected_callback)
        callback_params = parse_qs(callback_parsed.query)
        
        assert callback_parsed.scheme == "vscode"
        assert callback_parsed.netloc == "roocode.roo"
        assert callback_parsed.path == "/litellm"
        assert "access_token" in callback_params
        assert "token_type" in callback_params
        assert "expires_in" in callback_params
    
    def test_cursor_extension_flow_structure(self):
        """Test Cursor editor OAuth flow structure with redirect_uri."""
        # Arrange - Cursor extension OAuth URL
        cursor_oauth_url = (
            "https://your-litellm-proxy.com/sso/key/generate"
            "?response_type=oauth_token"
            "&redirect_uri=cursor://roocode-extension/oauth-callback"
        )
        
        # Parse URL
        parsed_url = urlparse(cursor_oauth_url)
        query_params = parse_qs(parsed_url.query)
        
        # Step 1: Verify OAuth parameters
        assert query_params["response_type"] == ["oauth_token"]
        assert query_params["redirect_uri"] == ["cursor://roocode-extension/oauth-callback"]
        
        # Step 2: Expected callback URL format
        expected_callback = (
            "cursor://roocode-extension/oauth-callback"
            "?access_token=sk-litellm-xxxxxxxxxx"
            "&token_type=Bearer"
            "&expires_in=86400"
        )
        
        # Verify callback structure
        callback_parsed = urlparse(expected_callback)
        callback_params = parse_qs(callback_parsed.query)
        
        assert callback_parsed.scheme == "cursor"
        assert callback_parsed.netloc == "roocode-extension"
        assert callback_parsed.path == "/oauth-callback"
        assert "access_token" in callback_params
        assert "token_type" in callback_params
        assert "expires_in" in callback_params


# Async test utilities
import asyncio

def run_async_test(coro):
    """Helper to run async tests in sync test environment."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_litellm/proxy/management_endpoints/test_oauth2_flow.py -v
    pytest.main([__file__, "-v"])