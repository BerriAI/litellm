"""Tests for MCP OAuth discoverable endpoints"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_authorize_endpoint_includes_response_type():
    """Test that authorize endpoint includes response_type=code parameter (fixes #15684)"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    # Mock the encryption functions to avoid needing a signing key
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state"

        # Call authorize endpoint
        response = await authorize(
            request=mock_request,
            client_id="test_oauth",
            redirect_uri="https://client.example.com/callback",
            state="test_state",
        )

    # Verify response is a redirect
    assert response.status_code == 307  # FastAPI RedirectResponse default

    # Verify response_type is in the redirect URL
    assert "response_type=code" in response.headers["location"]
    assert "https://provider.com/oauth/authorize" in response.headers["location"]
    assert "client_id=test_client_id" in response.headers["location"]
    assert "scope=read+write" in response.headers["location"]


@pytest.mark.asyncio
async def test_authorize_endpoint_forwards_pkce_parameters():
    """Test that authorize endpoint forwards PKCE parameters (code_challenge and code_challenge_method)"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server (simulating Google OAuth)
    oauth2_server = MCPServer(
        server_id="google_mcp",
        name="google_mcp",
        server_name="google_mcp",
        alias="google_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="669428968603-test.apps.googleusercontent.com",
        client_secret="GOCSPX-test_secret",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive", "openid", "email"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm-proxy.example.com/"
    mock_request.headers = {}

    # Mock the encryption function
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state_with_pkce"

        # Call authorize endpoint with PKCE parameters
        response = await authorize(
            request=mock_request,
            client_id="google_mcp",
            redirect_uri="http://localhost:60108/callback",
            state="test_client_state",
            code_challenge="x6YH_qgwbvOzbsHDuL1sW9gYkR9-gObUiIB5RkPwxDk",
            code_challenge_method="S256",
        )

    # Verify response is a redirect
    assert response.status_code == 307

    # Verify PKCE parameters are included in the redirect URL
    location = response.headers["location"]
    assert "https://accounts.google.com/o/oauth2/v2/auth" in location
    assert "code_challenge=x6YH_qgwbvOzbsHDuL1sW9gYkR9-gObUiIB5RkPwxDk" in location
    assert "code_challenge_method=S256" in location
    assert "client_id=669428968603-test.apps.googleusercontent.com" in location
    assert "response_type=code" in location


@pytest.mark.asyncio
async def test_token_endpoint_forwards_code_verifier():
    """Test that token endpoint forwards code_verifier for PKCE flow"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
        import httpx
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="google_mcp",
        name="google_mcp",
        server_name="google_mcp",
        alias="google_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="669428968603-test.apps.googleusercontent.com",
        client_secret="GOCSPX-test_secret",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive", "openid", "email"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm-proxy.example.com/"

    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "ya29.test_access_token",
        "token_type": "Bearer",
        "expires_in": 3599,
        "scope": "openid email https://www.googleapis.com/auth/drive",
    }
    mock_response.raise_for_status = MagicMock()

    # Mock the async httpx client with AsyncMock for async methods
    from unittest.mock import AsyncMock
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client"
    ) as mock_get_client:
        mock_async_client = MagicMock()
        # Use AsyncMock for the async post method
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_async_client

        # Call token endpoint with code_verifier
        response = await token_endpoint(
            request=mock_request,
            grant_type="authorization_code",
            code="4/test_authorization_code",
            redirect_uri="http://localhost:60108/callback",
            client_id="google_mcp",
            client_secret="dummy",
            code_verifier="test_code_verifier_from_client",
        )

    # Verify that the token endpoint was called with code_verifier
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args

    # Check the data parameter includes code_verifier
    assert call_args[1]["data"]["code_verifier"] == "test_code_verifier_from_client"
    assert call_args[1]["data"]["code"] == "4/test_authorization_code"
    assert call_args[1]["data"]["client_id"] == "669428968603-test.apps.googleusercontent.com"
    assert call_args[1]["data"]["client_secret"] == "GOCSPX-test_secret"
    assert call_args[1]["data"]["grant_type"] == "authorization_code"

    # Verify response
    response_data = response.body
    import json
    token_data = json.loads(response_data)
    assert token_data["access_token"] == "ya29.test_access_token"
    assert token_data["token_type"] == "Bearer"
