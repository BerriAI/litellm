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
