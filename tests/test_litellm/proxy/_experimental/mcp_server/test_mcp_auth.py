import pytest
from unittest.mock import AsyncMock, patch
from litellm.proxy._experimental.mcp_server.server import (
    set_auth_context,
    get_active_auth_context,
    extract_mcp_auth_context,
    get_auth_context,
    get_or_extract_auth_context,
)
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_auth_context_persistence():
    """Test that auth context is correctly set and retrieved."""
    auth_data = UserAPIKeyAuth(api_key="test-key")

    # Set context
    set_auth_context(auth_data)

    # Retrieve context
    retrieved = get_active_auth_context()
    assert retrieved is not None
    assert retrieved.user_api_key_auth.api_key == auth_data.api_key

    # Test get_auth_context tuple
    auth_tuple = get_auth_context()
    assert auth_tuple[0].api_key == auth_data.api_key


@pytest.mark.asyncio
async def test_get_or_extract_auth_context_fallback():
    """Test get_or_extract_auth_context fallback to server object."""
    from litellm.proxy._experimental.mcp_server.server import server, MCPAuthenticatedUser, auth_context_var

    auth_data = UserAPIKeyAuth(api_key="fallback-key")
    auth_user = MCPAuthenticatedUser(user_api_key_auth=auth_data)

    # Set on server object
    server._litellm_auth_context = auth_user

    # Ensure ContextVar is empty
    token = auth_context_var.set(None)
    try:
        result = await get_or_extract_auth_context()
        assert result[0].api_key == "fallback-key"
    finally:
        auth_context_var.reset(token)


@pytest.mark.asyncio
async def test_extract_mcp_auth_context_with_key():
    """Test extract_mcp_auth_context with a valid API key."""
    mock_scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer sk-123")],
        "path": "/mcp/sse",
        "method": "GET",
        "query_string": b"",
    }

    mock_user_auth = UserAPIKeyAuth(api_key="sk-123")

    with patch("litellm.proxy.auth.auth_checks.common_checks", new_callable=AsyncMock) as mock_auth:
        mock_auth.return_value = mock_user_auth

        result = await extract_mcp_auth_context(mock_scope, "/mcp/sse")

        # Returns (user_api_key_auth, mcp_auth_header, mcp_servers, mcp_server_auth_headers, oauth2_headers, raw_headers, client_ip)
        assert result[0].api_key == mock_user_auth.api_key
        assert result[5]["authorization"] == "Bearer sk-123"
