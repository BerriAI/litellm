import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.requests import Request
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.proxy.mcp_server.discoverable_endpoints import _managed_authorization_scope


@pytest.mark.asyncio
async def test_managed_authorization_scope_filters_unadvertised_prior_scopes():
    """Issue #50928: Unadvertised prior scopes (e.g. api://app/read from Entra) must not be requested."""
    server = MCPServer(
        name="test-entra-server",
        transport="sse",
        server_id="test-entra-server",
        url="https://mcp.example.com",
        scopes=["read", "write"],
        authorization_server_metadata={"scopes_supported": ["read", "write"]},
    )
    
    # Prior grant has the Azure Entra fully qualified scope
    fake_prior_grant = {
        "scopes": ["api://app-12345/read", "read"]
    }
    
    mock_request = MagicMock(spec=Request)
    
    with patch("litellm.proxy.mcp_server.db.get_user_oauth_credential", new_callable=AsyncMock) as mock_get_cred, \
         patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        mock_get_cred.return_value = fake_prior_grant
        
        selected_scope, user_id = await _managed_authorization_scope(
            request=mock_request,
            server=server,
            scope="read",
            authorized_user_id="user-xyz"
        )
        
        # 'api://app-12345/read' must be filtered out because it is not in advertised_scopes
        assert selected_scope is not None
        assert "api://app-12345/read" not in selected_scope
        assert "read" in selected_scope.split()


@pytest.mark.asyncio
async def test_managed_authorization_scope_preserves_valid_advertised_prior_scopes():
    """Valid advertised prior scopes should be unioned and preserved."""
    server = MCPServer(
        name="test-server",
        transport="sse",
        server_id="test-server",
        url="https://mcp.example.com",
        scopes=["read", "write", "admin"],
    )
    
    fake_prior_grant = {
        "scopes": ["read", "write"]
    }
    
    mock_request = MagicMock(spec=Request)
    
    with patch("litellm.proxy.mcp_server.db.get_user_oauth_credential", new_callable=AsyncMock) as mock_get_cred, \
         patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        mock_get_cred.return_value = fake_prior_grant
        
        selected_scope, user_id = await _managed_authorization_scope(
            request=mock_request,
            server=server,
            scope="admin",
            authorized_user_id="user-xyz"
        )
        
        # Both prior scopes (read, write) and requested scope (admin) are preserved
        assert selected_scope is not None
        scopes_list = selected_scope.split()
        assert "read" in scopes_list
        assert "write" in scopes_list
        assert "admin" in scopes_list


def test_callback_clears_credential_on_invalid_scope(monkeypatch):
    """Issue #50928: When callback receives error=invalid_scope, it clears the stale stored credential."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-for-50928")
    
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from litellm.proxy.mcp_server.discoverable_endpoints import (
        router,
        encode_state_with_base_url,
    )

    state = encode_state_with_base_url(
        base_url="http://localhost:3000/",
        original_state="client-state-123",
        client_redirect_uri="http://127.0.0.1:60108/callback",
        litellm_user_id="user-entra-1",
        mcp_server_id="server-entra-1",
    )

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("litellm.proxy.mcp_server.db.delete_user_credential", new_callable=AsyncMock) as mock_del_cred, \
         patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), \
         patch("litellm.proxy.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip", return_value=None):

        resp = client.get(
            "/callback",
            params={
                "error": "invalid_scope",
                "error_description": "Client was not registered with scope api://app/read",
                "state": state,
            },
            follow_redirects=False,
        )

        assert resp.status_code == 302
        assert mock_del_cred.await_count == 1
        mock_del_cred.assert_awaited_once_with(mock_del_cred.call_args[0][0], "user-entra-1", "server-entra-1")
