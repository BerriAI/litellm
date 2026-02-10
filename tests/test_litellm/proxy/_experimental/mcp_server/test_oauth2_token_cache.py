"""
Core tests for MCP OAuth2 machine-to-machine (client_credentials) token management.

Covers the critical path: resolve_mcp_auth(), token caching, auth priority,
fallback to static token, and the skip-condition property.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
    MCPOAuth2TokenCache,
    resolve_mcp_auth,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _server(**overrides) -> MCPServer:
    defaults = dict(
        server_id="srv-1",
        name="test",
        url="https://mcp.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="cid",
        client_secret="csec",
        token_url="https://auth.example.com/token",
    )
    defaults.update(overrides)
    return MCPServer(**defaults)


def _token_response(token="tok-abc", expires_in=3600):
    resp = MagicMock()
    resp.json.return_value = {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
    }
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_resolve_mcp_auth_fetches_oauth2_token():
    """resolve_mcp_auth fetches a token via client_credentials when the server has OAuth2 config."""
    server = _server()
    mock_client = AsyncMock()
    mock_client.post.return_value = _token_response("m2m-token-1")

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await resolve_mcp_auth(server)

    assert result == "m2m-token-1"
    mock_client.post.assert_called_once()
    post_data = mock_client.post.call_args[1]["data"]
    assert post_data["grant_type"] == "client_credentials"
    assert post_data["client_id"] == "cid"
    assert post_data["client_secret"] == "csec"


@pytest.mark.asyncio
async def test_token_cached_across_calls():
    """Second resolve_mcp_auth call reuses the cached token — only 1 HTTP POST."""
    cache = MCPOAuth2TokenCache()
    server = _server()
    mock_client = AsyncMock()
    mock_client.post.return_value = _token_response("cached-tok")

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
        return_value=mock_client,
    ), patch(
        "litellm.proxy._experimental.mcp_server.oauth2_token_cache.mcp_oauth2_token_cache",
        cache,
    ):
        t1 = await resolve_mcp_auth(server)
        t2 = await resolve_mcp_auth(server)

    assert t1 == t2 == "cached-tok"
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_per_request_header_beats_oauth2():
    """An explicit mcp_auth_header takes priority over the OAuth2 token."""
    server = _server()
    result = await resolve_mcp_auth(server, mcp_auth_header="Bearer user-tok")
    assert result == "Bearer user-tok"


@pytest.mark.asyncio
async def test_falls_back_to_static_token():
    """When no client_credentials config, resolve_mcp_auth returns the static authentication_token."""
    server = _server(
        client_id=None,
        client_secret=None,
        token_url=None,
        authentication_token="static-tok-xyz",
    )
    result = await resolve_mcp_auth(server)
    assert result == "static-tok-xyz"


def test_needs_user_oauth_token_property():
    """needs_user_oauth_token is True only for OAuth2 servers WITHOUT client_credentials."""
    # OAuth2 with credentials → M2M, no user token needed
    assert _server().needs_user_oauth_token is False

    # OAuth2 without credentials → needs per-user token
    assert _server(client_id=None, client_secret=None, token_url=None).needs_user_oauth_token is True

    # Non-OAuth2 → never needs user OAuth token
    assert _server(auth_type=MCPAuth.bearer_token).needs_user_oauth_token is False


@pytest.mark.asyncio
async def test_http_error_raises_value_error():
    """HTTP errors from the token endpoint are wrapped in a clear ValueError."""
    server = _server()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_response,
    )
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
        return_value=mock_client,
    ), pytest.raises(ValueError, match="failed with status 401"):
        await resolve_mcp_auth(server)


@pytest.mark.asyncio
async def test_non_dict_response_raises_value_error():
    """A non-dict JSON response raises a clear ValueError."""
    server = _server()
    resp = MagicMock()
    resp.json.return_value = ["not", "a", "dict"]
    resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post.return_value = resp

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
        return_value=mock_client,
    ), pytest.raises(ValueError, match="non-object JSON"):
        await resolve_mcp_auth(server)
