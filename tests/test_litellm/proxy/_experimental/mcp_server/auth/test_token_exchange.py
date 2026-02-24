"""
Tests for OAuth 2.0 Token Exchange (RFC 8693) handler for MCP servers.

Covers: exchange flow, caching, error handling, resolve_mcp_auth integration,
bearer token extraction, and config loading.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.proxy._experimental.mcp_server.auth.token_exchange import (
    TOKEN_EXCHANGE_GRANT_TYPE,
    TokenExchangeHandler,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
)
from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
    resolve_mcp_auth,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _obo_server(**overrides) -> MCPServer:
    defaults = dict(
        server_id="srv-obo-1",
        name="test-obo",
        url="https://mcp.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_token_exchange,
        client_id="litellm-client-id",
        client_secret="litellm-client-secret",
        token_exchange_endpoint="https://idp.example.com/oauth2/token",
        audience="api://mcp-server",
        scopes=["mcp.tools.read", "mcp.tools.execute"],
    )
    defaults.update(overrides)
    return MCPServer(**defaults)


def _exchange_response(token="exchanged-tok-abc", expires_in=3600):
    resp = MagicMock()
    resp.json.return_value = {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": expires_in,
    }
    resp.raise_for_status = MagicMock()
    resp.text = ""
    return resp


# ── Exchange Flow ──


@pytest.mark.asyncio
async def test_exchange_token_success():
    """Token exchange sends correct RFC 8693 parameters and returns access_token."""
    handler = TokenExchangeHandler()
    server = _obo_server()
    mock_client = AsyncMock()
    mock_client.post.return_value = _exchange_response("scoped-token-1")

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await handler.exchange_token("user-jwt-xyz", server)

    assert result == "scoped-token-1"
    mock_client.post.assert_called_once()

    _, kwargs = mock_client.post.call_args
    data = kwargs["data"]
    assert data["grant_type"] == TOKEN_EXCHANGE_GRANT_TYPE
    assert data["subject_token"] == "user-jwt-xyz"
    assert data["subject_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
    assert data["audience"] == "api://mcp-server"
    assert data["scope"] == "mcp.tools.read mcp.tools.execute"
    assert data["client_id"] == "litellm-client-id"
    assert data["client_secret"] == "litellm-client-secret"


@pytest.mark.asyncio
async def test_exchange_token_no_audience():
    """When audience is None, it is omitted from the request."""
    handler = TokenExchangeHandler()
    server = _obo_server(audience=None)
    mock_client = AsyncMock()
    mock_client.post.return_value = _exchange_response()

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.get_async_httpx_client",
        return_value=mock_client,
    ):
        await handler.exchange_token("user-jwt", server)

    _, kwargs = mock_client.post.call_args
    assert "audience" not in kwargs["data"]


@pytest.mark.asyncio
async def test_exchange_token_no_scopes():
    """When scopes is None, scope param is omitted from the request."""
    handler = TokenExchangeHandler()
    server = _obo_server(scopes=None)
    mock_client = AsyncMock()
    mock_client.post.return_value = _exchange_response()

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.get_async_httpx_client",
        return_value=mock_client,
    ):
        await handler.exchange_token("user-jwt", server)

    _, kwargs = mock_client.post.call_args
    assert "scope" not in kwargs["data"]


# ── Caching ──


@pytest.mark.asyncio
async def test_exchange_token_cached():
    """Second call with same user token uses cache — only 1 HTTP POST."""
    handler = TokenExchangeHandler()
    server = _obo_server()
    mock_client = AsyncMock()
    mock_client.post.return_value = _exchange_response("cached-exchange-tok")

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.get_async_httpx_client",
        return_value=mock_client,
    ):
        t1 = await handler.exchange_token("same-jwt", server)
        t2 = await handler.exchange_token("same-jwt", server)

    assert t1 == t2 == "cached-exchange-tok"
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_different_user_tokens_not_shared():
    """Different user JWTs get different exchanged tokens."""
    handler = TokenExchangeHandler()
    server = _obo_server()
    call_count = 0

    async def mock_post(url, data=None):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.json.return_value = {
            "access_token": f"exchanged-{call_count}",
            "expires_in": 3600,
        }
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.post = mock_post

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.get_async_httpx_client",
        return_value=mock_client,
    ):
        t1 = await handler.exchange_token("user-a-jwt", server)
        t2 = await handler.exchange_token("user-b-jwt", server)

    assert t1 == "exchanged-1"
    assert t2 == "exchanged-2"
    assert call_count == 2


# ── Error Handling ──


@pytest.mark.asyncio
async def test_exchange_token_http_error():
    """HTTP errors from the IDP are wrapped in a ValueError."""
    handler = TokenExchangeHandler()
    server = _obo_server()
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "invalid_grant"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request",
        request=MagicMock(),
        response=mock_response,
    )
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.get_async_httpx_client",
        return_value=mock_client,
    ), pytest.raises(ValueError, match="failed with status 400"):
        await handler.exchange_token("bad-jwt", server)


@pytest.mark.asyncio
async def test_exchange_token_missing_access_token():
    """Response without access_token raises ValueError."""
    handler = TokenExchangeHandler()
    server = _obo_server()
    resp = MagicMock()
    resp.json.return_value = {"token_type": "Bearer"}
    resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post.return_value = resp

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.get_async_httpx_client",
        return_value=mock_client,
    ), pytest.raises(ValueError, match="missing 'access_token'"):
        await handler.exchange_token("jwt", server)


@pytest.mark.asyncio
async def test_exchange_token_missing_endpoint():
    """Missing token_exchange_endpoint and token_url raises ValueError."""
    handler = TokenExchangeHandler()
    server = _obo_server(token_exchange_endpoint=None, token_url=None)

    with pytest.raises(ValueError, match="no token_exchange_endpoint or token_url"):
        await handler.exchange_token("jwt", server)


@pytest.mark.asyncio
async def test_exchange_token_missing_credentials():
    """Missing client_id or client_secret raises ValueError."""
    handler = TokenExchangeHandler()
    server = _obo_server(client_id=None, client_secret=None)
    # has_token_exchange_config will be False, so we call _do_exchange directly
    with pytest.raises(ValueError, match="missing client_id or client_secret"):
        await handler._do_exchange("jwt", server)


# ── resolve_mcp_auth Integration ──


@pytest.mark.asyncio
async def test_resolve_mcp_auth_with_token_exchange():
    """resolve_mcp_auth delegates to token exchange when server has OBO config and subject_token provided."""
    server = _obo_server()
    mock_handler = AsyncMock()
    mock_handler.exchange_token.return_value = "obo-scoped-token"

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.token_exchange.mcp_token_exchange_handler",
        mock_handler,
    ):
        result = await resolve_mcp_auth(server, subject_token="user-jwt")

    assert result == "obo-scoped-token"
    mock_handler.exchange_token.assert_called_once_with("user-jwt", server)


@pytest.mark.asyncio
async def test_resolve_mcp_auth_obo_without_subject_token_falls_through():
    """Without a subject_token, resolve_mcp_auth falls through to client_credentials."""
    server = _obo_server(
        token_url="https://auth.example.com/token",
    )
    mock_client = AsyncMock()
    mock_client.post.return_value = _exchange_response("cc-token")

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await resolve_mcp_auth(server, subject_token=None)

    # Falls through to client_credentials since subject_token is None
    # The server has client_id/client_secret/token_url so has_client_credentials is True
    assert result == "cc-token"


@pytest.mark.asyncio
async def test_resolve_mcp_auth_header_beats_obo():
    """An explicit mcp_auth_header takes priority over OBO token exchange."""
    server = _obo_server()
    result = await resolve_mcp_auth(
        server, mcp_auth_header="Bearer override", subject_token="user-jwt"
    )
    assert result == "Bearer override"


# ── Bearer Token Extraction ──


def test_extract_bearer_token_from_oauth2_headers():
    """Extracts token from oauth2_headers Authorization header."""
    result = MCPServerManager._extract_bearer_token(
        oauth2_headers={"Authorization": "Bearer my-jwt-token"},
        raw_headers=None,
    )
    assert result == "my-jwt-token"


def test_extract_bearer_token_from_raw_headers():
    """Falls back to raw_headers when oauth2_headers missing."""
    result = MCPServerManager._extract_bearer_token(
        oauth2_headers=None,
        raw_headers={"authorization": "Bearer raw-jwt"},
    )
    assert result == "raw-jwt"


def test_extract_bearer_token_no_bearer_prefix():
    """Returns token as-is when no Bearer prefix."""
    result = MCPServerManager._extract_bearer_token(
        oauth2_headers={"Authorization": "some-opaque-token"},
        raw_headers=None,
    )
    assert result == "some-opaque-token"


def test_extract_bearer_token_none():
    """Returns None when no auth headers present."""
    result = MCPServerManager._extract_bearer_token(
        oauth2_headers=None,
        raw_headers=None,
    )
    assert result is None


# ── MCPServer Properties ──


def test_has_token_exchange_config_true():
    """has_token_exchange_config is True for a fully configured OBO server."""
    server = _obo_server()
    assert server.has_token_exchange_config is True


def test_has_token_exchange_config_false_wrong_auth_type():
    """has_token_exchange_config is False when auth_type is not oauth2_token_exchange."""
    server = _obo_server(auth_type=MCPAuth.oauth2)
    assert server.has_token_exchange_config is False


def test_has_token_exchange_config_false_missing_creds():
    """has_token_exchange_config is False when client_id/client_secret missing."""
    server = _obo_server(client_id=None)
    assert server.has_token_exchange_config is False


def test_has_token_exchange_config_uses_token_url_fallback():
    """has_token_exchange_config is True when token_url is set instead of token_exchange_endpoint."""
    server = _obo_server(
        token_exchange_endpoint=None,
        token_url="https://idp.example.com/token",
    )
    assert server.has_token_exchange_config is True


# ── Config Loading ──


@pytest.mark.asyncio
async def test_config_loading_token_exchange_fields():
    """load_servers_from_config correctly maps OBO config fields to MCPServer."""
    manager = MCPServerManager()
    config = {
        "my_obo_server": {
            "url": "https://mcp.example.com/mcp",
            "transport": "http",
            "auth_type": "oauth2_token_exchange",
            "client_id": "my-client",
            "client_secret": "my-secret",
            "token_exchange_endpoint": "https://idp.example.com/oauth2/token",
            "audience": "api://my-mcp",
            "scopes": ["read", "write"],
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        }
    }
    await manager.load_servers_from_config(config)

    servers = list(manager.config_mcp_servers.values())
    assert len(servers) == 1

    server = servers[0]
    assert server.auth_type == MCPAuth.oauth2_token_exchange
    assert server.token_exchange_endpoint == "https://idp.example.com/oauth2/token"
    assert server.audience == "api://my-mcp"
    assert server.subject_token_type == "urn:ietf:params:oauth:token-type:jwt"
    assert server.client_id == "my-client"
    assert server.client_secret == "my-secret"
    assert server.scopes == ["read", "write"]
    assert server.has_token_exchange_config is True


@pytest.mark.asyncio
async def test_config_loading_default_subject_token_type():
    """subject_token_type defaults to access_token when not specified in config."""
    manager = MCPServerManager()
    config = {
        "obo_defaults": {
            "url": "https://mcp.example.com/mcp",
            "transport": "http",
            "auth_type": "oauth2_token_exchange",
            "client_id": "cid",
            "client_secret": "csec",
            "token_exchange_endpoint": "https://idp.example.com/token",
        }
    }
    await manager.load_servers_from_config(config)

    server = list(manager.config_mcp_servers.values())[0]
    assert server.subject_token_type == "urn:ietf:params:oauth:token-type:access_token"
