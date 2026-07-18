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
        oauth2_flow="client_credentials",
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

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
            return_value=mock_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.mcp_oauth2_token_cache",
            cache,
        ),
    ):
        t1 = await resolve_mcp_auth(server)
        t2 = await resolve_mcp_auth(server)

    assert t1 == t2 == "cached-tok"
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_m2m_token_not_shared_across_server_ids_with_identical_config():
    """Two servers with byte-identical client_credentials config but different server_ids must not
    share a cached M2M token: the cache is keyed by server_id, so a new server entry (even one
    recreated with the same URL and credentials) mints its own token instead of inheriting the
    sibling's. Guards against the cache key ever collapsing to the URL or the client config."""
    cache = MCPOAuth2TokenCache()
    server_a = _server(server_id="srv-a")
    server_b = _server(server_id="srv-b")
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_token_response("tok-for-a"), _token_response("tok-for-b")]

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
            return_value=mock_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.mcp_oauth2_token_cache",
            cache,
        ),
    ):
        token_a = await resolve_mcp_auth(server_a)
        token_b = await resolve_mcp_auth(server_b)

    assert token_a == "tok-for-a"
    assert token_b == "tok-for-b"
    assert mock_client.post.call_count == 2


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
        oauth2_flow=None,
        authentication_token="static-tok-xyz",
    )
    result = await resolve_mcp_auth(server)
    assert result == "static-tok-xyz"


def test_needs_user_oauth_token_property():
    """needs_user_oauth_token is True only for OAuth2 servers WITHOUT client_credentials."""
    # OAuth2 with credentials → M2M, no user token needed
    assert _server().needs_user_oauth_token is False

    # OAuth2 without credentials → needs per-user token
    assert (
        _server(
            client_id=None, client_secret=None, token_url=None, oauth2_flow=None
        ).needs_user_oauth_token
        is True
    )

    # Non-OAuth2 → never needs user OAuth token
    assert _server(auth_type=MCPAuth.bearer_token).needs_user_oauth_token is False


@pytest.mark.asyncio
async def test_http_error_raises_value_error():
    """HTTP errors from the token endpoint are wrapped in a clear ValueError."""
    server = _server()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized",
        request=MagicMock(),
        response=mock_response,
    )
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
            return_value=mock_client,
        ),
        pytest.raises(ValueError, match="failed with status 401"),
    ):
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

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
            return_value=mock_client,
        ),
        pytest.raises(ValueError, match="non-object JSON"),
    ):
        await resolve_mcp_auth(server)


@pytest.mark.asyncio
async def test_client_credentials_uses_client_secret_basic_when_configured():
    """LIT-4091: a client_credentials server with token_endpoint_auth_method=client_secret_basic
    authenticates via HTTP Basic and keeps the secret out of the form body."""
    import base64

    server = _server(server_id="srv-basic", token_endpoint_auth_method="client_secret_basic")
    mock_client = AsyncMock()
    mock_client.post.return_value = _token_response("m2m-basic")

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth2_token_cache.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await resolve_mcp_auth(server)

    assert result == "m2m-basic"
    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["Authorization"] == "Basic " + base64.b64encode(b"cid:csec").decode()
    assert "client_secret" not in kwargs["data"]
    assert "client_id" not in kwargs["data"]
    assert kwargs["data"]["grant_type"] == "client_credentials"


def test_storage_ttl_capped_at_token_lifetime():
    """A token_storage_ttl_seconds longer than the token's own lifetime must be capped at
    expires_in minus the expiry buffer. Before the cap, the configured TTL won outright and the
    Redis fast path (which never re-checks expires_at) kept serving the dead token until eviction,
    while the stored refresh_token sat unused because refresh only runs on the DB read-through."""
    from litellm.constants import MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
        _compute_per_user_token_ttl,
    )

    server = _server(oauth2_flow=None, token_storage_ttl_seconds=604800)
    assert _compute_per_user_token_ttl(server, expires_in=86400) == 86400 - MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS


def test_storage_ttl_shorter_than_token_lifetime_wins():
    """A configured TTL below the token lifetime is the operative value: the knob's purpose is to
    force earlier DB re-checks (staleness backstop), so the shorter side must win the min()."""
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
        _compute_per_user_token_ttl,
    )

    server = _server(oauth2_flow=None, token_storage_ttl_seconds=3600)
    assert _compute_per_user_token_ttl(server, expires_in=86400) == 3600


def test_storage_ttl_verbatim_when_token_lifetime_unknown():
    """With no expires_in from the upstream there is nothing to cap against, so the configured
    TTL applies as-is (matching the pre-cap behavior for lifetime-less tokens)."""
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
        _compute_per_user_token_ttl,
    )

    server = _server(oauth2_flow=None, token_storage_ttl_seconds=604800)
    assert _compute_per_user_token_ttl(server, expires_in=None) == 604800


def test_storage_ttl_floors_at_one_second_for_nearly_dead_token():
    """A token already inside the expiry buffer yields the 1-second floor, not zero or a negative
    TTL, mirroring the floor the default (unconfigured) path has always had."""
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
        _compute_per_user_token_ttl,
    )

    server = _server(oauth2_flow=None, token_storage_ttl_seconds=3600)
    assert _compute_per_user_token_ttl(server, expires_in=30) == 1


def test_default_ttl_paths_unchanged_without_storage_ttl():
    """With token_storage_ttl_seconds unset the TTL still derives from expires_in minus the
    buffer, and falls back to MCP_PER_USER_TOKEN_DEFAULT_TTL when expires_in is absent."""
    from litellm.constants import (
        MCP_PER_USER_TOKEN_DEFAULT_TTL,
        MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS,
    )
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
        _compute_per_user_token_ttl,
    )

    server = _server(oauth2_flow=None)
    assert _compute_per_user_token_ttl(server, expires_in=86400) == 86400 - MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS
    assert _compute_per_user_token_ttl(server, expires_in=None) == MCP_PER_USER_TOKEN_DEFAULT_TTL
