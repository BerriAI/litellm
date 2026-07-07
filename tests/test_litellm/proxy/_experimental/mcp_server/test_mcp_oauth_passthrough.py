"""Unit tests for MCP OAuth passthrough metadata behavior.

Covers:
- `MCPServer.is_oauth_passthrough` property semantics.
- `/.well-known/oauth-protected-resource/...` pass-through branch (proxies
  upstream metadata, normalizes the `resource` field, caches, and surfaces
  network errors as HTTP 502).
"""

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request

sys.path.insert(0, "../../../../../")


from litellm.proxy._experimental.mcp_server import discoverable_endpoints
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    _OAUTH_METADATA_CACHE,
    _OAUTH_METADATA_FETCH_LOCKS,
    _build_oauth_protected_resource_response,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.fixture(autouse=True)
def _mock_mcp_client_ip():
    """Bypass IP-based access control in tests."""
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    """Prevent cross-test cache bleed for the oauth-protected-resource TTL cache."""
    _OAUTH_METADATA_CACHE.clear()
    _OAUTH_METADATA_FETCH_LOCKS.clear()
    yield
    _OAUTH_METADATA_CACHE.clear()
    _OAUTH_METADATA_FETCH_LOCKS.clear()


def _make_request(base_url: str = "https://gateway.example.com/") -> Request:
    request = MagicMock(spec=Request)
    request.base_url = base_url
    request.headers = {}
    return request


# --------------------------------------------------------------------------
# is_oauth_passthrough property
# --------------------------------------------------------------------------


def test_is_oauth_passthrough_true_when_none_auth_and_authorization_header():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )
    assert server.is_oauth_passthrough is True


def test_is_oauth_passthrough_true_when_auth_type_none_and_mixed_case_header():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=None,
        extra_headers=["authorization", "x-request-id"],
        oauth_passthrough=True,
    )
    assert server.is_oauth_passthrough is True


def test_is_oauth_passthrough_false_for_oauth2_server():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )
    assert server.is_oauth_passthrough is False


def test_is_oauth_passthrough_false_without_authorization_header():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["x-api-key"],
        oauth_passthrough=True,
    )
    assert server.is_oauth_passthrough is False


def test_is_oauth_passthrough_false_without_extra_headers():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        oauth_passthrough=True,
    )
    assert server.is_oauth_passthrough is False


def test_is_oauth_passthrough_false_without_oauth_passthrough_flag():
    """The detection flag must be set explicitly. Without it, the legacy
    behavior is preserved for servers that forward Authorization for
    non-OAuth reasons (static bearer tokens, custom auth schemes)."""
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        # oauth_passthrough defaults to False
    )
    assert server.is_oauth_passthrough is False


def test_is_oauth_passthrough_false_when_oauth_passthrough_explicitly_false():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=False,
    )
    assert server.is_oauth_passthrough is False


def test_is_oauth_passthrough_false_when_only_delegate_auth_to_upstream_set():
    """Regression guard: ``delegate_auth_to_upstream`` is the oauth2-only
    PKCE-bypass flag and must NOT, on its own, turn a non-oauth2 server into
    an OAuth pass-through server. Pass-through requires the dedicated
    ``oauth_passthrough`` opt-in. This protects existing deployments that set
    ``delegate_auth_to_upstream`` from silently gaining pass-through behavior.
    """
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        delegate_auth_to_upstream=True,
        # oauth_passthrough intentionally left at its default (False)
    )
    assert server.is_oauth_passthrough is False


# --------------------------------------------------------------------------
# _build_oauth_protected_resource_response: pass-through branch
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oauth_protected_resource_passthrough_proxies_upstream_metadata():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    passthrough_server = MCPServer(
        server_id="passthrough-1",
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = passthrough_server

    upstream_payload = {
        "resource": "https://upstream.example.com/mcp",
        "authorization_servers": ["https://okta.example.com/oauth2/default"],
        "scopes_supported": ["openid", "profile"],
        "bearer_methods_supported": ["header"],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = upstream_payload
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
        result = await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="sample_docs",
            use_standard_pattern=True,
        )

    assert result["authorization_servers"] == ["https://okta.example.com/oauth2/default"]
    # resource is normalized to the gateway URL so bearers are sent back to us
    assert result["resource"].endswith("/mcp/sample_docs")
    assert result["scopes_supported"] == ["openid", "profile"]


@pytest.mark.asyncio
async def test_oauth_protected_resource_passthrough_cache_hit():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    passthrough_server = MCPServer(
        server_id="passthrough-2",
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = passthrough_server

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "authorization_servers": ["https://okta.example.com"],
    }
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
        await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="sample_docs",
            use_standard_pattern=True,
        )
        await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="sample_docs",
            use_standard_pattern=True,
        )

    assert mock_client.get.await_count == 1


def test_oauth_metadata_cache_prunes_to_max_size():
    now = 1_000_000.0
    max_size = discoverable_endpoints._OAUTH_METADATA_CACHE_MAX_SIZE

    for index in range(max_size + 10):
        _OAUTH_METADATA_CACHE[(f"server-{index}", f"https://upstream/{index}")] = (
            now + index + 1,
            {"index": index},
        )

    discoverable_endpoints._prune_oauth_metadata_cache(now)

    assert len(_OAUTH_METADATA_CACHE) == max_size
    assert ("server-0", "https://upstream/0") not in _OAUTH_METADATA_CACHE
    assert (
        f"server-{max_size + 9}",
        f"https://upstream/{max_size + 9}",
    ) in _OAUTH_METADATA_CACHE


def test_oauth_metadata_fetch_locks_pruned_alongside_cache():
    now = 1_000_000.0
    cached_key = ("server-active", "https://upstream/active")
    expired_key = ("server-expired", "https://upstream/expired")
    orphan_key = ("server-orphan", "https://upstream/orphan")

    _OAUTH_METADATA_CACHE[cached_key] = (now + 100, {"index": 0})
    _OAUTH_METADATA_CACHE[expired_key] = (now - 1, {"index": 1})

    _OAUTH_METADATA_FETCH_LOCKS[cached_key] = asyncio.Lock()
    _OAUTH_METADATA_FETCH_LOCKS[expired_key] = asyncio.Lock()
    _OAUTH_METADATA_FETCH_LOCKS[orphan_key] = asyncio.Lock()

    discoverable_endpoints._prune_oauth_metadata_cache(now)

    assert cached_key in _OAUTH_METADATA_FETCH_LOCKS
    assert expired_key not in _OAUTH_METADATA_FETCH_LOCKS
    assert orphan_key not in _OAUTH_METADATA_FETCH_LOCKS


@pytest.mark.asyncio
async def test_oauth_metadata_fetch_locks_held_lock_preserved_during_prune():
    held_key = ("server-busy", "https://upstream/busy")
    held_lock = asyncio.Lock()
    _OAUTH_METADATA_FETCH_LOCKS[held_key] = held_lock

    async with held_lock:
        discoverable_endpoints._prune_oauth_metadata_cache(time.time())
        assert held_key in _OAUTH_METADATA_FETCH_LOCKS


@pytest.mark.asyncio
async def test_oauth_metadata_cache_expired_entry_is_refetched():
    passthrough_server = MCPServer(
        server_id="expired-cache-server",
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )
    _OAUTH_METADATA_CACHE[(passthrough_server.server_id, passthrough_server.url)] = (
        0,
        {"authorization_servers": ["https://stale.example.com"]},
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "authorization_servers": ["https://fresh.example.com"],
    }
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
        result = await discoverable_endpoints.fetch_upstream_oauth_protected_resource(passthrough_server)

    assert result == {"authorization_servers": ["https://fresh.example.com"]}
    assert mock_client.get.await_count == 1


@pytest.mark.asyncio
async def test_oauth_protected_resource_passthrough_network_error_returns_502():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    passthrough_server = MCPServer(
        server_id="passthrough-3",
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = passthrough_server

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

    with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
        with pytest.raises(HTTPException) as exc_info:
            await _build_oauth_protected_resource_response(
                request=_make_request(),
                mcp_server_name="sample_docs",
                use_standard_pattern=True,
            )

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_fetch_upstream_metadata_returns_none_when_not_all_candidates_network_fail():
    passthrough_server = MCPServer(
        server_id="passthrough-partial-network",
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )

    not_found_response = MagicMock()
    not_found_response.status_code = 404
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=[not_found_response, httpx.ConnectError("path fallback failed")])

    with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
        result = await discoverable_endpoints.fetch_upstream_oauth_protected_resource(passthrough_server)

    assert result is None
    assert mock_client.get.await_count == 2


@pytest.mark.asyncio
async def test_oauth_protected_resource_gateway_managed_unchanged():
    """Regression guard: OAuth2 servers still advertise the gateway as AS."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="oauth2-1",
        name="keycloak_whoami",
        server_name="keycloak_whoami",
        alias="keycloak_whoami",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="cid",
        client_secret="cs",
        authorization_url="https://keycloak/auth",
        token_url="https://keycloak/token",
        scopes=["read"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # If the code mistakenly fetched upstream metadata for a gateway-managed
    # server, this spy would catch it.
    mock_client = MagicMock()
    mock_client.get = AsyncMock()

    with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
        result = await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="keycloak_whoami",
            use_standard_pattern=True,
        )

    mock_client.get.assert_not_awaited()
    assert result["authorization_servers"] == ["https://gateway.example.com/keycloak_whoami"]
    assert result["scopes_supported"] == ["read"]


def _make_upstream_metadata_client() -> tuple[dict, MagicMock]:
    upstream_payload = {
        "resource": "https://upstream.example.com/mcp",
        "authorization_servers": ["https://okta.example.com/oauth2/default"],
        "scopes_supported": ["openid", "profile"],
        "bearer_methods_supported": ["header"],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = upstream_payload
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    return upstream_payload, mock_client


@pytest.mark.asyncio
async def test_oauth_protected_resource_oauth_delegate_returns_upstream_metadata_verbatim():
    """oauth_delegate discovery must return the upstream metadata verbatim,
    resource included. The caller's token is forwarded to and validated by the
    upstream, so its audience must be the upstream; rewriting resource to the
    gateway would make a strict IdP refuse to mint it or the upstream reject it.
    A regression that dropped oauth_delegate from the pass-through predicate would
    fall through to the gateway-AS branch and advertise LiteLLM as the AS."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    delegate_server = MCPServer(
        server_id="delegate-1",
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth_delegate,
    )
    global_mcp_server_manager.registry[delegate_server.server_id] = delegate_server

    upstream_payload, mock_client = _make_upstream_metadata_client()
    try:
        with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
            result = await _build_oauth_protected_resource_response(
                request=_make_request(),
                mcp_server_name="sample_docs",
                use_standard_pattern=True,
            )

        assert result == upstream_payload
        assert result["authorization_servers"] == ["https://okta.example.com/oauth2/default"]
        assert result["resource"] == "https://upstream.example.com/mcp"
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_oauth_protected_resource_true_passthrough_returns_upstream_metadata_verbatim():
    """true_passthrough discovery must return the upstream metadata verbatim,
    resource included, so the client treats the upstream as the resource and
    authorizes directly against it. A regression that rewrote resource (the
    gateway-proxied behavior) would break the transparent-proxy contract."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    true_passthrough_server = MCPServer(
        server_id="tp-1",
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.true_passthrough,
    )
    global_mcp_server_manager.registry[true_passthrough_server.server_id] = true_passthrough_server

    upstream_payload, mock_client = _make_upstream_metadata_client()
    try:
        with patch.object(discoverable_endpoints, "get_async_httpx_client", return_value=mock_client):
            result = await _build_oauth_protected_resource_response(
                request=_make_request(),
                mcp_server_name="sample_docs",
                use_standard_pattern=True,
            )

        assert result == upstream_payload
        assert result["resource"] == "https://upstream.example.com/mcp"
    finally:
        global_mcp_server_manager.registry.clear()
