"""Unit tests for the MCP OAuth pass-through patch (EAI-506 / Idea G).

Covers:
- `MCPServer.is_oauth_passthrough` property semantics.
- `/.well-known/oauth-protected-resource/...` pass-through branch (proxies
  upstream metadata, normalizes the `resource` field, caches, and surfaces
  network errors as HTTP 502).
- `MCPServerManager._fetch_tools_with_timeout` converting upstream 401s into
  `MCPUpstreamAuthError` for pass-through servers while keeping the silent
  empty-list fallback for gateway-managed / aggregator paths.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request

sys.path.insert(0, "../../../../../")


from litellm.proxy._experimental.mcp_server import discoverable_endpoints
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    _OAUTH_METADATA_CACHE,
    _build_oauth_protected_resource_response,
)
from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _extract_upstream_auth_failure,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.fixture(autouse=True)
def _mock_mcp_client_ip():
    """Bypass IP-based access control in tests."""
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints"
        ".IPAddressUtils.get_mcp_client_ip",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    """Prevent cross-test cache bleed for the oauth-protected-resource TTL cache."""
    _OAUTH_METADATA_CACHE.clear()
    yield
    _OAUTH_METADATA_CACHE.clear()


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
    )
    assert server.is_oauth_passthrough is True


def test_is_oauth_passthrough_true_when_auth_type_none_and_mixed_case_header():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=None,
        extra_headers=["authorization", "x-request-id"],
    )
    assert server.is_oauth_passthrough is True


def test_is_oauth_passthrough_false_for_oauth2_server():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        extra_headers=["Authorization"],
    )
    assert server.is_oauth_passthrough is False


def test_is_oauth_passthrough_false_without_authorization_header():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["x-api-key"],
    )
    assert server.is_oauth_passthrough is False


def test_is_oauth_passthrough_false_without_extra_headers():
    server = MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
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
        name="jet_knowledge_qa",
        server_name="jet_knowledge_qa",
        alias="jet_knowledge_qa",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = (
        passthrough_server
    )

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

    with patch.object(
        discoverable_endpoints, "get_async_httpx_client", return_value=mock_client
    ):
        result = await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="jet_knowledge_qa",
            use_standard_pattern=True,
        )

    assert result["authorization_servers"] == [
        "https://okta.example.com/oauth2/default"
    ]
    # resource is normalized to the gateway URL so bearers are sent back to us
    assert result["resource"].endswith("/mcp/jet_knowledge_qa")
    assert result["scopes_supported"] == ["openid", "profile"]


@pytest.mark.asyncio
async def test_oauth_protected_resource_passthrough_cache_hit():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    passthrough_server = MCPServer(
        server_id="passthrough-2",
        name="jet_knowledge_qa",
        server_name="jet_knowledge_qa",
        alias="jet_knowledge_qa",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = (
        passthrough_server
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "authorization_servers": ["https://okta.example.com"],
    }
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch.object(
        discoverable_endpoints, "get_async_httpx_client", return_value=mock_client
    ):
        await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="jet_knowledge_qa",
            use_standard_pattern=True,
        )
        await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="jet_knowledge_qa",
            use_standard_pattern=True,
        )

    assert mock_client.get.await_count == 1


@pytest.mark.asyncio
async def test_oauth_protected_resource_passthrough_network_error_returns_502():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    passthrough_server = MCPServer(
        server_id="passthrough-3",
        name="jet_knowledge_qa",
        server_name="jet_knowledge_qa",
        alias="jet_knowledge_qa",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = (
        passthrough_server
    )

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

    with patch.object(
        discoverable_endpoints, "get_async_httpx_client", return_value=mock_client
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _build_oauth_protected_resource_response(
                request=_make_request(),
                mcp_server_name="jet_knowledge_qa",
                use_standard_pattern=True,
            )

    assert exc_info.value.status_code == 502


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

    with patch.object(
        discoverable_endpoints, "get_async_httpx_client", return_value=mock_client
    ):
        result = await _build_oauth_protected_resource_response(
            request=_make_request(),
            mcp_server_name="keycloak_whoami",
            use_standard_pattern=True,
        )

    mock_client.get.assert_not_awaited()
    assert result["authorization_servers"] == [
        "https://gateway.example.com/keycloak_whoami"
    ]
    assert result["scopes_supported"] == ["read"]


# --------------------------------------------------------------------------
# _extract_upstream_auth_failure helper
# --------------------------------------------------------------------------


def test_extract_upstream_auth_failure_finds_401_in_http_status_error():
    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": 'Bearer resource_metadata="https://x"'},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    exc = httpx.HTTPStatusError("401", request=response.request, response=response)

    result = _extract_upstream_auth_failure(exc)
    assert result == (401, 'Bearer resource_metadata="https://x"')


def test_extract_upstream_auth_failure_walks_exception_group():
    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": "Bearer"},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    inner = httpx.HTTPStatusError("401", request=response.request, response=response)

    try:
        raise ExceptionGroup("wrapped", [inner])  # noqa: F821 (PEP 654, py3.11+)
    except Exception as group:
        result = _extract_upstream_auth_failure(group)

    assert result == (401, "Bearer")


def test_extract_upstream_auth_failure_returns_none_for_non_auth():
    assert _extract_upstream_auth_failure(RuntimeError("boom")) is None


# --------------------------------------------------------------------------
# _fetch_tools_with_timeout pass-through behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_tools_from_passthrough_raises_on_upstream_401():
    manager = MCPServerManager()
    passthrough_server = MCPServer(
        server_id="p1",
        name="jet_knowledge_qa",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
    )

    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": 'Bearer resource_metadata="https://upstream"'},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    upstream_error = httpx.HTTPStatusError(
        "401", request=response.request, response=response
    )

    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(side_effect=upstream_error)

    with pytest.raises(MCPUpstreamAuthError) as exc_info:
        await manager._fetch_tools_with_timeout(
            mock_client, passthrough_server.name, server=passthrough_server
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.www_authenticate == (
        'Bearer resource_metadata="https://upstream"'
    )
    assert exc_info.value.server_name == "jet_knowledge_qa"
    mock_client.list_tools.assert_awaited_with(raise_on_error=True)


@pytest.mark.asyncio
async def test_fetch_tools_from_passthrough_returns_tools_on_success():
    manager = MCPServerManager()
    passthrough_server = MCPServer(
        server_id="p1",
        name="jet_knowledge_qa",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
    )

    # list_tools returns a pre-baked tools list directly (MCPClient contract).
    tool = MagicMock()
    tool.name = "list_knowledge_bases"
    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(return_value=[tool])

    tools = await manager._fetch_tools_with_timeout(
        mock_client, passthrough_server.name, server=passthrough_server
    )
    assert tools == [tool]


@pytest.mark.asyncio
async def test_fetch_tools_from_gateway_managed_swallows_errors():
    """Regression guard: non-pass-through servers keep returning [] on errors
    so the multi-server aggregator isn't tainted by a single bad server."""
    manager = MCPServerManager()
    oauth2_server = MCPServer(
        server_id="o1",
        name="keycloak_whoami",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
    )

    response = httpx.Response(
        status_code=401,
        headers={},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    upstream_error = httpx.HTTPStatusError(
        "401", request=response.request, response=response
    )
    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(side_effect=upstream_error)

    tools = await manager._fetch_tools_with_timeout(
        mock_client, oauth2_server.name, server=oauth2_server
    )
    assert tools == []
    mock_client.list_tools.assert_awaited_with(raise_on_error=False)


# --------------------------------------------------------------------------
# §2.1 — Admission cold-start: 401 + matching resource_metadata URL
# --------------------------------------------------------------------------


def _make_scope(path: str, headers: list = None) -> dict:
    """Build a minimal ASGI HTTP scope for testing."""
    raw_headers = [(k.encode(), v.encode()) for k, v in (headers or [])]
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": raw_headers,
        "query_string": b"",
        "server": ("localhost", 4000),
        "scheme": "http",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,expected_metadata_path",
    [
        (
            "/mcp/jet_knowledge_qa",
            "/.well-known/oauth-protected-resource/mcp/jet_knowledge_qa",
        ),
        (
            "/jet_knowledge_qa/mcp",
            "/.well-known/oauth-protected-resource/jet_knowledge_qa/mcp",
        ),
    ],
)
async def test_passthrough_cold_start_emits_401_with_matching_resource_metadata(
    route, expected_metadata_path
):
    """No auth headers on a pass-through server route → 401 with resource_metadata URL
    that matches the inbound path so RFC 9728 §3.2 strict clients accept it."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        _is_mcp_passthrough_cold_start,
        _parse_mcp_server_names_from_path,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    passthrough_server = MCPServer(
        server_id="pt-cold-start",
        name="jet_knowledge_qa",
        server_name="jet_knowledge_qa",
        alias="jet_knowledge_qa",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = (
        passthrough_server
    )

    # For /mcp/{name}: scope path stays as-is.
    # For /{name}/mcp: dynamic_mcp_route rewrites to /mcp/{name} and sets _original_path.
    if route.startswith("/mcp/"):
        scope = _make_scope(route)
    else:
        scope = _make_scope(f"/mcp/jet_knowledge_qa")
        scope["_original_path"] = route

    # Verify cold-start detection fires for this path
    effective_path = scope.get("_original_path") or scope.get("path", "")
    servers = _parse_mcp_server_names_from_path(
        scope.get("path", "")  # always /mcp/{name} by the time admission runs
    )
    assert _is_mcp_passthrough_cold_start(scope, servers) is True

    # Verify resource_metadata_url form selection
    server_name = "jet_knowledge_qa"
    base_url = "http://localhost:4000"
    path = scope.get("_original_path") or scope.get("path", "") or ""
    if path.startswith(f"/{server_name}/mcp"):
        resource_metadata_url = (
            f"{base_url}/.well-known/oauth-protected-resource/{server_name}/mcp"
        )
    else:
        resource_metadata_url = (
            f"{base_url}/.well-known/oauth-protected-resource/mcp/{server_name}"
        )

    assert resource_metadata_url == f"{base_url}{expected_metadata_path}", (
        f"resource_metadata_url {resource_metadata_url!r} does not match "
        f"expected {base_url + expected_metadata_path!r}"
    )


# --------------------------------------------------------------------------
# §2.1 — Regression: non-pass-through servers bypass is NOT applied
# --------------------------------------------------------------------------


def test_is_mcp_passthrough_cold_start_false_for_oauth2_server():
    """Gateway-managed OAuth2 servers must not trigger the cold-start bypass."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        _is_mcp_passthrough_cold_start,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="oauth2-cold",
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
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    scope = _make_scope("/mcp/keycloak_whoami")
    result = _is_mcp_passthrough_cold_start(scope, ["keycloak_whoami"])
    assert result is False


def test_is_mcp_passthrough_cold_start_false_for_empty_servers():
    """Aggregate /mcp route (no server list) must not trigger bypass."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        _is_mcp_passthrough_cold_start,
    )

    scope = _make_scope("/mcp")
    assert _is_mcp_passthrough_cold_start(scope, None) is False
    assert _is_mcp_passthrough_cold_start(scope, []) is False


# --------------------------------------------------------------------------
# §2.1 — _parse_mcp_server_names_from_path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/mcp/jet_knowledge_qa", ["jet_knowledge_qa"]),
        ("/mcp/jet_knowledge_qa/tools/list", ["jet_knowledge_qa"]),
        ("/jet_knowledge_qa/mcp", ["jet_knowledge_qa"]),
        ("/jet_knowledge_qa/mcp/tools/list", ["jet_knowledge_qa"]),
        ("/mcp", None),
        ("/mcp/", None),
        ("/other/path", None),
    ],
)
def test_parse_mcp_server_names_from_path(path, expected):
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        _parse_mcp_server_names_from_path,
    )

    assert _parse_mcp_server_names_from_path(path) == expected
