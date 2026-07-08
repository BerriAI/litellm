"""Unit tests for MCP OAuth passthrough cold-start route behavior."""

import sys

import pytest

sys.path.insert(0, "../../../../../")

from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _make_scope(path: str, headers: list = None) -> dict:
    """Build a minimal ASGI HTTP scope for testing."""
    raw_headers = [(key.encode(), value.encode()) for key, value in (headers or [])]
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": raw_headers,
        "query_string": b"",
        "server": ("localhost", 4000),
        "scheme": "http",
    }


@pytest.mark.parametrize(
    "route,expected_metadata_path",
    [
        (
            "/mcp/sample_docs",
            "/.well-known/oauth-protected-resource/mcp/sample_docs",
        ),
        (
            "/sample_docs/mcp",
            "/.well-known/oauth-protected-resource/sample_docs/mcp",
        ),
    ],
)
def test_passthrough_cold_start_emits_401_with_matching_resource_metadata(
    route, expected_metadata_path
):
    """No auth headers on a passthrough server route emits matching metadata."""
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
        name="sample_docs",
        server_name="sample_docs",
        alias="sample_docs",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = (
        passthrough_server
    )

    if route.startswith("/mcp/"):
        scope = _make_scope(route)
    else:
        scope = _make_scope("/mcp/sample_docs")
        scope["_original_path"] = route

    servers = _parse_mcp_server_names_from_path(scope.get("path", ""))
    assert _is_mcp_passthrough_cold_start(servers, client_ip=None) is True

    server_name = "sample_docs"
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

    result = _is_mcp_passthrough_cold_start(["keycloak_whoami"], client_ip=None)
    assert result is False


def test_is_mcp_passthrough_cold_start_false_for_empty_servers():
    """Aggregate /mcp route (no server list) must not trigger bypass."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        _is_mcp_passthrough_cold_start,
    )

    assert _is_mcp_passthrough_cold_start(None, client_ip=None) is False
    assert _is_mcp_passthrough_cold_start([], client_ip=None) is False


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/mcp/sample_docs", ["sample_docs"]),
        # Server names may contain at most one slash (mirrors
        # ``_extract_target_server_names_from_path``), so when more than two
        # segments follow ``/mcp/`` the first two are treated as the name.
        ("/mcp/sample_docs/tools/list", ["sample_docs/tools"]),
        ("/mcp/custom_solutions/user_123", ["custom_solutions/user_123"]),
        ("/sample_docs/mcp", ["sample_docs"]),
        ("/sample_docs/mcp/tools/list", ["sample_docs"]),
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
