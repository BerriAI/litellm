"""Unit tests for MCP OAuth passthrough tool-fetch behavior."""

import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

sys.path.insert(0, "../../../../../")

from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _extract_upstream_auth_failure,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


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


@pytest.mark.asyncio
async def test_fetch_tools_from_passthrough_raises_on_upstream_401():
    manager = MCPServerManager()
    passthrough_server = MCPServer(
        server_id="p1",
        name="sample_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
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
    assert exc_info.value.server_name == "sample_docs"
    mock_client.list_tools.assert_awaited_with(raise_on_error=True)


@pytest.mark.asyncio
async def test_fetch_tools_from_delegated_oauth2_raises_on_upstream_401():
    manager = MCPServerManager()
    delegated_server = MCPServer(
        server_id="oauth1",
        name="delegated_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        delegate_auth_to_upstream=True,
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
            mock_client, delegated_server.name, server=delegated_server
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.www_authenticate == (
        'Bearer resource_metadata="https://upstream"'
    )
    assert exc_info.value.server_name == "delegated_docs"
    mock_client.list_tools.assert_awaited_with(raise_on_error=True)


@pytest.mark.asyncio
async def test_fetch_tools_from_client_credentials_oauth2_keeps_swallow_behavior():
    manager = MCPServerManager()
    m2m_server = MCPServer(
        server_id="oauth-m2m",
        name="m2m_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        delegate_auth_to_upstream=True,
        oauth2_flow="client_credentials",
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

    tools = await manager._fetch_tools_with_timeout(
        mock_client, m2m_server.name, server=m2m_server
    )

    assert tools == []
    mock_client.list_tools.assert_awaited_with(raise_on_error=False)


@pytest.mark.asyncio
async def test_fetch_tools_from_passthrough_returns_tools_on_success():
    manager = MCPServerManager()
    passthrough_server = MCPServer(
        server_id="p1",
        name="sample_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )

    tool = MagicMock()
    tool.name = "list_documents"
    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(return_value=[tool])

    tools = await manager._fetch_tools_with_timeout(
        mock_client, passthrough_server.name, server=passthrough_server
    )
    assert tools == [tool]


def test_to_http_exception_preserves_upstream_www_authenticate():
    err = MCPUpstreamAuthError(
        status_code=401,
        www_authenticate='Bearer resource_metadata="https://upstream/.well-known/oauth-protected-resource"',
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception()
    assert http_exc.status_code == 401
    assert http_exc.headers == {
        "www-authenticate": 'Bearer resource_metadata="https://upstream/.well-known/oauth-protected-resource"'
    }


def test_to_http_exception_skips_fabrication_when_base_url_missing():
    """Without ``base_url`` we cannot build an RFC 9728 §3.2-compliant absolute
    URI, so we omit the fabricated ``WWW-Authenticate`` challenge entirely
    instead of emitting a relative URI strict clients reject."""
    err = MCPUpstreamAuthError(
        status_code=401,
        www_authenticate=None,
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception()
    assert http_exc.status_code == 401
    assert http_exc.headers is None


def test_to_http_exception_fabricates_absolute_resource_metadata_with_base_url():
    err = MCPUpstreamAuthError(
        status_code=401,
        www_authenticate=None,
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception(base_url="https://gateway.example.com/")
    assert http_exc.status_code == 401
    assert http_exc.headers == {
        "www-authenticate": 'Bearer resource_metadata="https://gateway.example.com/.well-known/oauth-protected-resource/mcp/sample_docs"'
    }


def test_to_http_exception_skips_challenge_for_non_401_status():
    err = MCPUpstreamAuthError(
        status_code=403,
        www_authenticate=None,
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception()
    assert http_exc.status_code == 403
    assert http_exc.headers is None


@pytest.mark.asyncio
async def test_fetch_tools_from_gateway_managed_swallows_errors():
    """Regression guard: non-pass-through servers keep returning [] on errors."""
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


def _http_server(server_id: str, name: str, **kwargs) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=name,
        url=f"https://{name}/mcp",
        transport=MCPTransport.http,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_aggregate_list_tools_absorbs_one_unauthenticated_server():
    """Regression: across the aggregate (/mcp), a delegate/passthrough server that raises
    MCPUpstreamAuthError must not empty every other server's tools. Re-raising it on the
    aggregate path (introduced with the passthrough feature) zeroed the whole list because the
    fan-out gather propagated it."""
    from unittest.mock import patch

    from mcp.types import Tool as MCPTool

    from litellm.proxy._experimental.mcp_server import server as mcp_server
    from litellm.proxy._types import UserAPIKeyAuth

    delegate = _http_server(
        "s1", "delegate_docs", auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
    )
    working = _http_server("s2", "working_docs", auth_type=MCPAuth.none)
    good_tool = MCPTool(name="working_docs-read", description="d", inputSchema={"type": "object"})

    async def fake_get_tools(server, **kwargs):
        if server.server_id == delegate.server_id:
            raise MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name=server.name)
        return [good_tool]

    with patch.object(mcp_server, "_get_allowed_mcp_servers", AsyncMock(return_value=[delegate, working])), patch.object(
        mcp_server, "_prefetch_oauth_creds_for_user", AsyncMock(return_value={})
    ), patch.object(mcp_server, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None))), patch.object(
        mcp_server, "_get_user_oauth_extra_headers_from_db", AsyncMock(return_value=None)
    ), patch.object(
        mcp_server, "filter_tools_by_key_team_permissions", AsyncMock(side_effect=lambda tools, **k: tools)
    ), patch.object(
        mcp_server.global_mcp_server_manager, "_get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
    ):
        tools = await mcp_server._get_tools_from_mcp_servers(
            user_api_key_auth=UserAPIKeyAuth(token="h", user_id="u1"),
            mcp_auth_header=None,
            mcp_servers=None,
        )

    assert [t.name for t in tools] == ["working_docs-read"]


@pytest.mark.asyncio
async def test_single_server_route_also_absorbs_upstream_auth_error():
    """A single-server route (/<server>/mcp) absorbs an upstream-auth error just like the aggregate:
    the failing server is omitted (empty list) rather than re-raised. Surfacing it to the client as a
    401 + WWW-Authenticate challenge cannot be done from this list handler — the MCP session manager
    serializes a raise into a JSON-RPC error, not an HTTP 401 — so re-auth surfacing is handled by a
    request-scope preemptive check, tracked separately."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server import server as mcp_server
    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_gateway_server_name
    from litellm.proxy._types import UserAPIKeyAuth

    delegate = _http_server(
        "s1", "delegate_docs", auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
    )

    async def fake_get_tools(server, **kwargs):
        raise MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name=server.name)

    # /<server>/mcp sets the path-derived single-server scope; absorption must hold even then.
    token = _mcp_gateway_server_name.set("delegate_docs")
    try:
        with patch.object(mcp_server, "_get_allowed_mcp_servers", AsyncMock(return_value=[delegate])), patch.object(
            mcp_server, "_prefetch_oauth_creds_for_user", AsyncMock(return_value={})
        ), patch.object(mcp_server, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None))), patch.object(
            mcp_server, "_get_user_oauth_extra_headers_from_db", AsyncMock(return_value=None)
        ), patch.object(
            mcp_server.global_mcp_server_manager, "_get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
        ):
            tools = await mcp_server._get_tools_from_mcp_servers(
                user_api_key_auth=UserAPIKeyAuth(token="h", user_id="u1"),
                mcp_auth_header=None,
                mcp_servers=["delegate_docs"],
            )
        assert tools == []
    finally:
        _mcp_gateway_server_name.reset(token)


@pytest.mark.asyncio
async def test_aggregate_with_single_accessible_server_still_absorbs():
    """Regression for the route-misclassification: an aggregate request (/mcp, mcp_servers=None)
    from a key that can access exactly one server must still absorb that server's
    MCPUpstreamAuthError, not surface it. Keying the surface decision off the allowed count rather
    than the request filter would re-raise here and leave the aggregate broken for one-server
    permission sets."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server import server as mcp_server
    from litellm.proxy._types import UserAPIKeyAuth

    delegate = _http_server(
        "s1", "delegate_docs", auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
    )

    async def fake_get_tools(server, **kwargs):
        raise MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name=server.name)

    with patch.object(mcp_server, "_get_allowed_mcp_servers", AsyncMock(return_value=[delegate])), patch.object(
        mcp_server, "_prefetch_oauth_creds_for_user", AsyncMock(return_value={})
    ), patch.object(mcp_server, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None))), patch.object(
        mcp_server, "_get_user_oauth_extra_headers_from_db", AsyncMock(return_value=None)
    ), patch.object(
        mcp_server.global_mcp_server_manager, "_get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
    ):
        # Aggregate route: no explicit server filter, even though only one server is accessible.
        tools = await mcp_server._get_tools_from_mcp_servers(
            user_api_key_auth=UserAPIKeyAuth(token="h", user_id="u1"),
            mcp_auth_header=None,
            mcp_servers=None,
        )

    assert tools == []
