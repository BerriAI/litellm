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
