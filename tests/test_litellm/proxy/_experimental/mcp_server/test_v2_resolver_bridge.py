"""Parity tests for the v1->v2 MCP resolver graft (none + api_key).

When the flag is on, resolve_mcp_auth routes none/api_key through the v2 resolver; the upstream
headers must be byte-identical to what v1 produces, and every other mode (or any v2 error) must
fall back to v1 unchanged.
"""

import pytest

from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._experimental.mcp_server.oauth2_token_cache import resolve_mcp_auth
from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
    resolve_v2_auth_value,
)
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

pytestmark = pytest.mark.asyncio

FLAG = "LITELLM_USE_V2_MCP_RESOLVER"


def _server(auth_type, token=None):
    return MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        url="https://up.example/mcp",
        auth_type=auth_type,
        authentication_token=token,
    )


def _v1_headers(auth_type, auth_value):
    """The final upstream headers v1's MCPClient would send for this resolved value."""
    return MCPClient(auth_type=auth_type, auth_value=auth_value)._get_auth_headers()


@pytest.fixture
def v2_on(monkeypatch):
    monkeypatch.setenv(FLAG, "true")


@pytest.fixture
def v2_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


async def test_flag_off_defers_to_v1(v2_off):
    assert await resolve_v2_auth_value(_server(MCPAuth.api_key, "k")) is None


async def test_api_key_parity(v2_on):
    token = "up-secret"
    server = _server(MCPAuth.api_key, token)
    v2_value = await resolve_v2_auth_value(server)
    assert v2_value == {"X-API-Key": token}
    # byte-identical to v1's final upstream headers
    assert _v1_headers(MCPAuth.api_key, token) == _v1_headers(MCPAuth.api_key, v2_value)


async def test_none_attaches_no_auth(v2_on):
    server = _server(MCPAuth.none)
    v2_value = await resolve_v2_auth_value(server)
    assert v2_value == {}
    # v1 none -> no auth header; v2 -> empty dict merged -> no auth header
    assert _v1_headers(MCPAuth.none, None) == _v1_headers(MCPAuth.none, v2_value) == {}


async def test_non_grafted_mode_defers_to_v1(v2_on):
    # bearer_token is not grafted yet -> v2 returns None so v1 handles it
    assert await resolve_v2_auth_value(_server(MCPAuth.bearer_token, "k")) is None


async def test_api_key_without_token_defers_to_v1(v2_on):
    assert await resolve_v2_auth_value(_server(MCPAuth.api_key, None)) is None


async def test_resolve_mcp_auth_hook_routes_api_key_when_on(v2_on):
    server = _server(MCPAuth.api_key, "up-secret")
    assert await resolve_mcp_auth(server) == {"X-API-Key": "up-secret"}


async def test_resolve_mcp_auth_hook_uses_v1_when_off(v2_off):
    # v1 returns the raw token string; MCPClient then maps it to the X-API-Key header
    server = _server(MCPAuth.api_key, "up-secret")
    assert await resolve_mcp_auth(server) == "up-secret"
