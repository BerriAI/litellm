"""
Simple test to validate MCP auth header priority behavior.

Validates that:
1. auth_value is not required in config.yaml
2. Server-specific headers (x-mcp-server-name-authorization) take precedence over config auth_value
"""

import pytest
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp import MCPAuth, MCPTransport, MCPSpecVersion
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.mark.asyncio
async def test_mcp_server_works_without_config_auth_value():
    """
    Test that MCP servers work without auth_value in config when headers are provided.
    This validates that auth_value is truly optional in config.yaml.
    """
    # Create a server WITHOUT config auth_value
    server_without_config_auth = MCPServer(
        server_id="test-server-no-config",
        name="Test MCP Server No Config Auth",
        server_name="test_server_no_config",
        alias="test_no_config",
        url="https://api.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.authorization,
        authentication_token=None,  # No config auth
    )

    manager = MCPServerManager()

    # Test that it works with only header auth
    client = await manager._create_mcp_client(
        server=server_without_config_auth,
        mcp_auth_header="Bearer token_from_header_only",
    )

    # Verify header token is used
    assert client._mcp_auth_value == "Bearer token_from_header_only"
    assert client.auth_type == MCPAuth.authorization


@pytest.mark.parametrize("token_key", ["authentication_token", "auth_value"])
async def test_mcp_server_config_auth_value_header_used(token_key):
    """Ensure the configured auth token is emitted as the upstream Authorization header.

    The token is resolved through the v2 credential resolver and rides on the client's
    httpx.Auth, so assert the header it writes onto the request rather than the (now
    credential-free) _get_auth_headers() dict.
    """
    import httpx

    from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
        StaticHeaderAuth,
    )

    config = {
        "test_server": {
            "url": "https://api.example.com/mcp",
            "transport": "http",
            "auth_type": "bearer_token",
            token_key: "example_token",
        }
    }

    manager = MCPServerManager()
    await manager.load_servers_from_config(config)

    server = next(iter(manager.config_mcp_servers.values()))
    client = await manager._create_mcp_client(server)

    assert isinstance(client._resolved_auth, StaticHeaderAuth)
    emitted = next(client._resolved_auth.auth_flow(httpx.Request("POST", server.url)))
    assert emitted.headers["Authorization"] == "Bearer example_token"
    assert client.auth_type == MCPAuth.bearer_token


@pytest.mark.asyncio
async def test_dcr_bridge_oauth_delegate_uses_admission_injected_credential():
    """A dcr_bridge + oauth_delegate server's mcp_auth_header must not be discarded.

    ``_admit_dcr_bridge_delegate`` opens the caller's signed envelope during admission and
    injects the real upstream credential it sealed as ``mcp_auth_header`` here -- it is not an
    arbitrary caller-supplied override. oauth_delegate maps to PassthroughConfig in the v2
    resolver (see outbound_credentials/adapter.py), which used to be exempted from the v1
    override path regardless of origin, silently discarding this credential and leaving the
    upstream MCP server with no Authorization header at all (regression: BerriAI/litellm#36358).
    """
    server = MCPServer(
        server_id="test-dcr-bridge-server",
        name="Test DCR Bridge Server",
        server_name="test_dcr_bridge_server",
        alias="test_dcr_bridge",
        url="https://internal-mcp-server.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth_delegate,
        dcr_bridge=True,
    )

    manager = MCPServerManager()

    client = await manager._create_mcp_client(
        server=server,
        mcp_auth_header="Bearer unsealed_upstream_token",
    )

    assert client._mcp_auth_value == "Bearer unsealed_upstream_token"
