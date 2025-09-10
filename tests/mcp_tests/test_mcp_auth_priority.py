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


def test_mcp_server_works_without_config_auth_value():
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
        spec_version=MCPSpecVersion.jun_2025,
        auth_type=MCPAuth.authorization,
        authentication_token=None  # No config auth
    )
    
    manager = MCPServerManager()
    
    # Test that it works with only header auth
    client = manager._create_mcp_client(
        server=server_without_config_auth,
        mcp_auth_header="Bearer token_from_header_only",
        protocol_version="2025-06-18"
    )
    
    # Verify header token is used
    assert client._mcp_auth_value == "Bearer token_from_header_only"
    assert client.auth_type == MCPAuth.authorization
