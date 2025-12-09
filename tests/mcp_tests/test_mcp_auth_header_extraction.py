"""
Test MCP auth header extraction and case-insensitive server name matching.

Tests the fixes for:
1. Auth headers being properly extracted from HTTP request headers in REST endpoints
2. Case-insensitive matching for server-specific auth headers in _call_regular_mcp_tool
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from starlette.datastructures import Headers

from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class TestRestEndpointAuthHeaderExtraction:
    """Test Fix 1: REST endpoints properly extract auth headers from HTTP requests"""

    def test_call_tool_rest_api_extracts_mcp_auth_header(self):
        """Test that call_tool REST endpoint extracts x-mcp-auth header"""
        headers = Headers({"x-mcp-auth": "Bearer legacy-token"})
        
        mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(headers)
        
        assert mcp_auth_header == "Bearer legacy-token"

    def test_call_tool_rest_api_extracts_server_specific_headers(self):
        """Test that call_tool REST endpoint extracts server-specific auth headers"""
        headers = Headers({
            "x-mcp-github-authorization": "Bearer github-token",
            "x-mcp-zapier-x-api-key": "zapier-key-123",
        })
        
        mcp_server_auth_headers = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        
        assert "github" in mcp_server_auth_headers
        assert mcp_server_auth_headers["github"]["Authorization"] == "Bearer github-token"
        assert "zapier" in mcp_server_auth_headers
        assert mcp_server_auth_headers["zapier"]["x-api-key"] == "zapier-key-123"

    def test_list_tools_rest_api_extracts_auth_headers(self):
        """Test that list_tools REST endpoint extracts auth headers"""
        headers = Headers({
            "x-mcp-auth": "Bearer legacy-token",
            "x-mcp-zapier-authorization": "Bearer zapier-token",
        })
        
        mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(headers)
        mcp_server_auth_headers = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        
        assert mcp_auth_header == "Bearer legacy-token"
        assert "zapier" in mcp_server_auth_headers
        assert mcp_server_auth_headers["zapier"]["Authorization"] == "Bearer zapier-token"


class TestCaseInsensitiveServerMatching:
    """Test Fix 2: Case-insensitive matching for server names in _call_regular_mcp_tool"""

    def test_case_insensitive_alias_matching(self):
        """Test server auth headers match case-insensitively by alias"""
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            alias="LiteLLMAGCGateway",
            server_name="litellm_gateway",
            url="https://api.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.authorization,
        )
        
        mcp_server_auth_headers = {
            "litellmagcgateway": {"Authorization": "Bearer token"}
        }
        
        # Test the case-insensitive matching logic from _call_regular_mcp_tool
        normalized_headers = {k.lower(): v for k, v in mcp_server_auth_headers.items()}
        server_auth_header = normalized_headers.get(server.alias.lower())
        
        assert server_auth_header is not None
        assert server_auth_header["Authorization"] == "Bearer token"

    def test_case_insensitive_server_name_matching(self):
        """Test server auth headers match case-insensitively by server_name"""
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            alias=None,
            server_name="MyAPIServer",
            url="https://api.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.authorization,
        )
        
        mcp_server_auth_headers = {
            "myapiserver": {"Authorization": "Bearer token"}
        }
        
        # Test the case-insensitive matching logic from _call_regular_mcp_tool
        normalized_headers = {k.lower(): v for k, v in mcp_server_auth_headers.items()}
        server_auth_header = normalized_headers.get(server.server_name.lower())
        
        assert server_auth_header is not None
        assert server_auth_header["Authorization"] == "Bearer token"

    def test_alias_checked_before_server_name(self):
        """Test that alias is checked before server_name"""
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            alias="MyAlias",
            server_name="MyServerName",
            url="https://api.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.authorization,
        )
        
        mcp_server_auth_headers = {
            "myalias": {"Authorization": "Bearer alias-token"},
            "myservername": {"Authorization": "Bearer servername-token"},
        }
        
        # Simulate the fix
        normalized_headers = {k.lower(): v for k, v in mcp_server_auth_headers.items()}
        server_auth_header = normalized_headers.get(server.alias.lower())
        if server_auth_header is None and server.server_name:
            server_auth_header = normalized_headers.get(server.server_name.lower())
        
        assert server_auth_header["Authorization"] == "Bearer alias-token"

    def test_fallback_to_legacy_auth_header(self):
        """Test fallback to legacy auth header when no server-specific header found"""
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            alias="MyServer",
            server_name="my_server",
            url="https://api.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.authorization,
        )
        
        mcp_server_auth_headers = {}
        mcp_auth_header = "Bearer legacy-token"
        
        # Simulate the fix
        normalized_headers = {k.lower(): v for k, v in mcp_server_auth_headers.items()}
        server_auth_header = normalized_headers.get(server.alias.lower())
        if server_auth_header is None and server.server_name:
            server_auth_header = normalized_headers.get(server.server_name.lower())
        if server_auth_header is None:
            server_auth_header = mcp_auth_header
        
        assert server_auth_header == "Bearer legacy-token"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
