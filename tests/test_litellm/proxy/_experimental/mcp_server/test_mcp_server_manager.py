import sys
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

import pytest

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, '../../../../../')

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _deserialize_env_dict,
)
from litellm.proxy._types import LiteLLM_MCPServerTable, MCPSpecVersion, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class TestMCPServerManager:
    """Test MCP Server Manager stdio functionality"""

    def test_deserialize_env_dict(self):
        """Test environment dictionary deserialization"""
        # Test JSON string
        env_json = '{"PATH": "/usr/bin", "DEBUG": "1"}'
        result = _deserialize_env_dict(env_json)
        assert result == {"PATH": "/usr/bin", "DEBUG": "1"}
        
        # Test already dict
        env_dict = {"PATH": "/usr/bin", "DEBUG": "1"}
        result = _deserialize_env_dict(env_dict)
        assert result == {"PATH": "/usr/bin", "DEBUG": "1"}
        
        # Test invalid JSON
        invalid_json = '{"PATH": "/usr/bin", "DEBUG": 1'
        result = _deserialize_env_dict(invalid_json)
        assert result is None

    def test_add_update_server_stdio(self):
        """Test adding stdio MCP server"""
        manager = MCPServerManager()
        
        stdio_server = LiteLLM_MCPServerTable(
            server_id="stdio-server-1",
            alias="test_stdio_server",
            description="Test stdio server",
            url=None,
            transport=MCPTransport.stdio,
            spec_version=MCPSpecVersion.mar_2025,
            command="python",
            args=["-m", "server"],
            env={"DEBUG": "1", "TEST": "1"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        manager.add_update_server(stdio_server)
        
        # Verify server was added
        assert "stdio-server-1" in manager.registry
        added_server = manager.registry["stdio-server-1"]
        
        assert added_server.server_id == "stdio-server-1"
        assert added_server.name == "test_stdio_server"
        assert added_server.transport == MCPTransport.stdio
        assert added_server.command == "python"
        assert added_server.args == ["-m", "server"]
        assert added_server.env == {"DEBUG": "1", "TEST": "1"}

    def test_create_mcp_client_stdio(self):
        """Test creating MCP client for stdio transport"""
        manager = MCPServerManager()
        
        stdio_server = MCPServer(
            server_id="stdio-server-2",
            name="test_stdio_server",
            url=None,
            transport=MCPTransport.stdio,
            spec_version=MCPSpecVersion.mar_2025,
            command="node",
            args=["server.js"],
            env={"NODE_ENV": "test"}
        )
        
        client = manager._create_mcp_client(stdio_server)
        
        assert client.transport_type == MCPTransport.stdio
        assert client.stdio_config is not None
        assert client.stdio_config["command"] == "node"
        assert client.stdio_config["args"] == ["server.js"]
        assert client.stdio_config["env"] == {"NODE_ENV": "test"}

    @pytest.mark.asyncio
    async def test_list_tools_with_server_specific_auth_headers(self):
        """Test list_tools method with server-specific auth headers"""
        manager = MCPServerManager()
        
        # Mock servers
        server1 = MagicMock()
        server1.name = "github"
        server1.alias = "github"
        server1.server_name = "github"
        
        server2 = MagicMock()
        server2.name = "zapier"
        server2.alias = "zapier"
        server2.server_name = "zapier"
        
        # Mock get_allowed_mcp_servers to return our test servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github", "zapier"])
        manager.get_mcp_server_by_id = MagicMock(side_effect=lambda x: server1 if x == "github" else server2)
        
        # Mock _get_tools_from_server to return different results
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            if server.name == "github":
                tool1 = MagicMock()
                tool1.name = "github_tool_1"
                tool2 = MagicMock()
                tool2.name = "github_tool_2"
                return [tool1, tool2]
            else:
                tool1 = MagicMock()
                tool1.name = "zapier_tool_1"
                return [tool1]
        
        manager._get_tools_from_server = mock_get_tools_from_server
        
        # Test with server-specific auth headers
        mcp_server_auth_headers = {
            "github": "Bearer github-token",
            "zapier": "zapier-api-key"
        }
        
        result = await manager.list_tools(mcp_server_auth_headers=mcp_server_auth_headers)
        
        # Verify that both servers were called with their specific auth headers
        assert len(result) == 3  # 2 from github + 1 from zapier
        
        # Verify the tools have the expected names
        tool_names = [tool.name for tool in result]
        assert "github_tool_1" in tool_names
        assert "github_tool_2" in tool_names
        assert "zapier_tool_1" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_fallback_to_legacy_auth_header(self):
        """Test that list_tools falls back to legacy auth header when server-specific not available"""
        manager = MCPServerManager()
        
        # Mock server
        server = MagicMock()
        server.name = "github"
        server.alias = "github"
        server.server_name = "github"
        
        # Mock get_allowed_mcp_servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
        manager.get_mcp_server_by_id = MagicMock(return_value=server)
        
        # Mock _get_tools_from_server
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            assert mcp_auth_header == "legacy-token"  # Should use legacy header
            tool = MagicMock()
            tool.name = "github_tool_1"
            return [tool]
        
        manager._get_tools_from_server = mock_get_tools_from_server
        
        # Test with only legacy auth header (no server-specific headers)
        result = await manager.list_tools(
            mcp_auth_header="legacy-token",
            mcp_server_auth_headers={}  # Empty server-specific headers
        )
        
        assert len(result) == 1
        assert result[0].name == "github_tool_1"

    @pytest.mark.asyncio
    async def test_list_tools_prioritizes_server_specific_over_legacy(self):
        """Test that server-specific auth headers take priority over legacy header"""
        manager = MCPServerManager()
        
        # Mock server
        server = MagicMock()
        server.name = "github"
        server.alias = "github"
        server.server_name = "github"
        
        # Mock get_allowed_mcp_servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
        manager.get_mcp_server_by_id = MagicMock(return_value=server)
        
        # Mock _get_tools_from_server
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            assert mcp_auth_header == "server-specific-token"  # Should use server-specific header
            tool = MagicMock()
            tool.name = "github_tool_1"
            return [tool]
        
        manager._get_tools_from_server = mock_get_tools_from_server
        
        # Test with both legacy and server-specific headers
        result = await manager.list_tools(
            mcp_auth_header="legacy-token",
            mcp_server_auth_headers={"github": "server-specific-token"}
        )
        
        assert len(result) == 1
        assert result[0].name == "github_tool_1"

    @pytest.mark.asyncio
    async def test_list_tools_handles_missing_server_alias(self):
        """Test that list_tools handles servers without alias gracefully"""
        manager = MCPServerManager()
        
        # Mock server without alias
        server = MagicMock()
        server.name = "github"
        server.alias = None  # No alias
        server.server_name = "github"
        
        # Mock get_allowed_mcp_servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
        manager.get_mcp_server_by_id = MagicMock(return_value=server)
        
        # Mock _get_tools_from_server
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            assert mcp_auth_header == "server-specific-token"  # Should use server-specific header via server_name
            tool = MagicMock()
            tool.name = "github_tool_1"
            return [tool]
        
        manager._get_tools_from_server = mock_get_tools_from_server
        
        # Test with server-specific headers that match server_name (even without alias)
        result = await manager.list_tools(
            mcp_auth_header="legacy-token",
            mcp_server_auth_headers={"github": "server-specific-token"}
        )
        
        assert len(result) == 1
        assert result[0].name == "github_tool_1"


if __name__ == "__main__":
    pytest.main([__file__]) 