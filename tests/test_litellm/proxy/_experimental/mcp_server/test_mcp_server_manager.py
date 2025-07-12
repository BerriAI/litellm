import sys
from datetime import datetime
from unittest.mock import MagicMock

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


if __name__ == "__main__":
    pytest.main([__file__]) 