import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, '../../../')

from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPStdioConfig, MCPTransport


class TestMCPClient:
    """Test MCP Client stdio functionality"""

    def test_mcp_client_stdio_init(self):
        """Test MCPClient initialization with stdio config"""
        stdio_config = MCPStdioConfig(
            command="python",
            args=["-m", "my_mcp_server"],
            env={"DEBUG": "1"}
        )
        
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            stdio_config=stdio_config
        )
        
        assert client.transport_type == MCPTransport.stdio
        assert client.stdio_config == stdio_config
        assert client.stdio_config["command"] == "python"
        assert client.stdio_config["args"] == ["-m", "my_mcp_server"]

    @pytest.mark.asyncio
    async def test_mcp_client_stdio_connect_error(self):
        """Test MCP client stdio connection error handling"""
        # Test missing stdio_config
        client = MCPClient(transport_type=MCPTransport.stdio)
        
        with pytest.raises(ValueError, match="stdio_config is required for stdio transport"):
            await client.connect()

    @pytest.mark.asyncio
    @patch('litellm.experimental_mcp_client.client.stdio_client')
    @patch('litellm.experimental_mcp_client.client.ClientSession')
    async def test_mcp_client_stdio_connect_success(self, mock_session, mock_stdio_client):
        """Test successful stdio connection"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_stdio_client.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
        
        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.initialize = AsyncMock()
        mock_session.return_value = mock_session_instance
        
        stdio_config = MCPStdioConfig(
            command="python",
            args=["-m", "my_mcp_server"],
            env={"DEBUG": "1"}
        )
        
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            stdio_config=stdio_config
        )
        
        await client.connect()
        
        # Verify stdio_client was called with correct parameters
        mock_stdio_client.assert_called_once()
        call_args = mock_stdio_client.call_args[0][0]
        assert call_args.command == "python"
        assert call_args.args == ["-m", "my_mcp_server"]
        assert call_args.env == {"DEBUG": "1"}


if __name__ == "__main__":
    pytest.main([__file__]) 