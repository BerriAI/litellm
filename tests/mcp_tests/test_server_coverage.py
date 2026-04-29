import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy._types import MCPTransportType
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.proxy._experimental.mcp_server.server import (
    _get_prompts_from_mcp_servers,
    _get_resources_from_mcp_servers,
    _get_resource_templates_from_mcp_servers,
    _get_tools_from_mcp_servers,
)

@pytest.mark.asyncio
async def test_get_prompts_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    server2 = MCPServer(
        server_id="test-2", name="test2", transport="stdio", url="http://test2"
    )
    mock_prompt = MagicMock()
    mock_prompt.name = "test_prompt"
    
    with patch("litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers", return_value=[server1, server2]):
        with patch("litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_prompts_from_server", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [[mock_prompt], Exception("Server error")]
            result = await _get_prompts_from_mcp_servers(
                user_api_key_auth=None,
                mcp_auth_header=None,
                mcp_servers=["test1", "test2"]
            )
            assert len(result) == 1
            assert result[0] == mock_prompt

@pytest.mark.asyncio
async def test_get_resources_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    mock_resource = MagicMock()
    mock_resource.name = "test_resource"
    
    with patch("litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers", return_value=[server1]):
        with patch("litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_resources_from_server", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [mock_resource]
            result = await _get_resources_from_mcp_servers(
                user_api_key_auth=None,
                mcp_auth_header=None,
                mcp_servers=["test1"]
            )
            assert len(result) == 1
            assert result[0] == mock_resource

@pytest.mark.asyncio
async def test_get_resource_templates_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    mock_template = MagicMock()
    mock_template.name = "test_template"
    
    with patch("litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers", return_value=[server1]):
        with patch("litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_resource_templates_from_server", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [mock_template]
            result = await _get_resource_templates_from_mcp_servers(
                user_api_key_auth=None,
                mcp_auth_header=None,
                mcp_servers=["test1"]
            )
            assert len(result) == 1
            assert result[0] == mock_template

@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    
    with patch("litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers", return_value=[server1]):
        with patch("litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager._get_tools_from_server", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [mock_tool]
            # test with some tracking headers
            result = await _get_tools_from_mcp_servers(
                user_api_key_auth=None,
                mcp_auth_header=None,
                mcp_servers=["test1"],
                log_list_tools_to_spendlogs=True,
                litellm_trace_id="test-trace"
            )
            assert len(result) == 1
            assert result[0] == mock_tool
