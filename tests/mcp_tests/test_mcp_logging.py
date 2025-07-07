import os
import sys
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch


sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server.server import (
   mcp_server_tool_call,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
   MCPServerManager,
)
from mcp.types import Tool as MCPTool, CallToolResult, TextContent


class TestMCPLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload = None
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("success event")
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)


@pytest.mark.asyncio
async def test_mcp_cost_tracking():
    # Create a mock tool call result
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")],
        isError=False
    )
    
    # Create a mock MCPClient
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(return_value=[
        MCPTool(
            name="add_tools",
            description="Test tool",
            inputSchema={"type": "object", "properties": {"test": {"type": "string"}}}
        )
    ])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient', mock_client_constructor):
        # Load the server config
        local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "zapier_gmail_server": {
                    "url": os.getenv("ZAPIER_MCP_HTTPS_SERVER_URL"),
                    "input_cost_per_request": 1.2
                }
            }
        )

        # Set up the test logger
        test_logger = TestMCPLogger()
        litellm.callbacks = [test_logger]

        # Initialize the tool mapping
        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()
        
        # Patch the global manager in both modules where it's used
        with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager', local_mcp_server_manager), \
             patch('litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager', local_mcp_server_manager):

            print("tool_name_to_mcp_server_name_mapping", local_mcp_server_manager.tool_name_to_mcp_server_name_mapping)

            # Call mcp tool
            response = await mcp_server_tool_call(
                name="zapier_gmail_server/add_tools",  # Use prefixed name
                arguments={
                    "test": "test"
                }
            )

            # wait 1-2 seconds for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload = test_logger.standard_logging_payload
            print("logged_standard_logging_payload", json.dumps(logged_standard_logging_payload, indent=4))
            
            # Add assertions
            assert response is not None
            response_list = list(response)  # Convert iterable to list
            assert len(response_list) == 1
            assert isinstance(response_list[0], TextContent)
            assert response_list[0].text == "Test response"
            
            # Verify client methods were called
            mock_client.__aenter__.assert_called()
            mock_client.call_tool.assert_called_once()