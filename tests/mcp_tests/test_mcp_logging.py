import os
import sys
import pytest
import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, patch


sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.types.utils import StandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server.server import (
   mcp_server_tool_call,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
   MCPServerManager,
)
from litellm.types.mcp import MCPPostCallResponseObject
from litellm.types.utils import HiddenParams
from mcp.types import Tool as MCPTool, CallToolResult, TextContent


class TestMCPLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload = None
        super().__init__()
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("success event")
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)
        print(f"Captured standard_logging_payload: {self.standard_logging_payload}")
    

@pytest.mark.asyncio
async def test_mcp_cost_tracking():
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
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
                    "mcp_info": {
                        "mcp_server_cost_info": {
                            "default_cost_per_query": 1.2,
                        }
                    }
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

            # Manually add the tool mapping to ensure it's available (since mocking might not capture it properly)
            local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["add_tools"] = "zapier_gmail_server"
            local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["zapier_gmail_server-add_tools"] = "zapier_gmail_server"

            # Call mcp tool
            response = await mcp_server_tool_call(
                name="zapier_gmail_server-add_tools",  # Use correct prefixed name with - separator
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

            ######
            # verify response cost is 1.2 as set on default_cost_per_query
            # Critical - the cost is tracked as $1.2
            assert logged_standard_logging_payload is not None, "Standard logging payload should not be None"
            assert logged_standard_logging_payload["response_cost"] == 1.2


@pytest.mark.asyncio
async def test_mcp_cost_tracking_per_tool():
    """Test that individual tool costs are tracked correctly when tool_name_to_cost_per_query is configured"""
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")],
        isError=False
    )
    
    # Create a mock MCPClient
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(return_value=[
        MCPTool(
            name="expensive_tool",
            description="Expensive tool",
            inputSchema={"type": "object", "properties": {"data": {"type": "string"}}}
        ),
        MCPTool(
            name="cheap_tool",
            description="Cheap tool",
            inputSchema={"type": "object", "properties": {"data": {"type": "string"}}}
        )
    ])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.disconnect = AsyncMock(return_value=None)
    
    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient', mock_client_constructor):
        # Load the server config with per-tool costs
        local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "test_server": {
                    "url": os.getenv("ZAPIER_MCP_HTTPS_SERVER_URL"),
                    "mcp_info": {
                        "mcp_server_cost_info": {
                            "default_cost_per_query": 0.5,  # Default cost
                            "tool_name_to_cost_per_query": {
                                "expensive_tool": 5.0,  # High cost tool
                                "cheap_tool": 0.1       # Low cost tool
                            }
                        }
                    }
                }
            }
        )

        # Set up the test logger
        test_logger = TestMCPLogger()
        litellm.callbacks = [test_logger]

        # Initialize the tool mapping
        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()
        
        # Manually add the tool mapping to ensure it's available (since mocking might not capture it properly)
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["expensive_tool"] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["test_server-expensive_tool"] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["cheap_tool"] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["test_server-cheap_tool"] = "test_server"
        
        # Patch the global manager in both modules where it's used
        with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager', local_mcp_server_manager), \
             patch('litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager', local_mcp_server_manager):

            print("tool_name_to_mcp_server_name_mapping", local_mcp_server_manager.tool_name_to_mcp_server_name_mapping)

            # Test 1: Call expensive_tool - should cost 5.0
            response1 = await mcp_server_tool_call(
                name="test_server-expensive_tool",  # Use correct prefixed name with - separator
                arguments={
                    "data": "test_expensive"
                }
            )

            # wait for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload_1 = test_logger.standard_logging_payload
            print("logged_standard_logging_payload_1", json.dumps(logged_standard_logging_payload_1, indent=4))
            
            # Verify expensive tool cost
            assert logged_standard_logging_payload_1 is not None, "Standard logging payload 1 should not be None"
            assert logged_standard_logging_payload_1["response_cost"] == 5.0
            
            # Reset logger for second test
            test_logger.standard_logging_payload = None

            # Test 2: Call cheap_tool - should cost 0.1
            response2 = await mcp_server_tool_call(
                name="test_server-cheap_tool",  # Use correct prefixed name with - separator
                arguments={
                    "data": "test_cheap"
                }
            )

            # wait for logging to be processed
            await asyncio.sleep(2)

            logged_standard_logging_payload_2 = test_logger.standard_logging_payload
            print("logged_standard_logging_payload_2", json.dumps(logged_standard_logging_payload_2, indent=4))
            
            # Verify cheap tool cost
            assert logged_standard_logging_payload_2 is not None, "Standard logging payload 2 should not be None"
            assert logged_standard_logging_payload_2["response_cost"] == 0.1
            
            # Add basic response assertions
            assert response1 is not None
            assert response2 is not None
            
            response_list_1 = list(response1)
            response_list_2 = list(response2)
            
            assert len(response_list_1) == 1
            assert len(response_list_2) == 1
            assert isinstance(response_list_1[0], TextContent)
            assert isinstance(response_list_2[0], TextContent)
            assert response_list_1[0].text == "Test response"
            assert response_list_2[0].text == "Test response"
            
            # Verify client methods were called twice
            assert mock_client.call_tool.call_count == 2




class MCPLoggerHook(CustomLogger):
    def __init__(self):
        self.standard_logging_payload = None
        super().__init__()
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("success event")
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)
        print(f"Captured standard_logging_payload: {self.standard_logging_payload}")
    
    async def async_post_mcp_tool_call_hook(self, kwargs, response_obj: MCPPostCallResponseObject, start_time, end_time) -> Optional[MCPPostCallResponseObject]:
        print("post mcp tool call response_obj", response_obj)
        # update the MCPPostCallResponseObject with the response_cost
        response_obj.hidden_params.response_cost = 1.42
        return response_obj


@pytest.mark.asyncio
async def test_mcp_tool_call_hook():
    # Create a mock tool call result
    litellm.logging_callback_manager._reset_all_callbacks()
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
    mock_client.disconnect = AsyncMock(return_value=None)
    
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
                }
            }
        )

        # Set up the test logger
        test_logger = MCPLoggerHook()
        litellm.callbacks = [test_logger]

        # Initialize the tool mapping
        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()
        
        # Manually add the tool mapping to ensure it's available (since mocking might not capture it properly)
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["add_tools"] = "zapier_gmail_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping["zapier_gmail_server-add_tools"] = "zapier_gmail_server"
        
        # Patch the global manager in both modules where it's used
        with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager', local_mcp_server_manager), \
             patch('litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager', local_mcp_server_manager):

            print("tool_name_to_mcp_server_name_mapping", local_mcp_server_manager.tool_name_to_mcp_server_name_mapping)

            # Call mcp tool using the correct separator format (- not /)
            response = await mcp_server_tool_call(
                name="zapier_gmail_server-add_tools",  # Use correct prefixed name with - separator
                arguments={
                    "test": "test"
                }
            )

            # wait 1-2 seconds for logging to be processed
            await asyncio.sleep(2)


            # check logged standard logging payload
            logged_standard_logging_payload = test_logger.standard_logging_payload
            print("logged_standard_logging_payload", json.dumps(logged_standard_logging_payload, indent=4))
            assert logged_standard_logging_payload is not None, "Standard logging payload should not be None"
            assert logged_standard_logging_payload["response_cost"] == 1.42