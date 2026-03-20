import os
import sys
import pytest
import asyncio
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
    set_auth_context,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
)
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
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
    

def _set_authorized_user(server_ids):
    """Configure auth context with permission to call the specified servers."""
    server_list = list(server_ids)
    user_auth = UserAPIKeyAuth(
        api_key="test",
        user_id="test_user",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="mcp-test-permissions",
            mcp_servers=server_list,
        ),
    )
    set_auth_context(user_api_key_auth=user_auth, mcp_servers=server_list)


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
    
    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient', mock_client_constructor):
        # Load the server config
        await local_mcp_server_manager.load_servers_from_config(
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

            _set_authorized_user(local_mcp_server_manager.get_all_mcp_server_ids())

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
            print("logged_standard_logging_payload", logged_standard_logging_payload)
            
            # Add assertions
            assert response is not None
            # Handle CallToolResult - access .content for the list of content items
            if isinstance(response, CallToolResult):
                response_list = response.content
            else:
                response_list = list(response)  # Convert iterable to list for backward compatibility
            assert len(response_list) == 1
            assert isinstance(response_list[0], TextContent)
            assert response_list[0].text == "Test response"
            
            # Verify client methods were called
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
    
    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient', mock_client_constructor):
        # Load the server config with per-tool costs
        await local_mcp_server_manager.load_servers_from_config(
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

            _set_authorized_user(local_mcp_server_manager.get_all_mcp_server_ids())

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
            print("logged_standard_logging_payload_1", logged_standard_logging_payload_1)
            
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
            print("logged_standard_logging_payload_2", logged_standard_logging_payload_2)
            
            # Verify cheap tool cost
            assert logged_standard_logging_payload_2 is not None, "Standard logging payload 2 should not be None"
            assert logged_standard_logging_payload_2["response_cost"] == 0.1
            
            # Add basic response assertions
            assert response1 is not None
            assert response2 is not None
            
            response_list_1 = list(response1.content)
            response_list_2 = list(response2.content)
            
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
    
    # Mock the MCPClient constructor
    def mock_client_constructor(*args, **kwargs):
        return mock_client

    # Initialize the server manager
    local_mcp_server_manager = MCPServerManager()
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient', mock_client_constructor):
        # Load the server config
        await local_mcp_server_manager.load_servers_from_config(
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

            _set_authorized_user(local_mcp_server_manager.get_all_mcp_server_ids())

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
            print("logged_standard_logging_payload", logged_standard_logging_payload)
            assert logged_standard_logging_payload is not None, "Standard logging payload should not be None"
            assert logged_standard_logging_payload["response_cost"] == 1.42


@pytest.mark.asyncio
async def test_mcp_custom_headers_in_logging_callback():
    """Test that custom x-* headers from MCP requests appear in requester_custom_headers."""
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")],
        isError=False,
    )

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(
        return_value=[
            MCPTool(
                name="add_tools",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                },
            )
        ]
    )

    def mock_client_constructor(*args, **kwargs):
        return mock_client

    local_mcp_server_manager = MCPServerManager()

    with patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient",
        mock_client_constructor,
    ):
        await local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "test_server": {
                    "url": "https://example.com/mcp",
                }
            }
        )

        test_logger = TestMCPLogger()
        litellm.callbacks = [test_logger]

        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "add_tools"
        ] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "test_server-add_tools"
        ] = "test_server"

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            local_mcp_server_manager,
        ), patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
            local_mcp_server_manager,
        ):
            # Set auth context WITH raw_headers containing custom headers
            raw_headers = {
                "x-custom-header-foo": "bar",
                "x-trace-id": "abc123",
                "content-type": "application/json",
                "user-agent": "test-agent/1.0",
            }
            set_auth_context(
                user_api_key_auth=UserAPIKeyAuth(
                    api_key="test",
                    user_id="test_user",
                    object_permission=LiteLLM_ObjectPermissionTable(
                        object_permission_id="mcp-test-permissions",
                        mcp_servers=list(
                            local_mcp_server_manager.get_all_mcp_server_ids()
                        ),
                    ),
                ),
                mcp_servers=list(
                    local_mcp_server_manager.get_all_mcp_server_ids()
                ),
                raw_headers=raw_headers,
            )

            await mcp_server_tool_call(
                name="test_server-add_tools",
                arguments={"test": "test"},
            )

            await asyncio.sleep(2)

            payload = test_logger.standard_logging_payload
            assert payload is not None, "Standard logging payload should not be None"

            metadata = payload["metadata"]
            custom_headers = metadata.get("requester_custom_headers")
            assert custom_headers is not None, (
                "requester_custom_headers should be populated"
            )
            assert custom_headers["x-custom-header-foo"] == "bar"
            assert custom_headers["x-trace-id"] == "abc123"
            # Non x-* headers should NOT be in requester_custom_headers
            assert "content-type" not in custom_headers
            assert "user-agent" not in custom_headers


@pytest.mark.asyncio
async def test_mcp_sensitive_headers_not_in_logging_callback():
    """Test that sensitive headers are excluded from requester_custom_headers.

    Disables the enterprise logging override so we test the open-source
    filtering in isolation (the enterprise code has its own filter).
    """
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")],
        isError=False,
    )

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(
        return_value=[
            MCPTool(
                name="add_tools",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                },
            )
        ]
    )

    def mock_client_constructor(*args, **kwargs):
        return mock_client

    local_mcp_server_manager = MCPServerManager()

    with patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient",
        mock_client_constructor,
    ):
        await local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "test_server": {
                    "url": "https://example.com/mcp",
                }
            }
        )

        test_logger = TestMCPLogger()
        litellm.callbacks = [test_logger]

        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "add_tools"
        ] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "test_server-add_tools"
        ] = "test_server"

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            local_mcp_server_manager,
        ), patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
            local_mcp_server_manager,
        ), patch(
            "litellm.litellm_core_utils.litellm_logging.EnterpriseStandardLoggingPayloadSetupVAR",
            None,
        ):
            # Include sensitive headers that should be stripped
            raw_headers = {
                "authorization": "Bearer secret-token",
                "x-litellm-api-key": "sk-secret-key",
                "x-mcp-github-authorization": "Bearer gh-token",
                "x-mcp-zapier-x-api-key": "zapier-secret",
                "x-custom-safe-header": "safe-value",
            }
            set_auth_context(
                user_api_key_auth=UserAPIKeyAuth(
                    api_key="test",
                    user_id="test_user",
                    object_permission=LiteLLM_ObjectPermissionTable(
                        object_permission_id="mcp-test-permissions",
                        mcp_servers=list(
                            local_mcp_server_manager.get_all_mcp_server_ids()
                        ),
                    ),
                ),
                mcp_servers=list(
                    local_mcp_server_manager.get_all_mcp_server_ids()
                ),
                raw_headers=raw_headers,
            )

            await mcp_server_tool_call(
                name="test_server-add_tools",
                arguments={"test": "test"},
            )

            await asyncio.sleep(2)

            payload = test_logger.standard_logging_payload
            assert payload is not None, "Standard logging payload should not be None"

            metadata = payload["metadata"]
            custom_headers = metadata.get("requester_custom_headers")

            # The safe custom header should be present
            assert custom_headers is not None
            assert custom_headers.get("x-custom-safe-header") == "safe-value"

            # Sensitive headers must NOT appear
            assert "authorization" not in custom_headers
            # x-litellm-api-key is stripped by clean_headers (SpecialHeaders)
            assert "x-litellm-api-key" not in custom_headers
            # Server-specific MCP auth headers are filtered by regex
            assert "x-mcp-github-authorization" not in custom_headers
            assert "x-mcp-zapier-x-api-key" not in custom_headers


@pytest.mark.asyncio
async def test_mcp_logging_without_raw_headers():
    """Test no regression when raw_headers is None (existing behavior)."""
    litellm.logging_callback_manager._reset_all_callbacks()
    mock_result = CallToolResult(
        content=[TextContent(type="text", text="Test response")],
        isError=False,
    )

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=mock_result)
    mock_client.list_tools = AsyncMock(
        return_value=[
            MCPTool(
                name="add_tools",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                },
            )
        ]
    )

    def mock_client_constructor(*args, **kwargs):
        return mock_client

    local_mcp_server_manager = MCPServerManager()

    with patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient",
        mock_client_constructor,
    ):
        await local_mcp_server_manager.load_servers_from_config(
            mcp_servers_config={
                "test_server": {
                    "url": "https://example.com/mcp",
                }
            }
        )

        test_logger = TestMCPLogger()
        litellm.callbacks = [test_logger]

        await local_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "add_tools"
        ] = "test_server"
        local_mcp_server_manager.tool_name_to_mcp_server_name_mapping[
            "test_server-add_tools"
        ] = "test_server"

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            local_mcp_server_manager,
        ), patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
            local_mcp_server_manager,
        ):
            # No raw_headers — existing behavior
            _set_authorized_user(
                local_mcp_server_manager.get_all_mcp_server_ids()
            )

            await mcp_server_tool_call(
                name="test_server-add_tools",
                arguments={"test": "test"},
            )

            await asyncio.sleep(2)

            payload = test_logger.standard_logging_payload
            assert payload is not None, "Standard logging payload should not be None"
            # Should not crash; requester_custom_headers should be None or empty
            metadata = payload["metadata"]
            custom_headers = metadata.get("requester_custom_headers")
            assert custom_headers is None or custom_headers == {}
