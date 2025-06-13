# Create server parameters for stdio connection
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    MCPServer,
    MCPTransport,
)
from mcp.types import Tool as MCPTool, CallToolResult, ListToolsResult
from mcp.types import TextContent


mcp_server_manager = MCPServerManager()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Local only test")
async def test_mcp_server_manager():
    mcp_server_manager.load_servers_from_config(
        {
            "zapier_mcp_server": {
                "url": os.environ.get("ZAPIER_MCP_SERVER_URL"),
            }
        }
    )
    tools = await mcp_server_manager.list_tools()
    print("TOOLS FROM MCP SERVER MANAGER== ", tools)

    result = await mcp_server_manager.call_tool(
        name="gmail_send_email", arguments={"body": "Test"}
    )
    print("RESULT FROM CALLING TOOL FROM MCP SERVER MANAGER== ", result)


@pytest.mark.asyncio
async def test_mcp_server_manager_https_server():
    mcp_server_manager.load_servers_from_config(
        {
            "zapier_mcp_server": {
                "url": os.environ.get("ZAPIER_MCP_HTTPS_SERVER_URL"),
                "transport": MCPTransport.http,
            }
        }
    )
    tools = await mcp_server_manager.list_tools()
    print("TOOLS FROM MCP SERVER MANAGER== ", tools)

    result = await mcp_server_manager.call_tool(
        name="gmail_send_email",
        arguments={
            "body": "Test",
            "message": "Test",
            "instructions": "Test",
        },
    )
    print("RESULT FROM CALLING TOOL FROM MCP SERVER MANAGER== ", result)


@pytest.mark.asyncio
async def test_mcp_http_transport_list_tools_mock():
    """Test HTTP transport list_tools functionality with mocked dependencies"""
    
    # Create a fresh manager for testing
    test_manager = MCPServerManager()
    
    # Mock tools that should be returned
    mock_tools = [
        MCPTool(
            name="gmail_send_email",
            description="Send an email via Gmail",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        ),
        MCPTool(
            name="calendar_create_event",
            description="Create a calendar event",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"}
                },
                "required": ["title", "date"]
            }
        )
    ]
    
    # Mock the session and its methods
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=ListToolsResult(tools=mock_tools))
    
    # Create an async context manager mock for streamablehttp_client
    @asynccontextmanager
    async def mock_streamablehttp_client(url):
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        get_session_id = MagicMock(return_value="test-session-123")
        yield (read_stream, write_stream, get_session_id)
    
    # Create an async context manager mock for ClientSession
    @asynccontextmanager
    async def mock_client_session(read_stream, write_stream):
        yield mock_session
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.streamablehttp_client', mock_streamablehttp_client), \
         patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.ClientSession', mock_client_session):
        
        # Load server config with HTTP transport
        test_manager.load_servers_from_config({
            "test_http_server": {
                "url": "https://test-mcp-server.com/mcp",
                "transport": MCPTransport.http,
                "description": "Test HTTP MCP Server"
            }
        })
        
        # Call list_tools
        tools = await test_manager.list_tools()
        
        # Assertions
        assert len(tools) == 2
        assert tools[0].name == "gmail_send_email"
        assert tools[1].name == "calendar_create_event"
        
        # Verify session methods were called
        mock_session.initialize.assert_called_once()
        mock_session.list_tools.assert_called_once()
        
        # Verify tool mapping was updated
        assert test_manager.tool_name_to_mcp_server_name_mapping["gmail_send_email"] == "test_http_server"
        assert test_manager.tool_name_to_mcp_server_name_mapping["calendar_create_event"] == "test_http_server"


@pytest.mark.asyncio
async def test_mcp_http_transport_call_tool_mock():
    """Test HTTP transport call_tool functionality with mocked dependencies"""
    
    # Create a fresh manager for testing
    test_manager = MCPServerManager()
    
    # Mock tool call result
    mock_result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text="Email sent successfully to test@example.com"
            )
        ],
        isError=False
    )
    
    # Mock the session and its methods
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    
    # Create an async context manager mock for streamablehttp_client
    @asynccontextmanager
    async def mock_streamablehttp_client(url):
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        get_session_id = MagicMock(return_value="test-session-456")
        yield (read_stream, write_stream, get_session_id)
    
    # Create an async context manager mock for ClientSession
    @asynccontextmanager
    async def mock_client_session(read_stream, write_stream):
        yield mock_session
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.streamablehttp_client', mock_streamablehttp_client), \
         patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.ClientSession', mock_client_session):
        
        # Load server config with HTTP transport
        test_manager.load_servers_from_config({
            "test_http_server": {
                "url": "https://test-mcp-server.com/mcp",
                "transport": MCPTransport.http,
                "description": "Test HTTP MCP Server"
            }
        })
        
        # Manually set up tool mapping (normally done by list_tools)
        test_manager.tool_name_to_mcp_server_name_mapping["gmail_send_email"] = "test_http_server"
        
        # Call the tool
        result = await test_manager.call_tool(
            name="gmail_send_email",
            arguments={
                "to": "test@example.com",
                "subject": "Test Subject",
                "body": "Test email body"
            }
        )
        
        # Assertions
        assert result.isError is False
        assert len(result.content) == 1
        # Type check before accessing text attribute
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Email sent successfully to test@example.com"
        
        # Verify session methods were called
        mock_session.initialize.assert_called_once()
        mock_session.call_tool.assert_called_once_with(
            "gmail_send_email",
            {
                "to": "test@example.com", 
                "subject": "Test Subject",
                "body": "Test email body"
            }
        )


@pytest.mark.asyncio
async def test_mcp_http_transport_call_tool_error_mock():
    """Test HTTP transport call_tool error handling with mocked dependencies"""
    
    # Create a fresh manager for testing
    test_manager = MCPServerManager()
    
    # Mock tool call error result
    mock_error_result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text="Error: Invalid email address"
            )
        ],
        isError=True
    )
    
    # Mock the session and its methods
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_error_result)
    
    # Create an async context manager mock for streamablehttp_client
    @asynccontextmanager
    async def mock_streamablehttp_client(url):
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        get_session_id = MagicMock(return_value="test-session-789")
        yield (read_stream, write_stream, get_session_id)
    
    # Create an async context manager mock for ClientSession
    @asynccontextmanager
    async def mock_client_session(read_stream, write_stream):
        yield mock_session
    
    with patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.streamablehttp_client', mock_streamablehttp_client), \
         patch('litellm.proxy._experimental.mcp_server.mcp_server_manager.ClientSession', mock_client_session):
        
        # Load server config with HTTP transport
        test_manager.load_servers_from_config({
            "test_http_server": {
                "url": "https://test-mcp-server.com/mcp", 
                "transport": MCPTransport.http,
                "description": "Test HTTP MCP Server"
            }
        })
        
        # Manually set up tool mapping
        test_manager.tool_name_to_mcp_server_name_mapping["gmail_send_email"] = "test_http_server"
        
        # Call the tool with invalid data
        result = await test_manager.call_tool(
            name="gmail_send_email",
            arguments={"to": "invalid-email", "subject": "Test", "body": "Test"}
        )
        
        # Assertions for error case
        assert result.isError is True
        assert len(result.content) == 1
        # Type check before accessing text attribute
        assert isinstance(result.content[0], TextContent)
        assert "Error: Invalid email address" in result.content[0].text
        
        # Verify session methods were called
        mock_session.initialize.assert_called_once()
        mock_session.call_tool.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_http_transport_tool_not_found():
    """Test calling a tool that doesn't exist"""
    
    # Create a fresh manager for testing
    test_manager = MCPServerManager()
    
    # Load server config
    test_manager.load_servers_from_config({
        "test_http_server": {
            "url": "https://test-mcp-server.com/mcp",
            "transport": MCPTransport.http,
            "description": "Test HTTP MCP Server"
        }
    })
    
    # Try to call a tool that doesn't exist in mapping
    with pytest.raises(ValueError, match="Tool nonexistent_tool not found"):
        await test_manager.call_tool(
            name="nonexistent_tool",
            arguments={"param": "value"}
        )


@pytest.mark.asyncio
async def test_streamable_http_mcp_handler_mock():
    """Test the streamable HTTP MCP handler functionality"""
    
    # Mock the session manager and its methods
    mock_session_manager = AsyncMock()
    mock_session_manager.handle_request = AsyncMock()
    
    # Mock scope, receive, send with proper ASGI scope format
    mock_scope = {
        "type": "http",
        "method": "POST", 
        "path": "/mcp",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "server": ("localhost", 8000),
        "scheme": "http"
    }
    mock_receive = AsyncMock()
    mock_send = AsyncMock()
    
    with patch('litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED', True), \
         patch('litellm.proxy._experimental.mcp_server.server.session_manager', mock_session_manager):
        
        from litellm.proxy._experimental.mcp_server.server import handle_streamable_http_mcp
        
        # Call the handler
        await handle_streamable_http_mcp(mock_scope, mock_receive, mock_send)
        
        # Verify session manager handle_request was called
        mock_session_manager.handle_request.assert_called_once_with(
            mock_scope, mock_receive, mock_send
        )


@pytest.mark.asyncio
async def test_sse_mcp_handler_mock():
    """Test the SSE MCP handler functionality"""
    
    # Mock the SSE session manager and its methods
    mock_sse_session_manager = AsyncMock()
    mock_sse_session_manager.handle_request = AsyncMock()
    
    # Mock scope, receive, send with proper ASGI scope format
    mock_scope = {
        "type": "http",
        "method": "GET",
        "path": "/mcp/sse",
        "headers": [(b"accept", b"text/event-stream")],
        "query_string": b"",
        "server": ("localhost", 8000),
        "scheme": "http"
    }
    mock_receive = AsyncMock()
    mock_send = AsyncMock()
    
    with patch('litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED', True), \
         patch('litellm.proxy._experimental.mcp_server.server.sse_session_manager', mock_sse_session_manager):
        
        from litellm.proxy._experimental.mcp_server.server import handle_sse_mcp
        
        # Call the handler
        await handle_sse_mcp(mock_scope, mock_receive, mock_send)
        
        # Verify SSE session manager handle_request was called
        mock_sse_session_manager.handle_request.assert_called_once_with(
            mock_scope, mock_receive, mock_send
        )


