import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
)
from mcp.types import Tool as MCPTool

from litellm.experimental_mcp_client.tools import (
    _get_function_arguments,
    call_mcp_tool,
    call_openai_tool,
    load_mcp_tools,
    transform_mcp_tool_to_openai_tool,
    transform_openai_tool_call_request_to_mcp_tool_call_request,
)


@pytest.fixture
def mock_mcp_tool():
    return MCPTool(
        name="test_tool",
        description="A test tool",
        inputSchema={"type": "object", "properties": {"test": {"type": "string"}}},
    )


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.list_tools = AsyncMock()
    session.call_tool = AsyncMock()
    return session


@pytest.fixture
def mock_list_tools_result():
    return ListToolsResult(
        tools=[
            MCPTool(
                name="test_tool",
                description="A test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                },
            )
        ]
    )


@pytest.fixture
def mock_mcp_tool_call_result():
    return CallToolResult(content=[TextContent(type="text", text="test_output")])


def test_transform_mcp_tool_to_openai_tool(mock_mcp_tool):
    openai_tool = transform_mcp_tool_to_openai_tool(mock_mcp_tool)
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "test_tool"
    assert openai_tool["function"]["description"] == "A test tool"
    assert openai_tool["function"]["parameters"] == {
        "type": "object",
        "properties": {"test": {"type": "string"}},
    }


def testtransform_openai_tool_call_request_to_mcp_tool_call_request(mock_mcp_tool):
    openai_tool = {
        "function": {"name": "test_tool", "arguments": json.dumps({"test": "value"})}
    }
    mcp_tool_call_request = transform_openai_tool_call_request_to_mcp_tool_call_request(
        openai_tool
    )
    assert mcp_tool_call_request.name == "test_tool"
    assert mcp_tool_call_request.arguments == {"test": "value"}


@pytest.mark.asyncio()
async def test_load_mcp_tools_mcp_format(mock_session, mock_list_tools_result):
    mock_session.list_tools.return_value = mock_list_tools_result
    result = await load_mcp_tools(mock_session, format="mcp")
    assert len(result) == 1
    assert isinstance(result[0], MCPTool)
    assert result[0].name == "test_tool"
    mock_session.list_tools.assert_called_once()


@pytest.mark.asyncio()
async def test_load_mcp_tools_openai_format(mock_session, mock_list_tools_result):
    mock_session.list_tools.return_value = mock_list_tools_result
    result = await load_mcp_tools(mock_session, format="openai")
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "test_tool"
    mock_session.list_tools.assert_called_once()


def test_get_function_arguments():
    # Test with string arguments
    function = {"arguments": '{"test": "value"}'}
    result = _get_function_arguments(function)
    assert result == {"test": "value"}

    # Test with dict arguments
    function = {"arguments": {"test": "value"}}
    result = _get_function_arguments(function)
    assert result == {"test": "value"}

    # Test with invalid JSON string
    function = {"arguments": "invalid json"}
    result = _get_function_arguments(function)
    assert result == {}

    # Test with no arguments
    function = {}
    result = _get_function_arguments(function)
    assert result == {}


@pytest.mark.asyncio()
async def test_call_openai_tool(mock_session, mock_mcp_tool_call_result):
    mock_session.call_tool.return_value = mock_mcp_tool_call_result
    openai_tool = {
        "function": {"name": "test_tool", "arguments": json.dumps({"test": "value"})}
    }
    result = await call_openai_tool(mock_session, openai_tool)
    print("result of call_openai_tool", result)
    assert result.content[0].text == "test_output"
    mock_session.call_tool.assert_called_once_with(
        name="test_tool", arguments={"test": "value"}
    )


@pytest.mark.asyncio()
async def test_call_mcp_tool(mock_session, mock_mcp_tool_call_result):
    mock_session.call_tool.return_value = mock_mcp_tool_call_result
    request_params = CallToolRequestParams(
        name="test_tool", arguments={"test": "value"}
    )
    result = await call_mcp_tool(mock_session, request_params)
    print("call_mcp_tool result", result)
    assert result.content[0].text == "test_output"
    mock_session.call_tool.assert_called_once_with(
        name="test_tool", arguments={"test": "value"}
    )
