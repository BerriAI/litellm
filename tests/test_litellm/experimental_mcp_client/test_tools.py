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
    _normalize_mcp_input_schema,
    call_mcp_tool,
    call_openai_tool,
    load_mcp_tools,
    transform_mcp_tool_to_openai_responses_api_tool,
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
        "additionalProperties": False,
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


def test_normalize_mcp_input_schema():
    """Test MCP input schema normalization for OpenAI compatibility."""
    # Test case 1: Empty/None schema should get default structure
    assert _normalize_mcp_input_schema(None) == {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }
    
    assert _normalize_mcp_input_schema({}) == {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }
    
    # Test case 2: Schema with only type should get properties added
    schema_with_type_only = {"type": "object"}
    normalized = _normalize_mcp_input_schema(schema_with_type_only)
    assert normalized == {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }
    
    # Test case 3: Schema missing type should get type added
    schema_missing_type = {"properties": {"param": {"type": "string"}}}
    normalized = _normalize_mcp_input_schema(schema_missing_type)
    assert normalized == {
        "type": "object",
        "properties": {"param": {"type": "string"}},
        "additionalProperties": False
    }
    
    # Test case 4: Complete schema should be preserved with additionalProperties added
    complete_schema = {
        "type": "object",
        "properties": {"param": {"type": "string"}},
        "required": ["param"]
    }
    normalized = _normalize_mcp_input_schema(complete_schema)
    assert normalized == {
        "type": "object",
        "properties": {"param": {"type": "string"}},
        "required": ["param"],
        "additionalProperties": False
    }
    
    # Test case 5: Schema with existing additionalProperties should be preserved
    schema_with_additional = {
        "type": "object",
        "properties": {"param": {"type": "string"}},
        "additionalProperties": True
    }
    normalized = _normalize_mcp_input_schema(schema_with_additional)
    assert normalized["additionalProperties"] == True


def test_transform_mcp_tool_to_openai_responses_api_tool():
    """Test transformation to OpenAI Responses API tool format with schema normalization."""
    # Test case 1: Tool with minimal schema (the problematic case from the error)
    minimal_tool = MCPTool(
        name="GitMCP-fetch_litellm_documentation",
        description="Fetch entire documentation file from GitHub repository",
        inputSchema={"type": "object"}  # This was causing the error
    )
    
    openai_tool = transform_mcp_tool_to_openai_responses_api_tool(minimal_tool)
    assert openai_tool["name"] == "GitMCP-fetch_litellm_documentation"
    assert openai_tool["type"] == "function"
    assert openai_tool["strict"] == False
    assert openai_tool["parameters"]["type"] == "object"
    assert openai_tool["parameters"]["properties"] == {}
    assert openai_tool["parameters"]["additionalProperties"] == False
    
    # Test case 2: Tool with complete schema
    complete_tool = MCPTool(
        name="test_tool_complete",
        description="A test tool with complete schema",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    )
    
    openai_tool = transform_mcp_tool_to_openai_responses_api_tool(complete_tool)
    assert openai_tool["parameters"]["type"] == "object"
    assert "query" in openai_tool["parameters"]["properties"]
    assert openai_tool["parameters"]["required"] == ["query"]
    assert openai_tool["parameters"]["additionalProperties"] == False
