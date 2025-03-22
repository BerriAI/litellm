import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult
from mcp.types import Tool as MCPTool

from litellm.experimental_mcp_client.tools import (
    _transform_openai_tool_call_to_mcp_tool_call_request,
    call_mcp_tool,
    call_openai_tool,
    load_mcp_tools,
    transform_mcp_tool_to_openai_tool,
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


def test_transform_mcp_tool_to_openai_tool(mock_mcp_tool):
    openai_tool = transform_mcp_tool_to_openai_tool(mock_mcp_tool)
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "test_tool"
    assert openai_tool["function"]["description"] == "A test tool"
    assert openai_tool["function"]["parameters"] == {
        "type": "object",
        "properties": {"test": {"type": "string"}},
    }


def test_transform_openai_tool_call_to_mcp_tool_call_request(mock_mcp_tool):
    openai_tool = {
        "function": {"name": "test_tool", "arguments": json.dumps({"test": "value"})}
    }
    mcp_tool_call_request = _transform_openai_tool_call_to_mcp_tool_call_request(
        openai_tool
    )
    assert mcp_tool_call_request.name == "test_tool"
    assert mcp_tool_call_request.arguments == {"test": "value"}
