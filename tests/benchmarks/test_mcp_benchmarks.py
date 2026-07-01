"""
Performance benchmarks for the MCP tool hot path.

Two layers are covered: the client-side translation between MCP and OpenAI
function-calling formats, and the server-side tool-name prefixing that the proxy
runs on every list-tools (prefix each tool) and call-tool (strip prefix to route)
request. Both are pure-CPU and deterministic.
"""

import pytest
from mcp.types import Tool as MCPTool

from litellm.experimental_mcp_client.tools import (
    transform_mcp_tool_to_openai_tool,
    transform_openai_tool_call_request_to_mcp_tool_call_request,
)
from litellm.proxy._experimental.mcp_server.utils import (
    add_server_prefix_to_name,
    split_server_prefix_from_name,
)


def _make_tool(index: int) -> MCPTool:
    return MCPTool(
        name=f"tool_{index}",
        description=f"Test tool number {index} that performs an operation",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        },
    )


SINGLE_TOOL = _make_tool(0)
TOOL_LIST = tuple(_make_tool(i) for i in range(20))
TOOL_NAMES = tuple(t.name for t in TOOL_LIST)

SERVER_NAME = "github_mcp"
PREFIXED_TOOL_NAME = add_server_prefix_to_name("tool_0", SERVER_NAME)

OPENAI_TOOL_CALL = {
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "tool_0",
        "arguments": '{"query": "weather in San Francisco", "limit": 5}',
    },
}


@pytest.mark.benchmark
def test_transform_single_mcp_tool_to_openai():
    """Benchmark translating one MCP tool into OpenAI tool format."""
    transform_mcp_tool_to_openai_tool(mcp_tool=SINGLE_TOOL)


@pytest.mark.benchmark
def test_transform_mcp_tool_list_to_openai():
    """Benchmark translating a full list-tools response into OpenAI format."""
    for tool in TOOL_LIST:
        transform_mcp_tool_to_openai_tool(mcp_tool=tool)


@pytest.mark.benchmark
def test_transform_openai_tool_call_to_mcp():
    """Benchmark translating an OpenAI tool call into an MCP call request."""
    transform_openai_tool_call_request_to_mcp_tool_call_request(openai_tool=OPENAI_TOOL_CALL)


@pytest.mark.benchmark
def test_mcp_server_prefix_tool_list():
    """Benchmark the proxy prefixing every tool name on a list-tools response."""
    for name in TOOL_NAMES:
        add_server_prefix_to_name(name, SERVER_NAME)


@pytest.mark.benchmark
def test_mcp_server_strip_prefix_on_call():
    """Benchmark the proxy stripping the server prefix to route a tool call."""
    split_server_prefix_from_name(PREFIXED_TOOL_NAME)
