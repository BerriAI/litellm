"""LIT-4216: helpers gating MCP tool names against the 64-char provider limit."""

from mcp.types import Tool as MCPTool

from litellm.proxy._experimental.mcp_server.utils import (
    split_tools_by_name_length,
    tool_name_length_warnings,
)


def _tool(name: str) -> MCPTool:
    return MCPTool(name=name, inputSchema={})


def test_split_keeps_names_at_the_limit_and_drops_longer():
    kept, dropped = split_tools_by_name_length([_tool("a" * 64), _tool("b" * 65)], 64)

    assert [tool.name for tool in kept] == ["a" * 64]
    assert [tool.name for tool in dropped] == ["b" * 65]


def test_split_zero_or_negative_limit_disables_the_check():
    tools = [_tool("a" * 300)]

    for limit in (0, -1):
        kept, dropped = split_tools_by_name_length(tools, limit)
        assert [tool.name for tool in kept] == ["a" * 300]
        assert dropped == []


def test_warnings_flag_only_tools_whose_prefixed_name_exceeds_the_limit():
    warnings = tool_name_length_warnings(["short", "t" * 60], "server_alias", 64)

    assert len(warnings) == 1
    assert f"server_alias-{'t' * 60}" in warnings[0]
    assert "(73 characters)" in warnings[0]


def test_warnings_disabled_when_limit_is_zero():
    assert tool_name_length_warnings(["t" * 300], "server_alias", 0) == []
