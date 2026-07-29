"""Unit coverage for the multi-worker MCP lag classifier used by await_call_tool."""

from __future__ import annotations

from e2e_http import Success, UnknownApiError
from mcp_client import McpCallToolResponse, _is_mcp_not_synced


def test_classifies_gateway_tool_not_found_for_named_tool() -> None:
    result = UnknownApiError(
        status_code=500,
        body='{"detail":{"error":"internal_server_error","message":"An unexpected error occurred: Tool search_datadog_logs not found"}}',
    )
    assert _is_mcp_not_synced(result, tool_name="search_datadog_logs") is True


def test_rejects_unrelated_tool_name_in_error() -> None:
    result = UnknownApiError(
        status_code=500,
        body='{"detail":{"message":"Tool other_tool not found"}}',
    )
    assert _is_mcp_not_synced(result, tool_name="search_datadog_logs") is False


def test_rejects_generic_not_found_mentioning_tool() -> None:
    result = UnknownApiError(
        status_code=500,
        body='{"detail":{"message":"upstream said the tool documentation was not found on the server"}}',
    )
    assert _is_mcp_not_synced(result, tool_name="search_datadog_logs") is False


def test_classifies_server_not_found() -> None:
    result = UnknownApiError(
        status_code=500,
        body='{"detail":{"error":"server_not_found","message":"MCP server \'abc\' was not found"}}',
    )
    assert _is_mcp_not_synced(result, tool_name="search_datadog_logs") is True


def test_success_is_not_lag() -> None:
    result = Success(status_code=200, data=McpCallToolResponse(content=[], isError=False))
    assert _is_mcp_not_synced(result, tool_name="search_datadog_logs") is False
