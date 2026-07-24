"""Tests for MCP header alias sanitization and auth header lookup."""

from litellm.proxy._experimental.mcp_server.utils import (
    lookup_mcp_server_auth_in_headers,
    sanitize_mcp_alias_for_header,
)


def test_sanitize_mcp_alias_for_header():
    assert sanitize_mcp_alias_for_header("My Server") == "my_server"
    assert sanitize_mcp_alias_for_header("GitHub-MCP!") == "github_mcp"
    assert sanitize_mcp_alias_for_header("github_mcp2") == "github_mcp2"


def test_lookup_mcp_server_auth_in_headers_sanitized_alias():
    headers = {"github_mcp": {"Authorization": "Bearer token"}}
    result = lookup_mcp_server_auth_in_headers(headers, alias="GitHub-MCP")
    assert result == {"Authorization": "Bearer token"}
