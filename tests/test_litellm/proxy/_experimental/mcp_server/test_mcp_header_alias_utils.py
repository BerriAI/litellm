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


def test_lookup_mcp_server_auth_in_headers_group_header_is_default_for_members():
    headers = {"shared": {"Authorization": "Bearer group-token"}}
    assert lookup_mcp_server_auth_in_headers(headers, alias="alpha", server_name="alpha", access_groups=["shared"]) == {
        "Authorization": "Bearer group-token"
    }
    assert lookup_mcp_server_auth_in_headers(headers, alias="beta", server_name="beta", access_groups=["Shared"]) == {
        "Authorization": "Bearer group-token"
    }


def test_lookup_mcp_server_auth_in_headers_group_header_sanitized_group_name():
    headers = {"dev_group": {"Authorization": "Bearer group-token"}}
    assert lookup_mcp_server_auth_in_headers(headers, alias="alpha", access_groups=["Dev Group"]) == {
        "Authorization": "Bearer group-token"
    }


def test_lookup_mcp_server_auth_in_headers_server_header_overrides_group_header():
    headers = {
        "shared": {"Authorization": "Bearer group-token"},
        "beta": {"Authorization": "Bearer beta-token"},
    }
    assert lookup_mcp_server_auth_in_headers(headers, alias="beta", server_name="beta", access_groups=["shared"]) == {
        "Authorization": "Bearer beta-token"
    }


def test_lookup_mcp_server_auth_in_headers_group_header_not_forwarded_outside_group():
    headers = {"shared": {"Authorization": "Bearer group-token"}}
    assert (
        lookup_mcp_server_auth_in_headers(headers, alias="gamma", server_name="gamma", access_groups=["other"]) is None
    )
    assert lookup_mcp_server_auth_in_headers(headers, alias="gamma", server_name="gamma", access_groups=None) is None


def test_lookup_mcp_server_auth_in_headers_alias_colliding_with_group_name_keeps_server_level_match():
    headers = {"shared": {"Authorization": "Bearer shared-token"}}
    assert lookup_mcp_server_auth_in_headers(headers, alias="shared", access_groups=["other"]) == {
        "Authorization": "Bearer shared-token"
    }
    assert lookup_mcp_server_auth_in_headers(headers, alias="alpha", access_groups=["shared"]) == {
        "Authorization": "Bearer shared-token"
    }
    assert lookup_mcp_server_auth_in_headers(headers, alias="gamma", access_groups=["other"]) is None


def test_lookup_mcp_server_auth_in_headers_conflicting_group_headers_fail_closed():
    headers = {
        "shared": {"Authorization": "Bearer group-token"},
        "other": {"Authorization": "Bearer other-token"},
    }
    assert lookup_mcp_server_auth_in_headers(headers, alias="delta", access_groups=["shared", "other"]) is None


def test_lookup_mcp_server_auth_in_headers_identical_group_headers_resolve():
    headers = {
        "shared": {"Authorization": "Bearer group-token"},
        "other": {"Authorization": "Bearer group-token"},
    }
    assert lookup_mcp_server_auth_in_headers(headers, alias="delta", access_groups=["shared", "other"]) == {
        "Authorization": "Bearer group-token"
    }
