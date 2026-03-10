"""Tests for litellm.proxy._experimental.mcp_server.utils"""

from litellm.proxy._experimental.mcp_server.utils import resolve_extra_headers


class TestResolveExtraHeaders:
    """Test resolve_extra_headers helper."""

    def test_matches_headers_case_insensitively(self):
        result = resolve_extra_headers(
            ["X-Custom-Auth", "X-Tenant-Id"],
            {"x-custom-auth": "secret", "x-tenant-id": "t1", "x-other": "ignored"},
        )
        assert result == {"X-Custom-Auth": "secret", "X-Tenant-Id": "t1"}

    def test_preserves_config_casing(self):
        result = resolve_extra_headers(
            ["X-My-Header"],
            {"x-my-header": "val"},
        )
        assert "X-My-Header" in result

    def test_returns_none_when_no_match(self):
        result = resolve_extra_headers(
            ["X-Missing"],
            {"x-other": "val"},
        )
        assert result is None

    def test_returns_none_when_extra_header_names_empty(self):
        assert resolve_extra_headers([], {"x-foo": "bar"}) is None

    def test_returns_none_when_raw_headers_none(self):
        assert resolve_extra_headers(["X-Foo"], None) is None

    def test_returns_none_when_both_none(self):
        assert resolve_extra_headers(None, None) is None

    def test_skips_non_string_header_names(self):
        result = resolve_extra_headers(
            ["X-Valid", 123, None],
            {"x-valid": "ok"},
        )
        assert result == {"X-Valid": "ok"}
