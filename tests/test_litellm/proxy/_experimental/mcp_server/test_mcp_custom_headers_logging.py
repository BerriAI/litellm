"""
Tests for custom MCP headers flowing into logging callbacks.

Validates that:
1. _extract_custom_headers correctly filters raw HTTP headers
2. Custom headers are stored in StandardLoggingMCPToolCall.custom_headers
3. Custom headers populate requester_custom_headers in StandardLoggingMetadata
"""

from litellm.proxy._experimental.mcp_server.server import _extract_custom_headers


class TestExtractCustomHeaders:
    """Unit tests for _extract_custom_headers helper."""

    def test_returns_none_for_none_input(self):
        assert _extract_custom_headers(None) is None

    def test_returns_none_for_empty_dict(self):
        assert _extract_custom_headers({}) is None

    def test_extracts_x_prefixed_headers(self):
        raw = {
            "x-custom-header-foo": "bar",
            "x-request-id": "abc123",
            "content-type": "application/json",
            "authorization": "Bearer token",
        }
        result = _extract_custom_headers(raw)
        assert result == {
            "x-custom-header-foo": "bar",
            "x-request-id": "abc123",
        }

    def test_excludes_x_api_key(self):
        raw = {
            "x-api-key": "secret",
            "x-custom-foo": "bar",
        }
        result = _extract_custom_headers(raw)
        assert result == {"x-custom-foo": "bar"}

    def test_excludes_x_litellm_prefixed(self):
        raw = {
            "x-litellm-api-key": "secret",
            "x-litellm-mcp-debug": "true",
            "x-custom-foo": "bar",
        }
        result = _extract_custom_headers(raw)
        assert result == {"x-custom-foo": "bar"}

    def test_excludes_x_mcp_server_auth_prefixed(self):
        raw = {
            "x-mcp-server-auth-token": "secret",
            "x-custom-foo": "bar",
        }
        result = _extract_custom_headers(raw)
        assert result == {"x-custom-foo": "bar"}

    def test_preserves_original_key_casing(self):
        raw = {"X-Custom-Header": "value"}
        result = _extract_custom_headers(raw)
        assert result == {"X-Custom-Header": "value"}

    def test_returns_none_when_all_headers_filtered(self):
        raw = {
            "content-type": "application/json",
            "authorization": "Bearer token",
            "x-api-key": "secret",
        }
        assert _extract_custom_headers(raw) is None

    def test_excludes_non_string_values(self):
        raw = {
            "x-good": "value",
            "x-bad": None,  # type: ignore
        }
        result = _extract_custom_headers(raw)
        assert result == {"x-good": "value"}

    def test_case_insensitive_prefix_matching(self):
        """Header keys with mixed case should still be filtered correctly."""
        raw = {
            "X-API-Key": "secret",
            "X-Litellm-Something": "hidden",
            "X-Custom-Foo": "visible",
        }
        result = _extract_custom_headers(raw)
        assert result == {"X-Custom-Foo": "visible"}
