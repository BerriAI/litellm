"""
Tests for custom MCP headers flowing into logging callbacks.

Covers:
- ``_extract_custom_headers`` deny-list and inclusion rules for ``x-*`` headers.
- ``_apply_requester_custom_headers_for_mcp_logging`` wiring into
  ``litellm_params.metadata`` (the path used by ``execute_mcp_tool``).
"""

import logging

from litellm.proxy._experimental.mcp_server.server import (
    _apply_requester_custom_headers_for_mcp_logging,
    _extract_custom_headers,
)


class _FakeLiteLLMLogging:
    __slots__ = ("model_call_details",)

    def __init__(self, model_call_details: dict):
        self.model_call_details = model_call_details


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

    def test_excludes_common_auth_style_x_headers(self):
        raw = {
            "x-auth-token": "secret",
            "x-access-token": "secret",
            "x-goog-api-key": "secret",
            "x-forwarded-authorization": "secret",
            "x-safe-correlation": "ok",
        }
        result = _extract_custom_headers(raw)
        assert result == {"x-safe-correlation": "ok"}

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


class TestApplyRequesterCustomHeadersForMcpLogging:
    """Unit tests for metadata wiring used by execute_mcp_tool."""

    def test_sets_requester_custom_headers_when_metadata_is_dict(self):
        meta: dict = {}
        fake = _FakeLiteLLMLogging({"litellm_params": {"metadata": meta}})
        headers = {"x-correlation-id": "abc"}
        _apply_requester_custom_headers_for_mcp_logging(fake, headers)
        assert meta["requester_custom_headers"] == headers

    def test_noop_when_custom_headers_empty(self):
        meta = {"before": True}
        fake = _FakeLiteLLMLogging({"litellm_params": {"metadata": meta}})
        _apply_requester_custom_headers_for_mcp_logging(fake, None)
        _apply_requester_custom_headers_for_mcp_logging(fake, {})
        assert "requester_custom_headers" not in meta

    def test_skips_when_litellm_params_not_dict(self, caplog):
        fake = _FakeLiteLLMLogging({"litellm_params": None})
        with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
            _apply_requester_custom_headers_for_mcp_logging(fake, {"x": "y"})
        assert "skipping requester_custom_headers" in caplog.text
        assert "litellm_params is not a dict" in caplog.text

    def test_skips_when_metadata_not_dict(self, caplog):
        fake = _FakeLiteLLMLogging({"litellm_params": {"metadata": object()}})
        with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
            _apply_requester_custom_headers_for_mcp_logging(fake, {"x": "y"})
        assert "skipping requester_custom_headers" in caplog.text
        assert "metadata is not a dict" in caplog.text
