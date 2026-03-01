"""
Tests for _format_timeout helper in http_handler.

Regression test for https://github.com/BerriAI/litellm/issues/14635
which reports 'Connection timed out after None seconds.' error messages.
"""

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import _format_timeout


class TestFormatTimeout:
    def test_none_returns_descriptive_string(self):
        result = _format_timeout(None)
        assert result == "default (client-level) timeout"

    def test_float_returns_number_with_unit(self):
        assert _format_timeout(5.0) == "5.0s"

    def test_int_returns_number_with_unit(self):
        assert _format_timeout(30) == "30s"

    def test_httpx_timeout_returns_repr(self):
        t = httpx.Timeout(timeout=10.0, connect=5.0)
        result = _format_timeout(t)
        assert "10" in result  # should mention the timeout value

    def test_zero_returns_zero_with_unit(self):
        assert _format_timeout(0) == "0s"
