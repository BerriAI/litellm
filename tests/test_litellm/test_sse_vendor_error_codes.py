"""
Regression tests for SSE vendor error code mapping.

Vendors like ZAI (ZhipuAI) and DashScope return error codes outside the
standard HTTP 100-599 range in SSE chunks (e.g. 1302 for rate limits).
_parse_event_data_for_error() must map these to 502 instead of returning None,
so that create_streaming_response() can detect the error and the router's
fallback/cooldown paths fire correctly.

Fixes: https://github.com/BerriAI/litellm/issues/31284
"""

import json
import pytest

from litellm.proxy.common_request_processing import _parse_event_data_for_error


class TestParseEventDataForError:
    """Unit tests for _parse_event_data_for_error."""

    @pytest.mark.asyncio
    async def test_standard_http_error_code_returned_as_is(self):
        """HTTP codes in 100-599 range are returned unchanged."""
        chunk = 'data: {"error": {"code": 429, "message": "Rate limited"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 429

    @pytest.mark.asyncio
    async def test_standard_http_500_returned_as_is(self):
        chunk = 'data: {"error": {"code": 500, "message": "Server error"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 500

    @pytest.mark.asyncio
    async def test_vendor_code_1302_mapped_to_502(self):
        """ZAI rate-limit code 1302 should map to 502, not silently return None."""
        chunk = 'data: {"error": {"code": 1302, "message": "\\u60a8\\u7684\\u8d26\\u6237\\u5df2\\u8fbe\\u5230\\u901f\\u7387\\u9650\\u5236"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert (
            result == 502
        ), "Vendor code 1302 must be mapped to 502 so the router detects the error"

    @pytest.mark.asyncio
    async def test_vendor_code_1305_mapped_to_502(self):
        """ZAI 1305 (quota exceeded) should map to 502."""
        chunk = 'data: {"error": {"code": 1305, "message": "Quota exceeded"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 502

    @pytest.mark.asyncio
    async def test_vendor_code_4digit_mapped_to_502(self):
        """DashScope-style 4-digit throttle codes (e.g. 4001) should map to 502."""
        chunk = 'data: {"error": {"code": 4001, "message": "Throttled"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 502

    @pytest.mark.asyncio
    async def test_no_error_field_returns_none(self):
        """Normal content chunks (no 'error' key) return None."""
        chunk = 'data: {"choices": [{"delta": {"content": "hello"}}]}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result is None

    @pytest.mark.asyncio
    async def test_done_sentinel_returns_none(self):
        """[DONE] sentinel returns None."""
        result = await _parse_event_data_for_error("data: [DONE]\n")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_data_returns_none(self):
        result = await _parse_event_data_for_error("data: \n")
        assert result is None

    @pytest.mark.asyncio
    async def test_string_vendor_code_mapped_to_502(self):
        """Vendor codes that arrive as strings and are >= 600 still map to 502."""
        chunk = 'data: {"error": {"code": "1302", "message": "Rate limit"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 502

    @pytest.mark.asyncio
    async def test_string_http_code_returned_as_is(self):
        """String HTTP code in 100-599 range is returned as the integer value."""
        chunk = 'data: {"error": {"code": "429", "message": "Rate limit"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 429

    @pytest.mark.asyncio
    async def test_bytes_input_vendor_code_mapped_to_502(self):
        """Byte-string input with vendor code >= 600 maps to 502."""
        chunk = b'data: {"error": {"code": 1302, "message": "Rate limit"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 502

    @pytest.mark.asyncio
    async def test_error_with_none_code_returns_none(self):
        """error object present but 'code' is absent — return None."""
        chunk = 'data: {"error": {"message": "Something went wrong"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result is None

    @pytest.mark.asyncio
    async def test_small_vendor_code_2_mapped_to_502(self):
        """Small integer vendor codes (< 100) must map to 502, not silently return None."""
        chunk = 'data: {"error": {"code": 2, "message": "Unknown error"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert (
            result == 502
        ), "Vendor code 2 must be mapped to 502 so the router detects the error"

    @pytest.mark.asyncio
    async def test_vendor_code_zero_mapped_to_502(self):
        """Code 0 is outside HTTP range and must map to 502."""
        chunk = 'data: {"error": {"code": 0, "message": "Connection reset"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 502

    @pytest.mark.asyncio
    async def test_vendor_code_99_mapped_to_502(self):
        """Code 99 (below HTTP minimum of 100) must map to 502."""
        chunk = 'data: {"error": {"code": 99, "message": "Internal code"}}\n'
        result = await _parse_event_data_for_error(chunk)
        assert result == 502
