"""
Unit tests for fix #28553: stream_options should NOT be injected for Responses API routes.

The proxy's `common_processing_pre_call_logic` injects `stream_options={'include_usage': True}`
when `always_include_stream_usage` is enabled. This must NOT happen for Responses API routes
(`aresponses`, `_aresponses_websocket`) because the Responses API does not support
`stream_options` — usage is included automatically in response.completed events.
"""

import pytest

from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


class TestStreamOptionsNotInjectedForResponsesAPI:
    """Verify stream_options is skipped for Responses API routes."""

    @pytest.mark.parametrize("route_type", ["aresponses", "_aresponses_websocket"])
    def test_stream_options_not_injected_for_responses_routes(self, route_type):
        """stream_options must NOT be added when route is a Responses API route."""
        data = {"stream": True, "model": "gpt-4"}
        ProxyBaseLLMRequestProcessing._apply_stream_options_for_usage(
            data, {"always_include_stream_usage": True}, route_type
        )
        assert "stream_options" not in data

    def test_stream_options_injected_for_chat_completions(self):
        """stream_options SHOULD be added for acompletion route."""
        data = {"stream": True, "model": "gpt-4"}
        ProxyBaseLLMRequestProcessing._apply_stream_options_for_usage(
            data, {"always_include_stream_usage": True}, "acompletion"
        )
        assert data["stream_options"] == {"include_usage": True}

    def test_stream_options_not_injected_when_disabled(self):
        """stream_options should NOT be added when always_include_stream_usage is False."""
        data = {"stream": True, "model": "gpt-4"}
        ProxyBaseLLMRequestProcessing._apply_stream_options_for_usage(
            data, {"always_include_stream_usage": False}, "acompletion"
        )
        assert "stream_options" not in data

    def test_existing_stream_options_not_overwritten(self):
        """If client already set stream_options with include_usage, don't overwrite."""
        data = {
            "stream": True,
            "model": "gpt-4",
            "stream_options": {"include_usage": False},
        }
        ProxyBaseLLMRequestProcessing._apply_stream_options_for_usage(
            data, {"always_include_stream_usage": True}, "acompletion"
        )
        assert data["stream_options"] == {"include_usage": False}

    def test_non_streaming_request_skipped(self):
        """stream_options should NOT be added for non-streaming requests."""
        data = {"stream": False, "model": "gpt-4"}
        ProxyBaseLLMRequestProcessing._apply_stream_options_for_usage(
            data, {"always_include_stream_usage": True}, "acompletion"
        )
        assert "stream_options" not in data
