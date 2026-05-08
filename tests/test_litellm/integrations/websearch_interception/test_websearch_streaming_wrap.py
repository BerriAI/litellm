"""
Unit tests for _maybe_wrap_in_fake_stream in BaseLLMHTTPHandler.

Tests that agentic loop responses are correctly wrapped in
FakeAnthropicMessagesStreamIterator when the original request was streaming.
"""

from unittest.mock import MagicMock


from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)


class TestMaybeWrapInFakeStream:
    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()

    def test_wraps_dict_when_converted_stream_flag_is_true(self):
        """When websearch_interception_converted_stream is True and response is dict, wrap it."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "websearch_interception_converted_stream": True
        }
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        result = self.handler._maybe_wrap_in_fake_stream(response, logging_obj)

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    def test_returns_response_unchanged_when_flag_is_false(self):
        """When flag is False, return response as-is."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "websearch_interception_converted_stream": False
        }
        response = {"id": "msg_123", "content": []}

        result = self.handler._maybe_wrap_in_fake_stream(response, logging_obj)

        assert result is response

    def test_returns_response_unchanged_when_not_dict(self):
        """When response is not a dict (e.g., already a stream), return as-is."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "websearch_interception_converted_stream": True
        }
        response = MagicMock()  # Not a dict

        result = self.handler._maybe_wrap_in_fake_stream(response, logging_obj)

        assert result is response

    def test_returns_response_when_logging_obj_is_none(self):
        """When logging_obj is None, return response as-is."""
        response = {"id": "msg_123", "content": []}

        result = self.handler._maybe_wrap_in_fake_stream(response, None)

        assert result is response
