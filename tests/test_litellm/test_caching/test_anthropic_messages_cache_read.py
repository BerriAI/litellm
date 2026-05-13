"""
Tests for _convert_cached_result_to_model_response handling of
call_type="anthropic_messages".

Verifies:
1. Non-streaming: returns the cached dict as-is
2. Streaming: returns a FakeAnthropicMessagesStreamIterator
"""

import datetime
import sys
from unittest.mock import MagicMock, patch

import pytest

from litellm.caching.caching_handler import LLMCachingHandler
from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.types.utils import CallTypes

# Patch update_response_metadata where it's actually used in caching_handler
_CACHING_HANDLER_MODULE = sys.modules["litellm.caching.caching_handler"]


def _make_caching_handler() -> LLMCachingHandler:
    """Create a LLMCachingHandler with minimal mocked dependencies."""
    handler = LLMCachingHandler.__new__(LLMCachingHandler)
    handler.start_time = datetime.datetime.now()
    return handler


def _sample_anthropic_response() -> dict:
    """Return a minimal valid AnthropicMessagesResponse dict."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
        },
    }


class TestAnthropicMessagesCacheRead:
    """Tests for cache read conversion with anthropic_messages call type."""

    @patch.object(_CACHING_HANDLER_MODULE, "update_response_metadata")
    def test_non_streaming_returns_dict_as_is(self, mock_update_metadata):
        """Non-streaming anthropic_messages cache hit returns the dict unchanged."""
        handler = _make_caching_handler()
        cached_dict = _sample_anthropic_response()
        logging_obj = MagicMock()

        result = handler._convert_cached_result_to_model_response(
            cached_result=cached_dict,
            call_type=CallTypes.anthropic_messages.value,
            kwargs={"stream": False},
            logging_obj=logging_obj,
            model="claude-3-5-sonnet-20241022",
            args=(),
            custom_llm_provider="anthropic",
        )

        # Should return the same dict object, not wrapped
        assert result is cached_dict
        assert isinstance(result, dict)
        assert result["id"] == "msg_123"
        assert result["content"][0]["text"] == "Hello!"

    @patch.object(_CACHING_HANDLER_MODULE, "update_response_metadata")
    def test_streaming_returns_fake_stream_iterator(self, mock_update_metadata):
        """Streaming anthropic_messages cache hit returns FakeAnthropicMessagesStreamIterator."""
        handler = _make_caching_handler()
        cached_dict = _sample_anthropic_response()
        logging_obj = MagicMock()

        result = handler._convert_cached_result_to_model_response(
            cached_result=cached_dict,
            call_type=CallTypes.anthropic_messages.value,
            kwargs={"stream": True},
            logging_obj=logging_obj,
            model="claude-3-5-sonnet-20241022",
            args=(),
            custom_llm_provider="anthropic",
        )

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)
        # Verify the iterator was constructed with the response dict
        assert result.response is cached_dict

    @patch.object(_CACHING_HANDLER_MODULE, "update_response_metadata")
    def test_non_streaming_without_stream_key(self, mock_update_metadata):
        """When stream key is absent from kwargs, should treat as non-streaming."""
        handler = _make_caching_handler()
        cached_dict = _sample_anthropic_response()
        logging_obj = MagicMock()

        result = handler._convert_cached_result_to_model_response(
            cached_result=cached_dict,
            call_type=CallTypes.anthropic_messages.value,
            kwargs={},
            logging_obj=logging_obj,
            model="claude-3-5-sonnet-20241022",
            args=(),
            custom_llm_provider="anthropic",
        )

        # Without stream key, should return dict as-is
        assert result is cached_dict
        assert isinstance(result, dict)

    @patch.object(_CACHING_HANDLER_MODULE, "update_response_metadata")
    def test_does_not_match_non_dict_cached_result(self, mock_update_metadata):
        """If cached_result is not a dict, the anthropic_messages branch is skipped."""
        handler = _make_caching_handler()
        # Simulate a non-dict cached result (e.g., a ModelResponse object)
        cached_result = MagicMock()
        cached_result._hidden_params = {"cache_hit": False}
        logging_obj = MagicMock()

        result = handler._convert_cached_result_to_model_response(
            cached_result=cached_result,
            call_type=CallTypes.anthropic_messages.value,
            kwargs={"stream": False},
            logging_obj=logging_obj,
            model="claude-3-5-sonnet-20241022",
            args=(),
            custom_llm_provider="anthropic",
        )

        # Should not be converted to FakeAnthropicMessagesStreamIterator
        assert not isinstance(result, FakeAnthropicMessagesStreamIterator)
        # The _hidden_params["cache_hit"] should be set to True by the final block
        assert cached_result._hidden_params["cache_hit"] is True
