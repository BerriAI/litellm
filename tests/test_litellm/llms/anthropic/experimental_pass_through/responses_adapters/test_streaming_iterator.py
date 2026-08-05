"""
Tests for AnthropicResponsesStreamWrapper cache token usage mapping.

Bug: streaming_iterator.py assigned input_tokens_details (an object) to
cache_creation_tokens, and output_tokens_details (wrong field entirely) to
cache_read_tokens. Both caused streaming responses to report zero cache tokens.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


def _make_completed_event(usage: MagicMock, status: str = "completed") -> MagicMock:
    response_obj = MagicMock()
    response_obj.status = status
    response_obj.usage = usage
    response_obj.output = []

    event = MagicMock()
    event.type = "response.completed"
    event.response = response_obj
    return event


def _get_message_delta_usage(wrapper: AnthropicResponsesStreamWrapper) -> dict:
    for chunk in wrapper._chunk_queue:
        if chunk.get("type") == "message_delta":
            return chunk["usage"]
    raise AssertionError("no message_delta chunk was queued")


class TestStreamingCacheTokenMapping:
    """response.completed -> message_delta usage cache token mapping."""

    def test_should_read_cache_creation_tokens_as_int_not_object(self):
        """cache_creation_input_tokens must be an integer count, not input_tokens_details."""
        usage = MagicMock()
        usage.input_tokens = 1936
        usage.output_tokens = 246
        usage.cache_creation_input_tokens = 1900
        usage.cache_read_input_tokens = 0
        usage.input_tokens_details = MagicMock(cached_tokens=0)

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=iter([]), model="m")
        wrapper._process_event(_make_completed_event(usage))

        usage_delta = _get_message_delta_usage(wrapper)
        assert usage_delta["cache_creation_input_tokens"] == 1900
        assert "cache_read_input_tokens" not in usage_delta

    def test_should_read_cache_read_tokens_from_correct_field(self):
        """cache_read_input_tokens must come from cache_read_input_tokens, not output_tokens_details."""
        usage = MagicMock()
        usage.input_tokens = 1936
        usage.output_tokens = 246
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 1900
        usage.input_tokens_details = MagicMock(cached_tokens=0)

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=iter([]), model="m")
        wrapper._process_event(_make_completed_event(usage))

        usage_delta = _get_message_delta_usage(wrapper)
        assert usage_delta["cache_read_input_tokens"] == 1900
        assert "cache_creation_input_tokens" not in usage_delta

    def test_should_fall_back_to_input_tokens_details_cached_tokens(self):
        """When cache_read_input_tokens is 0, fall back to input_tokens_details.cached_tokens."""
        usage = MagicMock()
        usage.input_tokens = 1936
        usage.output_tokens = 246
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 0
        usage.input_tokens_details = MagicMock(cached_tokens=1900)

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=iter([]), model="m")
        wrapper._process_event(_make_completed_event(usage))

        usage_delta = _get_message_delta_usage(wrapper)
        assert usage_delta["cache_read_input_tokens"] == 1900

    def test_should_not_add_cache_keys_when_no_cache_activity(self):
        """No cache_* keys when there's no cache activity."""
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 0
        usage.input_tokens_details = MagicMock(cached_tokens=0)

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=iter([]), model="m")
        wrapper._process_event(_make_completed_event(usage))

        usage_delta = _get_message_delta_usage(wrapper)
        assert "cache_creation_input_tokens" not in usage_delta
        assert "cache_read_input_tokens" not in usage_delta
        assert usage_delta["input_tokens"] == 100
        assert usage_delta["output_tokens"] == 50

    def test_should_handle_input_tokens_details_none(self):
        """input_tokens_details=None should not raise when falling back."""
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 0
        usage.input_tokens_details = None

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=iter([]), model="m")
        wrapper._process_event(_make_completed_event(usage))

        usage_delta = _get_message_delta_usage(wrapper)
        assert "cache_read_input_tokens" not in usage_delta
