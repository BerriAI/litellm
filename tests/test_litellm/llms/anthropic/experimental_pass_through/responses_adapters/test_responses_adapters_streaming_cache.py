"""
Tests for cache token extraction in AnthropicResponsesStreamWrapper.

Verifies that the streaming adapter correctly extracts cache_read_input_tokens
from both OpenAI Responses API (input_tokens_details.cached_tokens) and
Anthropic-native fields (cache_read_input_tokens).

Fixes: https://github.com/BerriAI/litellm/issues/28354
"""

import os
import sys
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


def _make_wrapper() -> AnthropicResponsesStreamWrapper:
    """Create a wrapper with a mock stream."""
    mock_stream = MagicMock()
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=mock_stream,
        model="gpt-4o",
    )
    return wrapper


def _make_completed_event(
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    use_openai_format: bool = True,
) -> MagicMock:
    """Build a mock response.completed event."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    if use_openai_format and cached_tokens:
        input_details = MagicMock()
        input_details.cached_tokens = cached_tokens
        usage.input_tokens_details = input_details
    else:
        usage.input_tokens_details = None

    usage.output_tokens_details = None

    # Set or delete Anthropic-native fields
    if cache_creation_input_tokens:
        usage.cache_creation_input_tokens = cache_creation_input_tokens
    else:
        # Use spec to make getattr return 0
        usage.cache_creation_input_tokens = 0

    if cache_read_input_tokens:
        usage.cache_read_input_tokens = cache_read_input_tokens
    else:
        usage.cache_read_input_tokens = 0

    response = MagicMock()
    response.id = "resp_test"
    response.model = "gpt-4o"
    response.status = "completed"
    response.output = []
    response.usage = usage

    event = MagicMock()
    event.type = "response.completed"
    event.response = response
    # Make dict-like access fail so getattr path is used
    event.__contains__ = lambda self, key: False
    event.get = lambda key, default=None: default if key != "response" else response

    return event


class TestStreamingCacheTokenExtraction:
    """Streaming adapter extracts cache tokens from OpenAI and Anthropic usage."""

    def test_openai_cached_tokens_extracted(self):
        """input_tokens_details.cached_tokens -> cache_read_input_tokens in usage delta."""
        wrapper = _make_wrapper()
        event = _make_completed_event(
            input_tokens=1000,
            output_tokens=50,
            cached_tokens=800,
            use_openai_format=True,
        )
        wrapper._process_event(event)

        # Find the message_delta chunk
        delta_chunks = [
            c for c in wrapper._chunk_queue if c.get("type") == "message_delta"
        ]
        assert len(delta_chunks) == 1
        usage = delta_chunks[0]["usage"]
        assert usage["cache_read_input_tokens"] == 800

    def test_anthropic_native_fields_used_as_fallback(self):
        """Anthropic-native cache fields override when present."""
        wrapper = _make_wrapper()
        event = _make_completed_event(
            input_tokens=1000,
            output_tokens=50,
            cached_tokens=0,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=700,
            use_openai_format=False,
        )
        wrapper._process_event(event)

        delta_chunks = [
            c for c in wrapper._chunk_queue if c.get("type") == "message_delta"
        ]
        assert len(delta_chunks) == 1
        usage = delta_chunks[0]["usage"]
        assert usage["cache_creation_input_tokens"] == 300
        assert usage["cache_read_input_tokens"] == 700

    def test_no_cache_tokens_omitted_from_usage(self):
        """When no cache tokens, fields are omitted from usage delta."""
        wrapper = _make_wrapper()
        event = _make_completed_event(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            use_openai_format=False,
        )
        wrapper._process_event(event)

        delta_chunks = [
            c for c in wrapper._chunk_queue if c.get("type") == "message_delta"
        ]
        assert len(delta_chunks) == 1
        usage = delta_chunks[0]["usage"]
        assert "cache_read_input_tokens" not in usage
        assert "cache_creation_input_tokens" not in usage
