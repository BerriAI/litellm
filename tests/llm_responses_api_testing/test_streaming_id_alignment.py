"""
Tests for streaming response ID alignment between client-facing IDs and SpendLogs.

The LiteLLMCompletionStreamingIterator must produce response IDs that, when decoded,
yield the same chatcmpl-{uuid} stored as request_id in SpendLogs.  This ensures
GET /responses/{id}, DELETE /responses/{id}, and multi-turn previous_response_id
all work correctly.
"""

import base64
import os
import sys
import uuid
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta


def _make_chunk(chat_completion_id: str = "chatcmpl-test123", content: str = "Hi") -> ModelResponseStream:
    """Create a minimal ModelResponseStream chunk."""
    return ModelResponseStream(
        id=chat_completion_id,
        choices=[StreamingChoices(index=0, delta=Delta(content=content))],
    )


def _make_iterator(
    custom_llm_provider: str = "bedrock",
    model_id: Optional[str] = "model-abc",
) -> LiteLLMCompletionStreamingIterator:
    """Create an iterator with mocked stream wrapper."""
    mock_stream = MagicMock()
    mock_stream.logging_obj = None

    metadata = {}
    if model_id:
        metadata["model_info"] = {"id": model_id}

    return LiteLLMCompletionStreamingIterator(
        model="claude-sonnet-4-20250514",
        litellm_custom_stream_wrapper=mock_stream,
        request_input="Hello",
        responses_api_request={},
        custom_llm_provider=custom_llm_provider,
        litellm_metadata=metadata,
    )


class TestStreamingIdAlignment:
    """Verify that the streaming iterator produces decodable response IDs."""

    def test_build_encoded_response_id_roundtrip(self):
        """Encoding then decoding should return the original chatcmpl ID."""
        it = _make_iterator()
        chatcmpl_id = "chatcmpl-abc123"

        encoded = it._build_encoded_response_id(chatcmpl_id)

        # Must start with resp_
        assert encoded.startswith("resp_")

        # Decode and verify roundtrip
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(encoded)
        assert decoded["response_id"] == chatcmpl_id
        assert decoded["custom_llm_provider"] == "bedrock"
        assert decoded["model_id"] == "model-abc"

    @pytest.mark.asyncio
    async def test_prefetch_sets_cached_response_id(self):
        """_prefetch_first_chunk_async should read first chunk and cache encoded ID."""
        chatcmpl_id = f"chatcmpl-{uuid.uuid4()}"
        chunk = _make_chunk(chatcmpl_id)

        it = _make_iterator()
        it.litellm_custom_stream_wrapper.__aiter__ = MagicMock(return_value=it.litellm_custom_stream_wrapper)
        it.litellm_custom_stream_wrapper.__anext__ = AsyncMock(return_value=chunk)

        await it._prefetch_first_chunk_async()

        assert it._first_chunk_resolved is True
        assert it._first_chunk is chunk  # buffered for later processing
        assert it._cached_response_id is not None
        assert it._cached_response_id.startswith("resp_")

        # Verify the inner ID matches the chatcmpl ID
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            it._cached_response_id
        )
        assert decoded["response_id"] == chatcmpl_id

    def test_prefetch_sync_sets_cached_response_id(self):
        """_prefetch_first_chunk_sync should read first chunk and cache encoded ID."""
        chatcmpl_id = f"chatcmpl-{uuid.uuid4()}"
        chunk = _make_chunk(chatcmpl_id)

        it = _make_iterator()
        it.litellm_custom_stream_wrapper.__iter__ = MagicMock(return_value=it.litellm_custom_stream_wrapper)
        it.litellm_custom_stream_wrapper.__next__ = MagicMock(return_value=chunk)

        it._prefetch_first_chunk_sync()

        assert it._first_chunk_resolved is True
        assert it._cached_response_id is not None

        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            it._cached_response_id
        )
        assert decoded["response_id"] == chatcmpl_id

    def test_response_created_event_uses_encoded_id(self):
        """The response.created event data should use the pre-resolved encoded ID."""
        it = _make_iterator()
        chatcmpl_id = "chatcmpl-xyz789"
        it._cached_response_id = it._build_encoded_response_id(chatcmpl_id)

        event_data = it._default_response_created_event_data()

        # The event ID should be the same encoded ID
        assert event_data["id"] == it._cached_response_id
        assert event_data["id"].startswith("resp_")

        # And it should decode to our chatcmpl ID
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            event_data["id"]
        )
        assert decoded["response_id"] == chatcmpl_id

    def test_fallback_id_is_still_decodable(self):
        """When no chunk is prefetched, fallback ID should still be decodable."""
        it = _make_iterator()
        assert it._cached_response_id is None

        event_data = it._default_response_created_event_data()

        # Should have generated a fallback encoded ID
        assert event_data["id"].startswith("resp_")

        # Should be decodable
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            event_data["id"]
        )
        assert decoded["response_id"].startswith("chatcmpl-")
        assert decoded["custom_llm_provider"] == "bedrock"

    def test_id_matches_spend_logs_request_id(self):
        """The decoded response_id should match what SpendLogs stores as request_id.

        SpendLogs stores response_obj.get("id") which is the chatcmpl-{uuid} ID
        from ModelResponse. Our encoded resp_ ID must decode to that same value.
        """
        chatcmpl_id = "chatcmpl-spend-logs-test"
        it = _make_iterator()

        encoded = it._build_encoded_response_id(chatcmpl_id)
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(encoded)

        # This is the critical assertion: decoded response_id == SpendLogs request_id
        assert decoded["response_id"] == chatcmpl_id
