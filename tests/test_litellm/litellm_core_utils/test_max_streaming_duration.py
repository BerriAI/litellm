"""
Tests for LITELLM_MAX_STREAMING_DURATION_SECONDS — the global cap on streaming response wall-clock time.

Covers:
  - CustomStreamWrapper (chat/completions) sync + async
  - BaseResponsesAPIStreamingIterator (responses) sync + async
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_custom_stream_wrapper() -> CustomStreamWrapper:
    """Build a minimal CustomStreamWrapper for testing."""
    return CustomStreamWrapper(
        completion_stream=None,
        model="test-model",
        logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )


# ---------------------------------------------------------------------------
# CustomStreamWrapper (chat/completions)
# ---------------------------------------------------------------------------


class TestCustomStreamWrapperMaxDuration:
    def test_should_not_raise_when_duration_is_none(self):
        """No limit configured → never raises."""
        wrapper = _make_custom_stream_wrapper()
        with patch("litellm.constants.LITELLM_MAX_STREAMING_DURATION_SECONDS", None):
            wrapper._check_max_streaming_duration()  # should not raise

    def test_should_not_raise_when_under_limit(self):
        """Stream is under the limit → no error."""
        wrapper = _make_custom_stream_wrapper()
        with patch("litellm.constants.LITELLM_MAX_STREAMING_DURATION_SECONDS", 60.0):
            wrapper._check_max_streaming_duration()  # should not raise

    def test_should_raise_timeout_when_exceeded(self):
        """Stream exceeded the limit → litellm.Timeout."""
        wrapper = _make_custom_stream_wrapper()
        wrapper._stream_created_time = time.time() - 20  # simulate 20s elapsed
        with patch("litellm.constants.LITELLM_MAX_STREAMING_DURATION_SECONDS", 10.0):
            with pytest.raises(litellm.Timeout, match="max streaming duration"):
                wrapper._check_max_streaming_duration()

    def test_should_raise_on_sync_next_when_exceeded(self):
        """__next__ should check the limit before iterating."""
        wrapper = _make_custom_stream_wrapper()
        wrapper._stream_created_time = time.time() - 20
        with patch("litellm.constants.LITELLM_MAX_STREAMING_DURATION_SECONDS", 10.0):
            with pytest.raises(litellm.Timeout):
                wrapper.__next__()

    @pytest.mark.asyncio
    async def test_should_raise_on_async_anext_when_exceeded(self):
        """__anext__ should check the limit before iterating."""
        wrapper = _make_custom_stream_wrapper()
        wrapper._stream_created_time = time.time() - 20
        with patch("litellm.constants.LITELLM_MAX_STREAMING_DURATION_SECONDS", 10.0):
            with pytest.raises(litellm.Timeout):
                await wrapper.__anext__()


# ---------------------------------------------------------------------------
# BaseResponsesAPIStreamingIterator (responses)
# ---------------------------------------------------------------------------

class TestResponsesStreamingIteratorMaxDuration:
    def _make_base_iterator(self):
        """Build a minimal BaseResponsesAPIStreamingIterator for testing."""
        from litellm.responses.streaming_iterator import (
            BaseResponsesAPIStreamingIterator,
        )

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.start_time = time.time()

        mock_provider_config = MagicMock()
        return BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="test-model",
            responses_api_provider_config=mock_provider_config,
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
        )

    def test_should_not_raise_when_duration_is_none(self):
        it = self._make_base_iterator()
        with patch(
            "litellm.responses.streaming_iterator.LITELLM_MAX_STREAMING_DURATION_SECONDS", None
        ):
            it._check_max_streaming_duration()

    def test_should_not_raise_when_under_limit(self):
        it = self._make_base_iterator()
        with patch(
            "litellm.responses.streaming_iterator.LITELLM_MAX_STREAMING_DURATION_SECONDS", 60.0
        ):
            it._check_max_streaming_duration()

    def test_should_raise_timeout_when_exceeded(self):
        it = self._make_base_iterator()
        it._stream_created_time = time.time() - 20
        with patch(
            "litellm.responses.streaming_iterator.LITELLM_MAX_STREAMING_DURATION_SECONDS", 10.0
        ):
            with pytest.raises(litellm.Timeout, match="max streaming duration"):
                it._check_max_streaming_duration()
