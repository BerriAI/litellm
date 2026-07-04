"""
Tests for BaseAnthropicMessagesStreamingIterator, in particular that
end-of-stream logging (used by every provider that streams through
async_sse_wrapper, e.g. Bedrock invoke-with-response-stream) is scheduled
reliably instead of via a bare, unreferenced asyncio.create_task -- see #32019.
"""

import os
import sys
import time
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    BaseAnthropicMessagesStreamingIterator,
)
from litellm.types.utils import CallTypes


def _anthropic_sse_events() -> List[Dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]


async def _fake_completion_stream() -> AsyncIterator[Dict[str, Any]]:
    for event in _anthropic_sse_events():
        yield event


def _make_logging_obj() -> LitellmLogging:
    logging_obj = LitellmLogging(
        model="bedrock/us.anthropic.claude-opus-4-7",
        messages=[{"role": "user", "content": "hey"}],
        stream=True,
        call_type=CallTypes.anthropic_messages.value,
        start_time=time.time(),
        litellm_call_id="test-call-id-32019",
        function_id="1",
    )
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.model_call_details["custom_llm_provider"] = "bedrock"
    logging_obj.stream = True
    return logging_obj


class _StreamOnlyLogger(CustomLogger):
    """Mirrors S3Logger: only overrides async_log_success_event, matching
    the class of callback (s3_v2) affected by #32019."""

    def __init__(self):
        self.async_log_success_event = AsyncMock()


@pytest.mark.asyncio
async def test_handle_streaming_logging_uses_global_logging_worker_not_bare_task():
    """Regression test for #32019: end-of-stream logging must be scheduled via
    GLOBAL_LOGGING_WORKER (which holds a strong reference to the coroutine
    until it completes), not a bare asyncio.create_task whose only reference
    is a local variable that disappears as soon as the generator returns.
    """
    logging_obj = _make_logging_obj()
    handler = BaseAnthropicMessagesStreamingIterator(
        litellm_logging_obj=logging_obj,
        request_body={"model": "us.anthropic.claude-opus-4-7", "stream": True},
    )

    with patch.object(GLOBAL_LOGGING_WORKER, "ensure_initialized_and_enqueue") as mock_enqueue:
        async for _ in handler.async_sse_wrapper(_fake_completion_stream()):
            pass

    mock_enqueue.assert_called_once()
    assert "async_coroutine" in mock_enqueue.call_args.kwargs
    # Silence "coroutine was never awaited" since we intercepted the enqueue call.
    mock_enqueue.call_args.kwargs["async_coroutine"].close()


@pytest.mark.asyncio
async def test_streaming_anthropic_messages_reaches_success_only_callback():
    """End-to-end regression test for #32019: a callback that (like s3_v2)
    only implements async_log_success_event -- not async_log_stream_event --
    must still receive the assembled response once the stream finishes and
    the logging worker has drained, for a Bedrock-style provider that streams
    through async_sse_wrapper directly (not chunk_processor).
    """
    logging_obj = _make_logging_obj()
    handler = BaseAnthropicMessagesStreamingIterator(
        litellm_logging_obj=logging_obj,
        request_body={"model": "us.anthropic.claude-opus-4-7", "stream": True},
    )

    mock_callback = _StreamOnlyLogger()
    original_async_callbacks = list(litellm._async_success_callback or [])
    litellm._async_success_callback = [mock_callback]

    try:
        async for _ in handler.async_sse_wrapper(_fake_completion_stream()):
            pass

        await GLOBAL_LOGGING_WORKER.flush()

        mock_callback.async_log_success_event.assert_awaited_once()
        _, call_kwargs = mock_callback.async_log_success_event.call_args
        response_obj = call_kwargs["response_obj"]
        assert response_obj.choices[0].message.content == "Hello"
    finally:
        litellm._async_success_callback = original_async_callbacks
