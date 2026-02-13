"""
Tests for Gemini streaming finish_reason handling.

Regression tests for issue #21041: Gemini returns finish_reason="stop" instead
of "tool_calls" in streaming mode when tool calls are present.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)


def _make_tool_call_chunk() -> ModelResponseStream:
    """Create a streaming chunk with tool_calls and no finish_reason."""
    return ModelResponseStream(
        id="chatcmpl-test-123",
        model="gemini-3-flash-preview",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id="call_xyz",
                            function=Function(
                                arguments='{"command": "ls -F"}',
                                name="bash",
                            ),
                            type="function",
                            index=0,
                        )
                    ],
                ),
            )
        ],
    )


def _make_finish_chunk(finish_reason: str = "stop") -> ModelResponseStream:
    """Create a streaming chunk with finish_reason and empty delta."""
    return ModelResponseStream(
        id="chatcmpl-test-123",
        model="gemini-3-flash-preview",
        choices=[
            StreamingChoices(
                finish_reason=finish_reason,
                index=0,
                delta=Delta(content=None),
            )
        ],
    )


def test_streaming_finish_reason_overridden_to_tool_calls():
    """
    When a streaming response has tool_calls in an earlier chunk and
    finish_reason="stop" in a later chunk, the final chunk should have
    finish_reason="tool_calls".

    This is a regression test for issue #21041.
    """
    chunks = [_make_tool_call_chunk(), _make_finish_chunk("stop")]

    mock_logging = MagicMock()
    mock_logging.completion_start_time = None
    mock_logging._update_completion_start_time = MagicMock()

    stream_wrapper = CustomStreamWrapper(
        completion_stream=iter(chunks),
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=mock_logging,
    )

    collected_chunks = []
    for chunk in stream_wrapper:
        collected_chunks.append(chunk)

    # Find the chunk with a finish_reason set
    finish_reasons = [
        c.choices[0].finish_reason
        for c in collected_chunks
        if c.choices[0].finish_reason is not None
    ]

    assert len(finish_reasons) > 0, "Expected at least one chunk with finish_reason"
    assert finish_reasons[-1] == "tool_calls", (
        f"Expected final finish_reason to be 'tool_calls', got '{finish_reasons[-1]}'. "
        f"Gemini returns STOP even when tool calls are present, "
        f"the streaming handler should override this to 'tool_calls'."
    )


def test_streaming_finish_reason_stop_without_tools():
    """
    When a streaming response has no tool calls and finish_reason="stop",
    the final chunk should keep finish_reason="stop" (no override).
    """
    text_chunk = ModelResponseStream(
        id="chatcmpl-test-456",
        model="gemini-3-flash-preview",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(role="assistant", content="Hello!"),
            )
        ],
    )
    finish_chunk = _make_finish_chunk("stop")
    finish_chunk.id = "chatcmpl-test-456"

    mock_logging = MagicMock()
    mock_logging.completion_start_time = None
    mock_logging._update_completion_start_time = MagicMock()

    stream_wrapper = CustomStreamWrapper(
        completion_stream=iter([text_chunk, finish_chunk]),
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=mock_logging,
    )

    collected_chunks = []
    for chunk in stream_wrapper:
        collected_chunks.append(chunk)

    finish_reasons = [
        c.choices[0].finish_reason
        for c in collected_chunks
        if c.choices[0].finish_reason is not None
    ]

    assert len(finish_reasons) > 0
    assert finish_reasons[-1] == "stop", (
        f"Expected finish_reason='stop' for text-only response, got '{finish_reasons[-1]}'"
    )
