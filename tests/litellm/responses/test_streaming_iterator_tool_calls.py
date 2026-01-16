"""
Unit tests for Responses API streaming iterator tool call handling.

Tests cover:
- Tool call delta handling during streaming
- Event emission order for function calls
- Accumulation of tool call arguments across chunks
"""

import os
import sys
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import (
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta


class TestToolCallDeltaHandling:
    """Tests for handling tool call deltas in streaming."""

    def _create_iterator(self):
        """Create a minimal streaming iterator for testing."""
        mock_stream = MagicMock()
        iterator = LiteLLMCompletionStreamingIterator(
            model="oracle/google.gemini-2.5-flash",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="test input",
            responses_api_request={},
        )
        # Skip initial events for testing
        iterator._sent_initial_events = True
        iterator._sent_response_created_event = True
        iterator._sent_first_output_text_item_events = True
        return iterator

    def _create_chunk_with_tool_call(
        self,
        chunk_id: str = "chunk_123",
        tool_index: int = 0,
        tool_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        arguments: str = "",
    ) -> ModelResponseStream:
        """Create a model response chunk with a tool call."""
        tool_call = {
            "index": tool_index,
            "function": {
                "name": tool_name,
                "arguments": arguments,
            },
            "type": "function",
        }
        if tool_id:
            tool_call["id"] = tool_id

        return ModelResponseStream(
            id=chunk_id,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        tool_calls=[tool_call],
                    ),
                    finish_reason=None,
                )
            ],
        )

    def test_handle_tool_call_delta_emits_output_item_added_on_first_chunk(self):
        """
        Test that the first chunk for a tool call emits OutputItemAddedEvent.
        """
        iterator = self._create_iterator()
        chunk = self._create_chunk_with_tool_call(
            tool_id="call_abc123",
            tool_name="get_weather",
            arguments='{"city":'
        )

        result = iterator._handle_tool_call_delta(chunk, chunk.choices[0].delta.tool_calls)

        assert result is not None
        assert isinstance(result, OutputItemAddedEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        assert result.item.type == "function_call"
        assert result.item.name == "get_weather"
        assert result.item.call_id == "call_abc123"

    def test_handle_tool_call_delta_emits_arguments_delta_event(self):
        """
        Test that argument chunks emit FunctionCallArgumentsDeltaEvent.
        """
        iterator = self._create_iterator()

        # First chunk - establishes the tool call
        chunk1 = self._create_chunk_with_tool_call(
            tool_id="call_abc123",
            tool_name="get_weather",
            arguments='{"city":'
        )
        result1 = iterator._handle_tool_call_delta(chunk1, chunk1.choices[0].delta.tool_calls)

        # Should have OutputItemAdded as first event, ArgumentsDelta queued
        assert isinstance(result1, OutputItemAddedEvent)
        assert len(iterator.pending_function_call_events) == 1
        assert isinstance(iterator.pending_function_call_events[0], FunctionCallArgumentsDeltaEvent)
        assert iterator.pending_function_call_events[0].delta == '{"city":'

    def test_handle_tool_call_delta_accumulates_arguments(self):
        """
        Test that arguments are accumulated across multiple chunks.
        """
        iterator = self._create_iterator()

        # Chunk 1: Start of arguments
        chunk1 = self._create_chunk_with_tool_call(
            tool_id="call_abc123",
            tool_name="get_weather",
            arguments='{"city":'
        )
        iterator._handle_tool_call_delta(chunk1, chunk1.choices[0].delta.tool_calls)

        # Clear pending events
        iterator.pending_function_call_events.clear()

        # Chunk 2: More arguments
        chunk2 = self._create_chunk_with_tool_call(
            tool_index=0,
            arguments='"Madrid"}'
        )
        iterator._handle_tool_call_delta(chunk2, chunk2.choices[0].delta.tool_calls)

        # Verify accumulated arguments
        assert iterator.accumulated_tool_calls[0]["arguments"] == '{"city":"Madrid"}'

    def test_handle_tool_call_delta_handles_multiple_tool_calls(self):
        """
        Test handling multiple tool calls in a single response.
        """
        iterator = self._create_iterator()

        # First tool call
        chunk1 = self._create_chunk_with_tool_call(
            tool_index=0,
            tool_id="call_1",
            tool_name="get_weather",
            arguments='{"city":"Madrid"}'
        )
        iterator._handle_tool_call_delta(chunk1, chunk1.choices[0].delta.tool_calls)

        # Second tool call
        chunk2 = self._create_chunk_with_tool_call(
            tool_index=1,
            tool_id="call_2",
            tool_name="get_time",
            arguments='{}'
        )
        iterator._handle_tool_call_delta(chunk2, chunk2.choices[0].delta.tool_calls)

        # Verify both tool calls are tracked
        assert len(iterator.accumulated_tool_calls) == 2
        assert iterator.accumulated_tool_calls[0]["name"] == "get_weather"
        assert iterator.accumulated_tool_calls[1]["name"] == "get_time"

    def test_emit_function_call_done_events(self):
        """
        Test that done events are properly emitted at end of stream.
        """
        iterator = self._create_iterator()

        # Set up accumulated tool calls
        iterator.accumulated_tool_calls = {
            0: {
                "id": "call_abc123",
                "name": "get_weather",
                "arguments": '{"city":"Madrid"}',
                "output_index": 1,
            }
        }

        done_events = iterator._emit_function_call_done_events()

        # Should emit ArgumentsDone and OutputItemDone
        assert len(done_events) == 2
        assert isinstance(done_events[0], FunctionCallArgumentsDoneEvent)
        assert done_events[0].arguments == '{"city":"Madrid"}'
        assert isinstance(done_events[1], OutputItemDoneEvent)
        assert done_events[1].item.name == "get_weather"
        assert done_events[1].item.status == "completed"

    def test_emit_function_call_done_events_multiple_tools(self):
        """
        Test done events for multiple tool calls.
        """
        iterator = self._create_iterator()

        # Set up multiple accumulated tool calls
        iterator.accumulated_tool_calls = {
            0: {
                "id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Madrid"}',
                "output_index": 1,
            },
            1: {
                "id": "call_2",
                "name": "get_time",
                "arguments": '{}',
                "output_index": 2,
            }
        }

        done_events = iterator._emit_function_call_done_events()

        # Should emit 2 events per tool call (ArgumentsDone + OutputItemDone)
        assert len(done_events) == 4

    def test_tool_call_with_generated_id_fallback(self):
        """
        Test that when tool_id is missing, we generate a fallback ID.
        """
        iterator = self._create_iterator()

        # Tool call without ID (like OCI Gemini returns)
        chunk = self._create_chunk_with_tool_call(
            tool_index=0,
            tool_id=None,  # No ID provided
            tool_name="get_weather",
            arguments='{"city":"Madrid"}'
        )
        iterator._handle_tool_call_delta(chunk, chunk.choices[0].delta.tool_calls)

        # Should have generated a fallback ID
        assert "id" in iterator.accumulated_tool_calls[0]
        assert iterator.accumulated_tool_calls[0]["id"] == "call_0"


class TestStreamingEventOrder:
    """Tests for correct event ordering in streaming."""

    def _create_iterator(self):
        """Create a minimal streaming iterator for testing."""
        mock_stream = MagicMock()
        iterator = LiteLLMCompletionStreamingIterator(
            model="oracle/google.gemini-2.5-flash",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="test input",
            responses_api_request={},
        )
        return iterator

    def test_stream_ended_flag_triggers_done_events(self):
        """
        Test that when stream ends, done events are queued.
        """
        iterator = self._create_iterator()
        iterator._sent_initial_events = True
        iterator._sent_response_created_event = True
        iterator._sent_first_output_text_item_events = True

        # Simulate accumulated tool calls
        iterator.accumulated_tool_calls = {
            0: {
                "id": "call_123",
                "name": "test_func",
                "arguments": "{}",
                "output_index": 1,
            }
        }

        # Simulate stream ending
        iterator.stream_ended = True
        iterator.pending_function_call_done_events = iterator._emit_function_call_done_events()

        # Verify done events were queued
        assert len(iterator.pending_function_call_done_events) == 2
