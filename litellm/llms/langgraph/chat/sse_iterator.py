"""
SSE Stream Iterator for LangGraph.

Handles Server-Sent Events (SSE) streaming responses from LangGraph.
"""

import json
import uuid
from typing import Any, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class LangGraphSSEStreamIterator:
    """
    Iterator for LangGraph SSE streaming responses.
    Supports both sync and async iteration.

    Supports two LangGraph SSE formats:

    1. Standard SSE (``stream_mode="messages"``):
       ``event: messages\\ndata: [{message_obj}, {metadata}]``

    2. Legacy tuple format (``stream_mode="messages-tuple"``):
       ``data: ["messages", payload]``
    """

    # Event types we act on
    _MESSAGE_EVENTS = frozenset({"messages", "messages/partial", "messages/complete"})
    _METADATA_EVENTS = frozenset({"metadata"})

    def __init__(self, response: httpx.Response, model: str):
        self.response = response
        self.model = model
        self.finished = False
        self.line_iterator = None
        self.async_line_iterator = None
        # Tracks the most recent ``event:`` header across SSE lines
        self._current_event_type: Optional[str] = None

    def __iter__(self):
        """Initialize sync iteration."""
        self.line_iterator = self.response.iter_lines()
        return self

    def __aiter__(self):
        """Initialize async iteration."""
        self.async_line_iterator = self.response.aiter_lines()
        return self

    def _parse_sse_line(self, line: str) -> Optional[ModelResponseStream]:
        """
        Parse a single SSE line.

        Per the SSE specification an event block looks like::

            event: <type>\\n
            data: <payload>\\n
            \\n

        The ``event:`` line is optional; when absent the event type defaults
        to ``"message"``.  We track event type across calls so that when
        ``data:`` arrives we already know the type.
        """
        line = line.strip()
        if not line:
            # Blank line = end of SSE event block; reset tracked type
            self._current_event_type = None
            return None

        # Capture ``event:`` header for the next ``data:`` line
        if line.startswith("event:"):
            self._current_event_type = line[6:].strip()
            return None

        # Handle ``data:`` lines
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if not json_str:
                return None

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                return None

            result = self._process_data(data, self._current_event_type)
            # Reset after consuming so repeated calls don't re-use a stale type
            self._current_event_type = None
            return result

        return None

    # Data processing

    def _process_data(
        self, data: Any, event_type: Optional[str] = None
    ) -> Optional[ModelResponseStream]:
        """
        Route parsed JSON *data* using *event_type* (from ``event:`` header).

        Backward-compatible: when no ``event:`` header was present and *data*
        is a list whose first element is a string, we fall back to the legacy
        tuple format ``["messages", payload]``.
        """

        # Legacy tuple format: ["messages", payload]
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[0], str):
            legacy_event = data[0]
            payload = data[1]
            if legacy_event == "messages":
                return self._process_messages_event(payload)
            if legacy_event == "metadata":
                return self._process_metadata_event(payload)
            return None

        # Standard SSE: event type comes from the header 
        if event_type is not None:
            if event_type in self._MESSAGE_EVENTS:
                return self._process_messages_event(data)
            if event_type in self._METADATA_EVENTS:
                return self._process_metadata_event(data)
            # Unknown event type – fall through to heuristic handling
            verbose_logger.debug(f"Ignoring unknown LangGraph event type: {event_type}")

        # Heuristic fallback for headerless dict payloads
        if isinstance(data, dict):
            if "content" in data:
                return self._create_content_chunk(data["content"])
            messages = data.get("messages")
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                if isinstance(last_msg, dict) and last_msg.get("type") == "ai":
                    return self._create_content_chunk(last_msg.get("content", ""))

        # Headerless list of message objects (no tuple string key)
        if isinstance(data, list) and event_type is None:
            # Attempt to treat as a messages payload directly
            return self._process_messages_event(data)

        return None

    def _process_messages_event(self, payload: Any) -> Optional[ModelResponseStream]:
        """
        Process a messages event from the stream.

        Handles multiple payload shapes emitted by LangGraph:

        * Flat list of message objects:
          ``[{message_obj}, {metadata}]``
        * Nested list (legacy ``messages-tuple`` second element):
          ``[[message_obj, metadata], ...]``
        * Single message dict
        """
        if isinstance(payload, dict):
            return self._extract_ai_content(payload)

        if isinstance(payload, list):
            for item in payload:
                # Nested list: [[msg, meta], ...]
                if isinstance(item, list) and len(item) >= 1:
                    result = self._extract_ai_content(item[0])
                    if result is not None:
                        return result
                # Flat list of dicts: [{msg}, {meta}]
                elif isinstance(item, dict):
                    result = self._extract_ai_content(item)
                    if result is not None:
                        return result

        return None

    def _extract_ai_content(self, msg: Any) -> Optional[ModelResponseStream]:
        """
        Return a content chunk if *msg* is an AI message dict with content.
        """
        if not isinstance(msg, dict):
            return None
        msg_type = msg.get("type", "")
        content = msg.get("content", "")
        if msg_type in ("ai", "AIMessageChunk") and content:
            return self._create_content_chunk(content)
        return None

    def _process_metadata_event(self, payload) -> Optional[ModelResponseStream]:
        """
        Process a metadata event, which may signal the end of the stream.
        """
        if isinstance(payload, dict):
            # Check if this is a final event
            if "run_id" in payload:
                self.finished = True
                return self._create_final_chunk()
        return None

    def _create_content_chunk(self, text: str) -> ModelResponseStream:
        """Create a ModelResponseStream chunk with content."""
        chunk = ModelResponseStream(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=0,
            model=self.model,
            object="chat.completion.chunk",
        )

        chunk.choices = [
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content=text, role="assistant"),
            )
        ]

        return chunk

    def _create_final_chunk(self) -> ModelResponseStream:
        """Create a final ModelResponseStream chunk with finish_reason."""
        chunk = ModelResponseStream(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=0,
            model=self.model,
            object="chat.completion.chunk",
        )

        chunk.choices = [
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(),
            )
        ]

        return chunk

    def __next__(self) -> ModelResponseStream:
        """Sync iteration - parse SSE events and yield ModelResponse chunks."""
        try:
            if self.line_iterator is None:
                raise StopIteration

            for line in self.line_iterator:
                result = self._parse_sse_line(line)
                if result is not None:
                    return result

            # Stream ended naturally - send final chunk if not already finished
            if not self.finished:
                self.finished = True
                return self._create_final_chunk()

            raise StopIteration

        except StopIteration:
            raise
        except httpx.StreamConsumed:
            raise StopIteration
        except httpx.StreamClosed:
            raise StopIteration
        except Exception as e:
            verbose_logger.error(f"Error in LangGraph SSE stream: {str(e)}")
            raise StopIteration

    async def __anext__(self) -> ModelResponseStream:
        """Async iteration - parse SSE events and yield ModelResponse chunks."""
        try:
            if self.async_line_iterator is None:
                raise StopAsyncIteration

            async for line in self.async_line_iterator:
                result = self._parse_sse_line(line)
                if result is not None:
                    return result

            # Stream ended naturally - send final chunk if not already finished
            if not self.finished:
                self.finished = True
                return self._create_final_chunk()

            raise StopAsyncIteration

        except StopAsyncIteration:
            raise
        except httpx.StreamConsumed:
            raise StopAsyncIteration
        except httpx.StreamClosed:
            raise StopAsyncIteration
        except Exception as e:
            verbose_logger.error(f"Error in LangGraph SSE stream: {str(e)}")
            raise StopAsyncIteration
