"""
SSE Stream Iterator for LangGraph.

Handles Server-Sent Events (SSE) streaming responses from LangGraph.
"""

import json
import uuid
from typing import TYPE_CHECKING, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

if TYPE_CHECKING:
    pass


class LangGraphSSEStreamIterator:
    """
    Iterator for LangGraph SSE streaming responses.
    Supports both sync and async iteration.

    LangGraph SSE wire format (stream_mode="messages-tuple"):

        event: messages
        data: [<AIMessageChunk>, <metadata_dict>]

        event: metadata
        data: {"run_id": "...", ...}

    The event type is delivered via the SSE ``event:`` header line.
    The ``data:`` payload for a ``messages`` event is a two-element array
    ``[message_object, metadata_object]``, NOT a tuple
    ``[event_type_string, payload]``.

    The previous implementation incorrectly treated ``data[0]`` as the event
    type, causing every real ``messages`` frame to be silently dropped because
    ``data[0]`` is a dict (the AI message), not the string ``"messages"``.
    """

    def __init__(self, response: httpx.Response, model: str):
        self.response = response
        self.model = model
        self.finished = False
        self.line_iterator = None
        self.async_line_iterator = None
        # Tracks the most recent ``event:`` field within the current SSE event block.
        self._current_event: Optional[str] = None

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
        Parse a single SSE line and return a ModelResponse chunk if applicable.

        Standard SSE multi-line format::

            event: messages
            data: [...]

        An empty line signals the end of an event block and resets
        ``_current_event`` to ``None``.
        """
        # SSE spec: an empty line dispatches the event; reset state.
        if not line.strip():
            self._current_event = None
            return None

        # Track the event type from the ``event:`` field.
        if line.startswith("event:"):
            self._current_event = line[6:].strip()
            return None

        # Handle SSE data lines.
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if not json_str:
                return None

            try:
                data = json.loads(json_str)
                return self._process_data(data, event_type=self._current_event)
            except json.JSONDecodeError:
                verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                return None

        return None

    def _process_data(
        self, data, event_type: Optional[str] = None
    ) -> Optional[ModelResponseStream]:
        """
        Process parsed data from an SSE ``data:`` line.

        Dispatch is based on *event_type* (the value of the preceding
        ``event:`` header), not on the contents of *data* itself.

        LangGraph ``messages`` event payload::

            [<AIMessageChunk dict>, <metadata dict>]

        LangGraph ``metadata`` event payload::

            {"run_id": "...", ...}
        """
        # --- Standard SSE protocol: event type from the ``event:`` header ---
        if event_type == "messages":
            # data = [message_object, metadata_object]
            if isinstance(data, list) and len(data) >= 1:
                return self._process_messages_payload(data[0])
            return None

        if event_type == "metadata":
            return self._process_metadata_event(data)

        # --- Fallback: legacy/non-standard tuple format [event_type, payload] ---
        # Retained for backward compatibility with any client that wraps the
        # data array as ["messages", payload] or ["metadata", payload].
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[0], str):
            legacy_event = data[0]
            payload = data[1]
            if legacy_event == "messages":
                return self._process_messages_event(payload)
            elif legacy_event == "metadata":
                return self._process_metadata_event(payload)

        # --- Dict format (alternative / non-streaming-style response) ---
        if isinstance(data, dict):
            if "content" in data:
                return self._create_content_chunk(data.get("content", ""))
            elif "messages" in data:
                messages = data.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict) and last_msg.get("type") == "ai":
                        return self._create_content_chunk(last_msg.get("content", ""))

        return None

    def _process_messages_payload(self, msg: object) -> Optional[ModelResponseStream]:
        """
        Extract content from a single LangGraph AIMessageChunk dict.

        This is the direct ``data[0]`` element from a ``messages`` SSE event.
        """
        if not isinstance(msg, dict):
            return None

        msg_type = msg.get("type", "")
        content = msg.get("content", "")

        if msg_type in ("ai", "AIMessageChunk") and content:
            return self._create_content_chunk(content)

        return None

    def _process_messages_event(self, payload) -> Optional[ModelResponseStream]:
        """
        Process a messages payload in the legacy tuple format.

        Legacy payload format: [[message_object, metadata], ...]
        """
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, list) and len(item) >= 1:
                    msg = item[0]
                    if isinstance(msg, dict):
                        msg_type = msg.get("type", "")
                        content = msg.get("content", "")

                        # Only return AI messages with content
                        if msg_type == "ai" and content:
                            return self._create_content_chunk(content)
                        elif msg_type == "AIMessageChunk" and content:
                            return self._create_content_chunk(content)
                elif isinstance(item, dict):
                    msg_type = item.get("type", "")
                    content = item.get("content", "")
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
