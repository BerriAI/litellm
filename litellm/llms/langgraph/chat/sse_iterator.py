"""
SSE Stream Iterator for LangGraph.

Handles Server-Sent Events (SSE) streaming responses from LangGraph.
"""

import json
import uuid
from typing import TYPE_CHECKING, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.types.utils import Delta, ModelResponse, StreamingChoices

if TYPE_CHECKING:
    pass


class LangGraphSSEStreamIterator:
    """
    Iterator for LangGraph SSE streaming responses.
    Supports both sync and async iteration.

    LangGraph stream format with stream_mode="messages-tuple":
    Each SSE event is a tuple: (event_type, data)
    Common event types: "messages", "metadata"
    """

    def __init__(self, response: httpx.Response, model: str):
        self.response = response
        self.model = model
        self.finished = False
        self.line_iterator = None
        self.async_line_iterator = None

    def __iter__(self):
        """Initialize sync iteration."""
        self.line_iterator = self.response.iter_lines()
        return self

    def __aiter__(self):
        """Initialize async iteration."""
        self.async_line_iterator = self.response.aiter_lines()
        return self

    def _parse_sse_line(self, line: str) -> Optional[ModelResponse]:
        """
        Parse a single SSE line and return a ModelResponse chunk if applicable.

        LangGraph SSE format can vary:
        - data: [...] (tuple format)
        - event: ...\ndata: ...
        """
        line = line.strip()
        if not line:
            return None

        # Handle SSE data lines
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if not json_str:
                return None

            try:
                data = json.loads(json_str)
                return self._process_data(data)
            except json.JSONDecodeError:
                verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                return None

        return None

    def _process_data(self, data) -> Optional[ModelResponse]:
        """
        Process parsed data from SSE stream.

        LangGraph uses tuple format: [event_type, payload]
        """
        # Handle tuple format: ["messages", ...]
        if isinstance(data, list) and len(data) >= 2:
            event_type = data[0]
            payload = data[1]

            if event_type == "messages":
                return self._process_messages_event(payload)
            elif event_type == "metadata":
                # Metadata event, might contain usage info
                return self._process_metadata_event(payload)

        # Handle dict format (alternative response format)
        elif isinstance(data, dict):
            if "content" in data:
                return self._create_content_chunk(data.get("content", ""))
            elif "messages" in data:
                messages = data.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict) and last_msg.get("type") == "ai":
                        return self._create_content_chunk(last_msg.get("content", ""))

        return None

    def _process_messages_event(self, payload) -> Optional[ModelResponse]:
        """
        Process a messages event from the stream.

        payload format: [[message_object, metadata], ...]
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

    def _process_metadata_event(self, payload) -> Optional[ModelResponse]:
        """
        Process a metadata event, which may signal the end of the stream.
        """
        if isinstance(payload, dict):
            # Check if this is a final event
            if "run_id" in payload:
                self.finished = True
                return self._create_final_chunk()
        return None

    def _create_content_chunk(self, text: str) -> ModelResponse:
        """Create a ModelResponse chunk with content."""
        chunk = ModelResponse(
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

    def _create_final_chunk(self) -> ModelResponse:
        """Create a final ModelResponse chunk with finish_reason."""
        chunk = ModelResponse(
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

    def __next__(self) -> ModelResponse:
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

    async def __anext__(self) -> ModelResponse:
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

