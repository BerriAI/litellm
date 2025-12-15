"""
SSE Stream Iterator for Vertex AI Agent Engine.

Handles Server-Sent Events (SSE) streaming responses from Vertex AI Reasoning Engines.
"""

import json
from typing import TYPE_CHECKING, Any, Optional

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.types.utils import Delta, ModelResponse, StreamingChoices, Usage

if TYPE_CHECKING:
    pass


class VertexAgentEngineSSEStreamIterator:
    """
    Iterator for Vertex Agent Engine SSE streaming responses.
    Supports both sync and async iteration.

    The response format from Vertex Agent Engine is line-delimited JSON,
    where each line contains a response chunk with content.parts[].text
    """

    def __init__(self, response: httpx.Response, model: str):
        self.response = response
        self.model = model
        self.finished = False
        self._sync_iter: Any = None
        self._async_iter: Any = None
        self._sync_iter_initialized = False
        self._async_iter_initialized = False

    def __iter__(self):
        """Initialize sync iteration - create iterator lazily on first call only."""
        if not self._sync_iter_initialized:
            self._sync_iter = iter(self.response.iter_lines())
            self._sync_iter_initialized = True
        return self

    def __aiter__(self):
        """Initialize async iteration - create iterator lazily on first call only."""
        if not self._async_iter_initialized:
            self._async_iter = self.response.aiter_lines().__aiter__()
            self._async_iter_initialized = True
        return self

    def _extract_text_from_chunk(self, data: dict) -> Optional[str]:
        """
        Extract text content from a response chunk.

        Vertex Agent Engine response format:
        {
            "content": {
                "parts": [{"text": "..."}],
                "role": "model"
            },
            "actions": {...}
        }
        """
        # Try content.parts
        content = data.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                return part["text"]

        return None

    def _parse_line(self, line: str) -> Optional[ModelResponse]:
        """
        Parse a single line and return a ModelResponse chunk if applicable.
        """
        line = line.strip()
        if not line:
            return None

        try:
            data = json.loads(line)

            # Skip non-dict data
            if not isinstance(data, dict):
                return None

            # Extract text from the chunk
            text = self._extract_text_from_chunk(data)

            if text:
                # Return chunk with text
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

        except json.JSONDecodeError:
            verbose_logger.debug(f"Skipping non-JSON line: {line[:100]}")

        return None

    def _create_final_chunk(self) -> ModelResponse:
        """Create a final chunk to signal stream completion."""
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
        """
        Sync iteration - parse lines and yield ModelResponse chunks.
        """
        try:
            if self._sync_iter is None:
                raise StopIteration

            # Keep getting lines until we have a result to return
            while True:
                try:
                    line = next(self._sync_iter)
                except StopIteration:
                    # Stream ended - send final chunk if not already finished
                    if not self.finished:
                        self.finished = True
                        return self._create_final_chunk()
                    raise

                result = self._parse_line(line)
                if result is not None:
                    return result

        except StopIteration:
            raise
        except httpx.StreamConsumed:
            raise StopIteration
        except httpx.StreamClosed:
            raise StopIteration
        except Exception as e:
            verbose_logger.error(f"Error in Vertex Agent Engine SSE stream: {str(e)}")
            raise StopIteration

    async def __anext__(self) -> ModelResponse:
        """
        Async iteration - parse lines and yield ModelResponse chunks.
        """
        try:
            if self._async_iter is None:
                raise StopAsyncIteration

            # Keep getting lines until we have a result to return
            while True:
                try:
                    line = await self._async_iter.__anext__()
                except StopAsyncIteration:
                    # Stream ended - send final chunk if not already finished
                    if not self.finished:
                        self.finished = True
                        return self._create_final_chunk()
                    raise

                result = self._parse_line(line)
                if result is not None:
                    return result

        except StopAsyncIteration:
            raise
        except httpx.StreamConsumed:
            raise StopAsyncIteration
        except httpx.StreamClosed:
            raise StopAsyncIteration
        except Exception as e:
            verbose_logger.error(f"Error in Vertex Agent Engine SSE stream: {str(e)}")
            raise StopAsyncIteration

