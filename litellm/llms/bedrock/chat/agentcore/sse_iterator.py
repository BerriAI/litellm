"""
SSE Stream Iterator for Bedrock AgentCore.

Handles Server-Sent Events (SSE) streaming responses from AgentCore.
"""

import json
from typing import TYPE_CHECKING, Any, Optional

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.bedrock_agentcore import AgentCoreUsage
from litellm.types.utils import Delta, ModelResponse, StreamingChoices, Usage

if TYPE_CHECKING:
    pass


class AgentCoreSSEStreamIterator:
    """
    Iterator for AgentCore SSE streaming responses.
    Supports both sync and async iteration.

    CRITICAL: The line iterators are created lazily on first access and reused.
    We must NOT create new iterators in __aiter__/__iter__ because
    CustomStreamWrapper calls __aiter__ on every call to its __anext__,
    which would create new iterators and cause StreamConsumed errors.
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

    def _parse_sse_line(self, line: str) -> Optional[ModelResponse]:
        """
        Parse a single SSE line and return a ModelResponse chunk if applicable.

        AgentCore SSE format:
        - data: {"event": {"contentBlockDelta": {"delta": {"text": "..."}}}}
        - data: {"event": {"metadata": {"usage": {...}}}}
        - data: {"message": {...}}
        """
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None

        json_str = line[5:].strip()
        if not json_str:
            return None

        try:
            data = json.loads(json_str)

            # Skip non-dict data (some lines contain Python repr strings)
            if not isinstance(data, dict):
                return None

            # Process content delta events
            if "event" in data and isinstance(data["event"], dict):
                event_payload = data["event"]
                content_block_delta = event_payload.get("contentBlockDelta")

                if content_block_delta:
                    delta = content_block_delta.get("delta", {})
                    text = delta.get("text", "")

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

                # Check for metadata/usage - this signals the end
                metadata = event_payload.get("metadata")
                if metadata and "usage" in metadata:
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

                    usage_data: AgentCoreUsage = metadata["usage"]  # type: ignore
                    setattr(
                        chunk,
                        "usage",
                        Usage(
                            prompt_tokens=usage_data.get("inputTokens", 0),
                            completion_tokens=usage_data.get("outputTokens", 0),
                            total_tokens=usage_data.get("totalTokens", 0),
                        ),
                    )

                    self.finished = True
                    return chunk

            # Check for final message (alternative finish signal)
            if "message" in data and isinstance(data["message"], dict):
                if not self.finished:
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

                    self.finished = True
                    return chunk

        except json.JSONDecodeError:
            verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")

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
        Sync iteration - parse SSE events and yield ModelResponse chunks.
        
        Uses next() on the stored iterator to properly resume between calls.
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

                result = self._parse_sse_line(line)
                if result is not None:
                    return result

        except StopIteration:
            raise
        except httpx.StreamConsumed:
            raise StopIteration
        except httpx.StreamClosed:
            raise StopIteration
        except Exception as e:
            verbose_logger.error(f"Error in AgentCore SSE stream: {str(e)}")
            raise StopIteration

    async def __anext__(self) -> ModelResponse:
        """
        Async iteration - parse SSE events and yield ModelResponse chunks.

        Uses __anext__() on the stored iterator to properly resume between calls.
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

                result = self._parse_sse_line(line)
                if result is not None:
                    return result

        except StopAsyncIteration:
            raise
        except httpx.StreamConsumed:
            raise StopAsyncIteration
        except httpx.StreamClosed:
            raise StopAsyncIteration
        except Exception as e:
            verbose_logger.error(f"Error in AgentCore SSE stream: {str(e)}")
            raise StopAsyncIteration
