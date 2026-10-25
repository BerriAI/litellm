"""
Streaming utilities for ChatGPT provider.

Normalizes non-spec-compliant tool_call chunks from the ChatGPT backend API.
"""

from collections.abc import Awaitable
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.types.utils import (
    ChatCompletionDeltaCustomToolCall,
    ChatCompletionDeltaToolCall,
    Delta,
    ModelResponseStream,
)


class ChatGPTChunkStream(Protocol):
    """A ChatGPT chunk source driven either synchronously or asynchronously."""

    def __next__(self) -> ModelResponseStream: ...

    def __anext__(self) -> Awaitable[ModelResponseStream]: ...


_DELEGATED_ATTRIBUTE: Final = TypeAdapter(object)


def _first_choice_delta(chunk: ModelResponseStream) -> Delta | None:
    return chunk.choices[0].delta


class ChatGPTToolCallNormalizer:
    """
    Wraps a streaming response and fixes tool_call index/dedup issues.

    The ChatGPT backend API (chatgpt.com/backend-api) can send tool_call chunks
    with:
    1. `index` values that are sparse or repeated across parallel tool calls
    2. `id` and `name` repeated in "closing" chunks that shouldn't exist

    This wrapper maps unique source indices to sequential output indices and
    uses the last identified call when repeated indices make a continuation
    ambiguous.
    """

    def __init__(self, stream: ChatGPTChunkStream):
        self._stream: Final = stream
        self._seen_ids: dict[str, int] = {}  # tool_call_id -> assigned_index
        self._next_index: int = 0
        self._source_index_to_output_index: dict[int, int | None] = {}  # mutable-ok: incremental per-stream state
        self._last_id: str | None = None  # tracks which tool call the next delta belongs to

    def __getattr__(self, name: str) -> object:
        return _DELEGATED_ATTRIBUTE.validate_python(getattr(self._stream, name))

    def __iter__(self) -> "ChatGPTToolCallNormalizer":
        return self

    def __aiter__(self) -> "ChatGPTToolCallNormalizer":
        return self

    def __next__(self) -> ModelResponseStream:
        while True:
            chunk = next(self._stream)
            result = self._normalize(chunk)
            if result is not None:
                return result

    async def __anext__(self) -> ModelResponseStream:
        while True:
            chunk = await self._stream.__anext__()
            result = self._normalize(chunk)
            if result is not None:
                return result

    def _normalize(self, chunk: ModelResponseStream) -> ModelResponseStream | None:
        """Fix tool_calls in the chunk. Returns None to skip duplicate chunks."""
        if not chunk.choices:
            return chunk

        delta: Final = _first_choice_delta(chunk)
        if delta is None or not delta.tool_calls:
            return chunk

        normalized: Final[list[ChatCompletionDeltaToolCall | ChatCompletionDeltaCustomToolCall]] = []
        for tc in delta.tool_calls:
            if tc.id and tc.id not in self._seen_ids:
                # New tool call — assign correct index
                if tc.index in self._source_index_to_output_index:
                    # Preserve the last-id fallback for repeated source indices.
                    self._source_index_to_output_index[tc.index] = None
                else:
                    self._source_index_to_output_index[tc.index] = self._next_index
                self._seen_ids[tc.id] = self._next_index
                tc.index = self._next_index
                self._last_id = tc.id
                self._next_index += 1
                normalized.append(tc)
            elif tc.id and tc.id in self._seen_ids:
                # Duplicate "closing" chunk — skip it
                continue
            else:
                # Continuation delta (id=None) — use its source index when unambiguous
                normalized_index: Final = self._source_index_to_output_index.get(tc.index)
                if normalized_index is not None:
                    tc.index = normalized_index
                elif self._last_id:
                    tc.index = self._seen_ids[self._last_id]
                normalized.append(tc)

        if not normalized:
            return None  # all tool_calls were duplicates, skip chunk

        delta.tool_calls = normalized
        return chunk
