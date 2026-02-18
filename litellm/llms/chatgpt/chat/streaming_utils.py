"""
Streaming utilities for ChatGPT provider.

Normalizes non-spec-compliant tool_call chunks from the ChatGPT backend API.
"""

from typing import Any


class ChatGPTToolCallNormalizer:
    """
    Wraps a streaming response and fixes tool_call index/dedup issues.

    The ChatGPT backend API (chatgpt.com/backend-api) sends non-spec-compliant
    streaming tool call chunks:
    1. `index` is always 0, even for multiple parallel tool calls
    2. `id` and `name` get repeated in "closing" chunks that shouldn't exist

    This wrapper normalizes the stream to match the OpenAI spec before yielding
    chunks to the consumer.
    """

    def __init__(self, stream: Any):
        self._stream = stream
        self._seen_ids: dict[str, int] = {}  # tool_call_id -> assigned_index
        self._next_index: int = 0
        self._last_id: str | None = None  # tracks which tool call the next delta belongs to

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        while True:
            chunk = next(self._stream)
            result = self._normalize(chunk)
            if result is not None:
                return result

    async def __anext__(self):
        while True:
            chunk = await self._stream.__anext__()
            result = self._normalize(chunk)
            if result is not None:
                return result

    def _normalize(self, chunk: Any) -> Any:
        """Fix tool_calls in the chunk. Returns None to skip duplicate chunks."""
        if not chunk.choices:
            return chunk

        delta = chunk.choices[0].delta
        if delta is None or not delta.tool_calls:
            return chunk

        normalized = []
        for tc in delta.tool_calls:
            if tc.id and tc.id not in self._seen_ids:
                # New tool call — assign correct index
                self._seen_ids[tc.id] = self._next_index
                tc.index = self._next_index
                self._last_id = tc.id
                self._next_index += 1
                normalized.append(tc)
            elif tc.id and tc.id in self._seen_ids:
                # Duplicate "closing" chunk — skip it
                continue
            else:
                # Continuation delta (id=None) — fix index
                if self._last_id:
                    tc.index = self._seen_ids[self._last_id]
                normalized.append(tc)

        if not normalized:
            return None  # all tool_calls were duplicates, skip chunk

        delta.tool_calls = normalized
        return chunk
