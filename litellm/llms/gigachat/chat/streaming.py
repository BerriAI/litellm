"""
GigaChat Streaming Response Handler

Based on official GigaChat SDK streaming implementation.
Supports:
- SSE (Server-Sent Events) parsing
- Function calls in streaming mode
- reasoning_content for reasoning models
"""

import json
import uuid
from typing import Any, Optional

from litellm.types.llms.openai import ChatCompletionToolCallChunk, ChatCompletionToolCallFunctionChunk
from litellm.types.utils import GenericStreamingChunk


class GigaChatModelResponseIterator:
    """
    Iterator for GigaChat streaming responses.

    Parses SSE events and converts GigaChat-specific format to OpenAI format.

    GigaChat SSE format:
        data: {"choices": [{"delta": {"content": "...", "reasoning_content": "..."}, ...}]}

    Converted to GenericStreamingChunk for litellm compatibility.
    """

    def __init__(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.json_mode = json_mode
        self._is_done = False

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        """
        Parse a single streaming chunk from GigaChat.

        GigaChat chunk structure (from models/chat.py ChatCompletionChunk):
        {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": "...",
                    "reasoning_content": "...",  # optional
                    "function_call": {...},  # optional
                    "functions_state_id": "..."  # optional
                },
                "index": 0,
                "finish_reason": null | "stop" | "function_call"
            }],
            "created": 1234567890,
            "model": "GigaChat",
            "object": "chat.completion.chunk",
            "usage": null | {...}  # optional, may appear in final chunk
        }
        """
        text = ""
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        is_finished = False
        finish_reason: Optional[str] = None
        reasoning_content: Optional[str] = None

        choices = chunk.get("choices", [])
        if not choices:
            return GenericStreamingChunk(
                text="",
                tool_use=None,
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
            )

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        index = choice.get("index", 0)

        # Extract text content
        text = delta.get("content", "") or ""

        # Extract reasoning_content if present (for reasoning models)
        reasoning_content = delta.get("reasoning_content")

        # Handle function_call in stream
        if delta.get("function_call"):
            func_call = delta["function_call"]
            args = func_call.get("arguments", {})

            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            elif args is None:
                args = ""

            tool_use = ChatCompletionToolCallChunk(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=ChatCompletionToolCallFunctionChunk(
                    name=func_call.get("name", ""),
                    arguments=args,
                ),
                index=0,
            )

            # If finish_reason is function_call, convert to tool_calls
            if finish_reason == "function_call":
                finish_reason = "tool_calls"

        if finish_reason is not None:
            is_finished = True

        result = GenericStreamingChunk(
            text=text,
            tool_use=tool_use,
            is_finished=is_finished,
            finish_reason=finish_reason or "",
            usage=None,
            index=index,
        )

        # Add reasoning_content to result if present
        if reasoning_content:
            # Using setattr since GenericStreamingChunk may not have this field
            setattr(result, "reasoning_content", reasoning_content)

        return result

    def _parse_sse_line(self, line: str) -> Optional[dict]:
        """
        Parse a single SSE line.

        SSE format:
            data: {...json...}
            data: [DONE]

        Returns:
            Parsed JSON dict or None if line should be skipped
        """
        if not line:
            return None

        # Handle "data: " prefix
        if line.startswith("data:"):
            data = line[5:].strip()
        elif line.startswith("data "):
            data = line[5:].strip()
        else:
            # Not a data line, skip
            return None

        # Check for end marker
        if data == "[DONE]":
            self._is_done = True
            return None

        # Parse JSON
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # Log but don't fail - might be partial data
            return None

    def __iter__(self):
        return self

    def __next__(self) -> GenericStreamingChunk:
        if self._is_done:
            raise StopIteration

        try:
            chunk = self.response_iterator.__next__()

            # Handle string chunks (SSE format)
            if isinstance(chunk, str):
                parsed = self._parse_sse_line(chunk)
                if parsed is None:
                    # Skip empty/done lines, try next
                    return self.__next__()
                chunk = parsed

            # Handle dict chunks (already parsed)
            if isinstance(chunk, dict):
                return self.chunk_parser(chunk)

            # Handle bytes
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", errors="replace")
                parsed = self._parse_sse_line(chunk_str)
                if parsed is None:
                    return self.__next__()
                return self.chunk_parser(parsed)

            # Unknown type, return empty chunk
            return GenericStreamingChunk(
                text="",
                tool_use=None,
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
            )

        except StopIteration:
            raise

    def __aiter__(self):
        return self

    async def __anext__(self) -> GenericStreamingChunk:
        if self._is_done:
            raise StopAsyncIteration

        try:
            chunk = await self.response_iterator.__anext__()

            # Handle string chunks (SSE format)
            if isinstance(chunk, str):
                parsed = self._parse_sse_line(chunk)
                if parsed is None:
                    # Skip empty/done lines, try next
                    return await self.__anext__()
                chunk = parsed

            # Handle dict chunks (already parsed)
            if isinstance(chunk, dict):
                return self.chunk_parser(chunk)

            # Handle bytes
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", errors="replace")
                parsed = self._parse_sse_line(chunk_str)
                if parsed is None:
                    return await self.__anext__()
                return self.chunk_parser(parsed)

            # Unknown type, return empty chunk
            return GenericStreamingChunk(
                text="",
                tool_use=None,
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
            )

        except StopAsyncIteration:
            raise
