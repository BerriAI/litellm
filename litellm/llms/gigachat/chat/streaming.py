"""
GigaChat Streaming Response Handler
"""

import json
import uuid
from typing import Any, Optional

from litellm.types.llms.openai import ChatCompletionToolCallChunk, ChatCompletionToolCallFunctionChunk
from litellm.types.utils import GenericStreamingChunk


class GigaChatModelResponseIterator:
    """Iterator for GigaChat streaming responses."""

    def __init__(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.json_mode = json_mode

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        """Parse a single streaming chunk from GigaChat."""
        text = ""
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        is_finished = False
        finish_reason: Optional[str] = None

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

        # Extract text content
        text = delta.get("content", "") or ""

        # Handle function_call in stream
        if finish_reason == "function_call" and delta.get("function_call"):
            func_call = delta["function_call"]
            args = func_call.get("arguments", {})

            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)

            tool_use = ChatCompletionToolCallChunk(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=ChatCompletionToolCallFunctionChunk(
                    name=func_call.get("name", ""),
                    arguments=args,
                ),
                index=0,
            )
            finish_reason = "tool_calls"

        if finish_reason is not None:
            is_finished = True

        return GenericStreamingChunk(
            text=text,
            tool_use=tool_use,
            is_finished=is_finished,
            finish_reason=finish_reason or "",
            usage=None,
            index=choice.get("index", 0),
        )

    def __iter__(self):
        return self

    def __next__(self) -> GenericStreamingChunk:
        try:
            chunk = self.response_iterator.__next__()
            if isinstance(chunk, str):
                # Parse SSE format: data: {...}
                if chunk.startswith("data: "):
                    chunk = chunk[6:]
                if chunk.strip() == "[DONE]":
                    raise StopIteration
                try:
                    chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    return GenericStreamingChunk(
                        text="",
                        tool_use=None,
                        is_finished=False,
                        finish_reason="",
                        usage=None,
                        index=0,
                    )
            return self.chunk_parser(chunk)
        except StopIteration:
            raise

    def __aiter__(self):
        return self

    async def __anext__(self) -> GenericStreamingChunk:
        try:
            chunk = await self.response_iterator.__anext__()
            if isinstance(chunk, str):
                # Parse SSE format
                if chunk.startswith("data: "):
                    chunk = chunk[6:]
                if chunk.strip() == "[DONE]":
                    raise StopAsyncIteration
                try:
                    chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    return GenericStreamingChunk(
                        text="",
                        tool_use=None,
                        is_finished=False,
                        finish_reason="",
                        usage=None,
                        index=0,
                    )
            return self.chunk_parser(chunk)
        except StopAsyncIteration:
            raise
