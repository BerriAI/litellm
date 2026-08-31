"""
GigaChat Streaming Response Handler
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Final

from litellm.llms.gigachat.utils import convert_usage
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import ChatCompletionUsageBlock, GenericStreamingChunk


class GigaChatModelResponseIterator:
    """Iterator for GigaChat streaming responses."""

    def __init__(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: bool | None = False,
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.json_mode = json_mode

    def chunk_parser(self, chunk: Mapping[str, object]) -> GenericStreamingChunk:
        """Parse a single streaming chunk from GigaChat."""
        choices: Sequence = chunk.get("choices") or ()  # mutable-ok: tuple literal as default
        if not choices:
            return GenericStreamingChunk(
                text="",
                tool_use=None,
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
            )

        choice: Final = choices[0]
        delta: Mapping[str, object] = choice.get("delta") or {}  # mutable-ok: empty dict default for get
        chunk_finish_reason: Final = choice.get("finish_reason")

        # Extract text content
        text: Final = delta.get("content", "") or ""

        usage_block: ChatCompletionUsageBlock | None = None  # rebind-ok: conditionally assigned after stop detection
        tool_use: ChatCompletionToolCallChunk | None = None  # rebind-ok: conditionally assigned on function_call
        finish_reason: str | None = chunk_finish_reason

        # Handle function_call in stream
        raw_function_call: Final = delta.get("function_call")
        if chunk_finish_reason == "function_call" and isinstance(raw_function_call, Mapping) and raw_function_call:
            func_call: Final[Mapping[str, object]] = raw_function_call
            args_raw: Final[object] = func_call.get("arguments") or {}
            args_str: str  # rebind-ok: conditionally assigned from dict or str
            if isinstance(args_raw, dict):
                args_str = json.dumps(args_raw, ensure_ascii=False)  # rebind-ok: build from dict
            else:
                args_str = str(args_raw)

            name_raw: Final = func_call.get("name")
            tool_use = ChatCompletionToolCallChunk(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=ChatCompletionToolCallFunctionChunk(
                    name=name_raw if isinstance(name_raw, str) else "",
                    arguments=args_str,
                ),
                index=0,
            )
            finish_reason = "tool_calls"

        usage_data: Final = chunk.get("usage") or {}  # mutable-ok: empty dict default
        if usage_data and isinstance(usage_data, dict):
            validated_usage: Final = {k: int(v) for k, v in usage_data.items()}
            usage = convert_usage(validated_usage)
            _prompt_details: dict | None = (
                usage.prompt_tokens_details.model_dump() if usage.prompt_tokens_details else None
            )  # rebind-ok: conditional
            _completion_details: dict | None = (
                usage.completion_tokens_details.model_dump() if usage.completion_tokens_details else None
            )  # rebind-ok: conditional
            usage_block = ChatCompletionUsageBlock(  # pyright: ignore[reportCallIssue]  # TypedDict kwarg constructor
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                prompt_tokens_details=_prompt_details,
                completion_tokens_details=_completion_details,
            )

        return GenericStreamingChunk(
            text=str(text),
            tool_use=tool_use,
            is_finished=chunk_finish_reason is not None,
            finish_reason=finish_reason or "",
            usage=usage_block,
            index=choice.get("index", 0),
        )

    def __iter__(self):
        return self

    def __next__(self) -> GenericStreamingChunk:
        try:
            chunk = self.response_iterator.__next__()
            if isinstance(chunk, str):
                # Parse SSE format: data: {...}
                chunk = chunk.removeprefix("data: ")
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
                chunk = chunk.removeprefix("data: ")
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
