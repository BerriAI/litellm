import json
from collections.abc import Sequence
from typing import Final

import litellm
from litellm import verbose_logger
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
from litellm.types.utils import GenericStreamingChunk, Usage


def _usage_block(usage: Usage) -> ChatCompletionUsageBlock:
    return ChatCompletionUsageBlock(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
        prompt_tokens_details=usage.prompt_tokens_details.model_dump() if usage.prompt_tokens_details else None,
        completion_tokens_details=(
            usage.completion_tokens_details.model_dump() if usage.completion_tokens_details else None
        ),
    )


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool):
        self.streaming_response = streaming_response

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            processed_chunk: Final = litellm.ModelResponseStream(**chunk)

            # Usage-only final chunk (OpenAI ``stream_options.include_usage``)
            # arrives with an empty ``choices`` list — return usage without
            # indexing ``choices[0]``.
            if len(processed_chunk.choices) == 0:
                final_usage: Final[Usage | None] = getattr(processed_chunk, "usage", None)
                return GenericStreamingChunk(
                    text="",
                    tool_use=None,
                    is_finished=False,
                    finish_reason="",
                    usage=_usage_block(final_usage) if final_usage is not None else None,
                    index=0,
                )

            choice_0: Final = processed_chunk.choices[0]
            choice_delta: Final = choice_0.delta
            text: Final[str] = choice_delta.content if choice_delta.content is not None else ""
            reasoning_content: Final[str | None] = getattr(choice_delta, "reasoning_content", None)

            tool_calls_list: list[ChatCompletionToolCallChunk] = []  # mutable-ok: accumulates delta calls
            if choice_delta.tool_calls is not None:
                for tc in choice_delta.tool_calls:
                    tc_fn = getattr(tc, "function", None)
                    fn_chunk = (
                        ChatCompletionToolCallFunctionChunk(
                            name=getattr(tc_fn, "name", None),
                            arguments=getattr(tc_fn, "arguments", "") or "",
                        )
                        if tc_fn is not None
                        else ChatCompletionToolCallFunctionChunk(name=None, arguments="")
                    )
                    call_chunk = ChatCompletionToolCallChunk(
                        id=getattr(tc, "id", None),
                        type="function",
                        function=fn_chunk,
                        index=getattr(tc, "index", 0) if getattr(tc, "index", None) is not None else 0,
                    )
                    tool_calls_list.append(call_chunk)

            tool_calls: Final[Sequence[ChatCompletionToolCallChunk] | None] = tool_calls_list or None
            tool_use: Final[ChatCompletionToolCallChunk | None] = tool_calls_list[0] if tool_calls_list else None

            is_finished: Final[bool] = choice_0.finish_reason is not None
            finish_reason: Final[str] = choice_0.finish_reason or ""

            usage_chunk: Final[Usage | None] = getattr(processed_chunk, "usage", None)
            usage: Final[ChatCompletionUsageBlock | None] = (
                _usage_block(usage_chunk) if usage_chunk is not None else None
            )

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=0,
            )
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        self.response_iterator = self.streaming_response
        return self

    def __next__(self):
        if not hasattr(self, "response_iterator"):
            self.response_iterator = self.streaming_response
        chunk: Final = self._next_sse_data()
        try:
            json_chunk: Final = json.loads(chunk)
            return self.chunk_parser(chunk=json_chunk)
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return _empty_generic_chunk()

    def _next_sse_data(self) -> str:
        try:
            for raw_chunk in self.response_iterator:
                if sse_data := _sse_data(raw_chunk):
                    return sse_data
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")
        raise StopIteration

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        chunk: Final = await self._anext_sse_data()
        if chunk == "[DONE]":
            raise StopAsyncIteration
        try:
            json_chunk: Final = json.loads(chunk)
            return self.chunk_parser(chunk=json_chunk)
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return _empty_generic_chunk()

    async def _anext_sse_data(self) -> str:
        try:
            async for raw_chunk in self.async_response_iterator:
                if sse_data := _sse_data(raw_chunk):
                    return sse_data
        except Exception as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")
        raise StopAsyncIteration


def _sse_data(raw_chunk: str) -> str:
    return (litellm.CustomStreamWrapper.strip_sse_data_from_chunk(raw_chunk) or "").strip()


def _empty_generic_chunk() -> GenericStreamingChunk:
    return GenericStreamingChunk(
        text="",
        is_finished=False,
        finish_reason="",
        usage=None,
        index=0,
        tool_use=None,
    )
