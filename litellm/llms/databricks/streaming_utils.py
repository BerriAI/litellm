import json
from typing import Final

import litellm
from litellm import verbose_logger
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
from litellm.types.utils import GenericStreamingChunk, Usage


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool):
        self.streaming_response = streaming_response

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            processed_chunk: Final = litellm.ModelResponseStream(**chunk)

            text = ""
            tool_use: ChatCompletionToolCallChunk | None = None
            is_finished = False
            finish_reason = ""
            usage: ChatCompletionUsageBlock | None = None

            # Usage-only final chunk (OpenAI ``stream_options.include_usage``)
            # arrives with an empty ``choices`` list — return usage without
            # indexing ``choices[0]``.
            if len(processed_chunk.choices) == 0:
                final_usage: Final = getattr(processed_chunk, "usage", None)
                return GenericStreamingChunk(
                    text="",
                    tool_use=None,
                    is_finished=False,
                    finish_reason="",
                    usage=(
                        ChatCompletionUsageBlock(
                            prompt_tokens=final_usage.prompt_tokens or 0,
                            completion_tokens=final_usage.completion_tokens or 0,
                            total_tokens=final_usage.total_tokens or 0,
                        )
                        if final_usage is not None
                        else None
                    ),
                    index=0,
                )

            if processed_chunk.choices[0].delta.content is not None:
                text = processed_chunk.choices[0].delta.content

            if (
                processed_chunk.choices[0].delta.tool_calls is not None
                and len(processed_chunk.choices[0].delta.tool_calls) > 0
                and processed_chunk.choices[0].delta.tool_calls[0].function is not None
                and processed_chunk.choices[0].delta.tool_calls[0].function.arguments is not None
            ):
                tool_use = ChatCompletionToolCallChunk(
                    id=processed_chunk.choices[0].delta.tool_calls[0].id,
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=processed_chunk.choices[0].delta.tool_calls[0].function.name,
                        arguments=processed_chunk.choices[0].delta.tool_calls[0].function.arguments,
                    ),
                    index=processed_chunk.choices[0].delta.tool_calls[0].index,
                )

            if processed_chunk.choices[0].finish_reason is not None:
                is_finished = True
                finish_reason = processed_chunk.choices[0].finish_reason

            usage_chunk: Final[Usage | None] = getattr(processed_chunk, "usage", None)
            if usage_chunk is not None:
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=usage_chunk.prompt_tokens,
                    completion_tokens=usage_chunk.completion_tokens,
                    total_tokens=usage_chunk.total_tokens,
                )

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
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
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
            chunk = chunk.strip()
            if len(chunk) > 0:
                json_chunk: Final = json.loads(chunk)
                return self.chunk_parser(chunk=json_chunk)
            else:
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")
        except Exception as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
            chunk = chunk.strip()
            if chunk == "[DONE]":
                raise StopAsyncIteration
            if len(chunk) > 0:
                json_chunk: Final = json.loads(chunk)
                return self.chunk_parser(chunk=json_chunk)
            else:
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )
