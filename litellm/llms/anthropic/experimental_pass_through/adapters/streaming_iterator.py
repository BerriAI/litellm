# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import copy
import json
import traceback
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
)

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.anthropic import (
    ContentBlockContentBlockDict,
    TextBlock,
    UsageDelta,
)
from litellm.types.utils import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


class _CombinedChunkSplitter:
    """
    Splits a streaming chunk that carries BOTH response content and a
    ``finish_reason`` into two chunks: a content-only chunk followed by a
    finish-only chunk.

    ``AnthropicStreamWrapper`` (via ``translate_streaming_openai_response_to_anthropic``)
    assumes content and ``finish_reason`` never arrive in the same chunk — true for
    real provider streams, but false for fake-streamed providers (e.g. Vertex AI
    Gemma ``:predict``) where ``MockResponseIterator`` collapses the entire response
    into a single chunk. Without this split the assumption causes all content to be
    silently dropped (only the ``message_delta`` stop event is emitted).

    Supports both sync and async iteration, since ``AnthropicStreamWrapper`` exposes
    both ``__next__`` and ``__anext__``.
    """

    def __init__(self, completion_stream: Any):
        self._stream = completion_stream
        self._sync_iter: Optional[Iterator[Any]] = None
        self._async_iter: Optional[AsyncIterator[Any]] = None
        self._buffer: deque = deque()

    @staticmethod
    def _is_combined(chunk: Any) -> bool:
        """True if ``chunk`` carries response content AND a finish_reason."""
        choices = getattr(chunk, "choices", None)
        if not choices:
            return False
        choice = choices[0]
        if getattr(choice, "finish_reason", None) is None:
            return False
        delta = getattr(choice, "delta", None)
        if delta is None:
            return False
        return bool(
            getattr(delta, "content", None)
            or getattr(delta, "tool_calls", None)
            or getattr(delta, "reasoning_content", None)
            or getattr(delta, "thinking_blocks", None)
        )

    @staticmethod
    def _split(chunk: Any) -> List[Any]:
        """Return ``[chunk]``, or ``[content_chunk, finish_chunk]`` if combined."""
        if not _CombinedChunkSplitter._is_combined(chunk):
            return [chunk]

        # Content chunk: keep the delta payload, clear the finish_reason.
        content_chunk = copy.deepcopy(chunk)
        content_chunk.choices[0].finish_reason = None

        # Finish chunk: keep finish_reason (and usage), clear the delta payload.
        finish_chunk = copy.deepcopy(chunk)
        finish_delta = finish_chunk.choices[0].delta
        finish_delta.content = None
        if hasattr(finish_delta, "tool_calls"):
            finish_delta.tool_calls = None
        if hasattr(finish_delta, "reasoning_content"):
            finish_delta.reasoning_content = None
        if hasattr(finish_delta, "thinking_blocks"):
            finish_delta.thinking_blocks = None
        return [content_chunk, finish_chunk]

    def __iter__(self) -> "Iterator[Any]":
        return self

    def __next__(self) -> Any:
        if self._buffer:
            return self._buffer.popleft()
        if self._sync_iter is None:
            self._sync_iter = iter(self._stream)
        chunk = next(self._sync_iter)  # propagates StopIteration when exhausted
        self._buffer.extend(self._split(chunk))
        return self._buffer.popleft()

    def __aiter__(self) -> "AsyncIterator[Any]":
        return self

    async def __anext__(self) -> Any:
        if self._buffer:
            return self._buffer.popleft()
        if self._async_iter is None:
            self._async_iter = self._stream.__aiter__()
        chunk = await self._async_iter.__anext__()  # propagates StopAsyncIteration
        self._buffer.extend(self._split(chunk))
        return self._buffer.popleft()


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    def __init__(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ):
        # Wrap the upstream stream so chunks that carry both content and a
        # finish_reason (fake-streamed providers) are split into two — see
        # _CombinedChunkSplitter.
        super().__init__(_CombinedChunkSplitter(completion_stream))
        self.model = model
        # Mapping of truncated tool names to original names (for OpenAI's 64-char limit)
        self.tool_name_mapping = tool_name_mapping or {}
        # Per-request streaming state — MUST be instance attributes. A
        # class-level mutable (the old ``chunk_queue = deque()``) is shared
        # across instances and leaks chunks between concurrent requests.
        self.sent_first_chunk: bool = False
        self.sent_content_block_start: bool = False
        self.sent_content_block_finish: bool = False
        self.current_content_block_type: Literal["text", "tool_use", "thinking"] = (
            "text"
        )
        self.sent_last_message: bool = False
        self.holding_chunk: Optional[Any] = None
        self.holding_stop_reason_chunk: Optional[Any] = None
        self.queued_usage_chunk: bool = False
        self.current_content_block_index: int = 0
        self.current_content_block_start: ContentBlockContentBlockDict = TextBlock(
            type="text",
            text="",
        )
        self.chunk_queue: deque = deque()  # buffers multiple chunks

    def _create_initial_usage_delta(self) -> UsageDelta:
        """
        Create the initial UsageDelta for the message_start event.

        Initializes cache token fields (cache_creation_input_tokens, cache_read_input_tokens)
        to 0 to indicate to clients (like Claude Code) that prompt caching is supported.

        The actual cache token values will be provided in the message_delta event at the
        end of the stream, since Bedrock Converse API only returns usage data in the final
        response chunk.

        Returns:
            UsageDelta with all token counts initialized to 0.
        """
        return UsageDelta(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def __next__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                self.chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": self.current_content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                return self.chunk_queue.popleft()

            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                should_start_new_block = self._should_start_new_content_block(chunk)
                if should_start_new_block:
                    self._increment_content_block_index()

                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                if should_start_new_block and not self.sent_content_block_finish:
                    # Queue the sequence: content_block_stop -> content_block_start
                    # For text blocks the trigger chunk is not emitted as a separate
                    # delta because content_block_start carries the information.
                    # For tool_use blocks we must also emit the trigger chunk's delta
                    # when it carries input_json_delta data, because some providers
                    # (e.g. xAI, Gemini) include tool arguments in the same streaming
                    # chunk as the function name/id.

                    # 1. Stop current content block
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": max(self.current_content_block_index - 1, 0),
                        }
                    )

                    # 2. Start new content block
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )

                    # 3. If the trigger chunk carries tool argument data, queue it
                    # so the input_json_delta is not silently dropped.
                    if (
                        processed_chunk.get("type") == "content_block_delta"
                        and isinstance(processed_chunk.get("delta"), dict)
                        and processed_chunk["delta"].get("type") == "input_json_delta"
                        and processed_chunk["delta"].get("partial_json")
                    ):
                        self.chunk_queue.append(processed_chunk)

                    self.sent_content_block_finish = False
                    return self.chunk_queue.popleft()

                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    # Queue both the content_block_stop and the message_delta
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        }
                    )
                    self.sent_content_block_finish = True
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()
                elif self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.chunk_queue.append(processed_chunk)
                    self.holding_chunk = None
                    return self.chunk_queue.popleft()
                else:
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if self.holding_chunk is not None:
                self.chunk_queue.append(self.holding_chunk)
                self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration
        except StopIteration:
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            if self.sent_last_message is False:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopIteration
        except Exception as e:
            verbose_logger.error(
                "Anthropic Adapter - {}\n{}".format(e, traceback.format_exc())
            )
            raise StopAsyncIteration

    async def __anext__(self):  # noqa: PLR0915
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                self.chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": self.current_content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                return self.chunk_queue.popleft()

            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                # Check if we need to start a new content block
                should_start_new_block = self._should_start_new_content_block(chunk)
                if should_start_new_block:
                    self._increment_content_block_index()

                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if (
                    self.holding_stop_reason_chunk is not None
                    and getattr(chunk, "usage", None) is not None
                ):
                    # Merge usage into the held stop_reason chunk
                    merged_chunk = self.holding_stop_reason_chunk.copy()
                    if "delta" not in merged_chunk:
                        merged_chunk["delta"] = {}

                    # Add usage to the held chunk
                    uncached_input_tokens = chunk.usage.prompt_tokens or 0
                    if (
                        hasattr(chunk.usage, "prompt_tokens_details")
                        and chunk.usage.prompt_tokens_details
                    ):
                        cached_tokens = (
                            getattr(
                                chunk.usage.prompt_tokens_details, "cached_tokens", 0
                            )
                            or 0
                        )
                        uncached_input_tokens -= cached_tokens

                    usage_dict: UsageDelta = {
                        "input_tokens": uncached_input_tokens,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
                    # Add cache tokens if available (for prompt caching support)
                    if (
                        hasattr(chunk.usage, "_cache_creation_input_tokens")
                        and chunk.usage._cache_creation_input_tokens > 0
                    ):
                        usage_dict["cache_creation_input_tokens"] = (
                            chunk.usage._cache_creation_input_tokens
                        )
                    if (
                        hasattr(chunk.usage, "_cache_read_input_tokens")
                        and chunk.usage._cache_read_input_tokens > 0
                    ):
                        usage_dict["cache_read_input_tokens"] = (
                            chunk.usage._cache_read_input_tokens
                        )
                    merged_chunk["usage"] = usage_dict

                    # Queue the merged chunk and reset
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                # Check if this processed chunk has a stop_reason - hold it for next chunk

                if not self.queued_usage_chunk:
                    if should_start_new_block and not self.sent_content_block_finish:
                        # Queue the sequence: content_block_stop -> content_block_start
                        # For text blocks the trigger chunk is not emitted as a separate
                        # delta because content_block_start carries the information.
                        # For tool_use blocks we must also emit the trigger chunk's delta
                        # when it carries input_json_delta data, because some providers
                        # (e.g. xAI, Gemini) include tool arguments in the same streaming
                        # chunk as the function name/id.

                        # 1. Stop current content block
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": max(self.current_content_block_index - 1, 0),
                            }
                        )
                        self.chunk_queue.append(
                            {
                                "type": "content_block_start",
                                "index": self.current_content_block_index,
                                "content_block": self.current_content_block_start,
                            }
                        )

                        # 3. If the trigger chunk carries tool argument data, queue it
                        # so the input_json_delta is not silently dropped.
                        if (
                            processed_chunk.get("type") == "content_block_delta"
                            and isinstance(processed_chunk.get("delta"), dict)
                            and processed_chunk["delta"].get("type")
                            == "input_json_delta"
                            and processed_chunk["delta"].get("partial_json")
                        ):
                            self.chunk_queue.append(processed_chunk)

                        # Reset state for new block
                        self.sent_content_block_finish = False
                        return self.chunk_queue.popleft()

                    if (
                        processed_chunk["type"] == "message_delta"
                        and self.sent_content_block_finish is False
                    ):
                        # Queue both the content_block_stop and the holding chunk
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            }
                        )
                        self.sent_content_block_finish = True
                        if (
                            processed_chunk.get("delta", {}).get("stop_reason")
                            is not None
                        ):
                            self.holding_stop_reason_chunk = processed_chunk
                        else:
                            self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()
                    elif self.holding_chunk is not None:
                        # Queue both chunks
                        self.chunk_queue.append(self.holding_chunk)
                        self.chunk_queue.append(processed_chunk)
                        self.holding_chunk = None
                        return self.chunk_queue.popleft()
                    else:
                        # Queue the current chunk
                        self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if not self.queued_usage_chunk:
                if self.holding_stop_reason_chunk is not None:
                    self.chunk_queue.append(self.holding_stop_reason_chunk)
                    self.holding_stop_reason_chunk = None

                if self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            # Return queued items if any
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration

        except StopIteration:
            # Handle any remaining queued chunks before stopping
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            # Handle any held stop_reason chunk
            if self.holding_stop_reason_chunk is not None:
                return self.holding_stop_reason_chunk
            if not self.sent_last_message:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopAsyncIteration

    def anthropic_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        Similar to the Bedrock bedrock_sse_wrapper implementation.

        This wrapper ensures dict chunks are SSE formatted with both event and data lines.
        """
        for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of anthropic_sse_wrapper.
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        """
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    def _increment_content_block_index(self):
        self.current_content_block_index += 1

    def _should_start_new_content_block(self, chunk: "ModelResponseStream") -> bool:
        """
        Determine if we should start a new content block based on the processed chunk.
        Override this method with your specific logic for detecting new content blocks.

        Examples of when you might want to start a new content block:
        - Switching from text to tool calls
        - Different content types in the response
        - Specific markers in the content
        """
        from .transformation import LiteLLMAnthropicMessagesAdapter

        # Example logic - customize based on your needs:
        # If chunk indicates a tool call
        if chunk.choices[0].finish_reason is not None:
            return False

        (
            block_type,
            content_block_start,
        ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices  # type: ignore
        )

        # Restore original tool name if it was truncated for OpenAI's 64-char limit
        if block_type == "tool_use":
            # Type narrowing: content_block_start is ToolUseBlock when block_type is "tool_use"
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)

            if tool_block.get("name"):
                truncated_name = tool_block["name"]
                original_name = self.tool_name_mapping.get(
                    truncated_name, truncated_name
                )
                tool_block["name"] = original_name

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        # For parallel tool calls, we'll necessarily have a new content block
        # if we get a function name since it signals a new tool call
        if block_type == "tool_use":
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)
            if tool_block.get("name"):
                self.current_content_block_type = block_type
                self.current_content_block_start = content_block_start
                return True

        return False
