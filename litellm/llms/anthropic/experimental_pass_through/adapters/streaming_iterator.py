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
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


class _MultiToolCallSplitter:
    """Wraps an upstream OpenAI-format streaming iterator and splits any chunk
    whose ``delta.tool_calls`` contains more than one entry into multiple
    chunks (one tool_call each).

    Supports both sync (``__iter__`` / ``__next__``) and async (``__aiter__``
    / ``__anext__``) consumption transparently, deciding which protocol to
    use based on which one is invoked first. This is necessary because some
    upstream stream wrappers expose both protocols, and consumers
    (``AnthropicStreamWrapper.__next__`` vs ``__anext__``) pick the matching
    one — wrapping the upstream stream with a sync-only or async-only
    generator at construction time would break whichever protocol is unused.

    Without this splitting, the downstream converter in
    ``AnthropicStreamWrapper`` (which indexes ``tool_calls[0]`` in
    ``_translate_streaming_openai_chunk_to_anthropic_content_block``)
    silently drops every tool_call beyond the first when a provider emits
    multiple parallel tool_calls in one OpenAI delta (e.g. mlx_lm.server).
    """

    def __init__(self, stream: Any):
        self._stream = stream
        self._buffer: deque = deque()
        # Lazily set the first time __iter__ / __aiter__ is called. Typed
        # ``Any`` (rather than ``Optional[Any]``) so mypy doesn't ask us to
        # narrow the None case at every call site — Python's iteration
        # protocol contract already guarantees __iter__ runs before __next__.
        self._sync_iter_obj: Any = None
        self._async_iter_obj: Any = None

    def __iter__(self) -> "Iterator[Any]":
        if self._sync_iter_obj is None:
            self._sync_iter_obj = iter(self._stream)
        return self

    def __next__(self) -> Any:
        if self._buffer:
            return self._buffer.popleft()
        chunk = next(self._sync_iter_obj)  # raises StopIteration at EOF
        splits = AnthropicStreamWrapper._split_chunk_by_tool_calls(chunk)
        if len(splits) > 1:
            self._buffer.extend(splits[1:])
        return splits[0]

    def __aiter__(self) -> "AsyncIterator[Any]":
        if self._async_iter_obj is None:
            self._async_iter_obj = self._stream.__aiter__()
        return self

    async def __anext__(self) -> Any:
        if self._buffer:
            return self._buffer.popleft()
        chunk = await self._async_iter_obj.__anext__()
        splits = AnthropicStreamWrapper._split_chunk_by_tool_calls(chunk)
        if len(splits) > 1:
            self._buffer.extend(splits[1:])
        return splits[0]


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    from litellm.types.llms.anthropic import (
        ContentBlockContentBlockDict,
        ContentBlockStart,
        ContentBlockStartText,
        TextBlock,
    )

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    current_content_block_type: Literal["text", "tool_use", "thinking"] = "text"
    sent_last_message: bool = False
    holding_chunk: Optional[Any] = None
    holding_stop_reason_chunk: Optional[Any] = None
    queued_usage_chunk: bool = False
    current_content_block_index: int = 0
    current_content_block_start: ContentBlockContentBlockDict = TextBlock(
        type="text",
        text="",
    )
    chunk_queue: deque = deque()  # Queue for buffering multiple chunks

    def __init__(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(completion_stream)
        self.model = model
        # Mapping of truncated tool names to original names (for OpenAI's 64-char limit)
        self.tool_name_mapping = tool_name_mapping or {}

        # Wrap upstream stream so chunks containing multiple tool_calls in a
        # single delta are split into one tool_call per chunk before reaching
        # the downstream converter. The converter assumes one tool_call per
        # chunk (it indexes ``tool_calls[0]`` in
        # ``_translate_streaming_openai_chunk_to_anthropic_content_block``),
        # so without this split, providers that emit parallel tool_calls in
        # one OpenAI delta (e.g. mlx_lm.server) lose all but the first call.
        # The wrapper supports both sync and async iteration; whichever
        # ``AnthropicStreamWrapper.__next__`` / ``__anext__`` invokes is
        # served from the underlying stream's matching protocol.
        #
        # Stored under a separate attribute (rather than overwriting
        # ``self.completion_stream`` from the superclass) so consumers that
        # rely on the original stream still see it, and to keep static
        # analyzers happy about subclass attribute shadowing.
        self._completion_stream_splitter = _MultiToolCallSplitter(completion_stream)

    @staticmethod
    def _split_chunk_by_tool_calls(chunk: Any) -> List[Any]:
        """Split one streaming chunk into N chunks if its delta contains
        multiple tool_calls. Returns a list of chunks (length 1 for normal
        chunks, N for multi-tool-call chunks).
        """
        if chunk is None or chunk == "None":
            return [chunk]
        try:
            tcs = (
                chunk.choices[0].delta.tool_calls
                if chunk.choices and chunk.choices[0].delta is not None
                else None
            )
        except (AttributeError, IndexError):
            return [chunk]
        if tcs is None or len(tcs) <= 1:
            return [chunk]
        out: List[Any] = []
        for one_tc in tcs:
            sub = copy.deepcopy(chunk)
            # Deep-copy the tool_call too so each split chunk has fully
            # independent state — without this, mutations downstream on
            # ``one_tc`` (e.g. argument deltas) would leak into the original
            # chunk's tool_calls list and into peer split chunks.
            sub.choices[0].delta.tool_calls = [copy.deepcopy(one_tc)]
            out.append(sub)
        return out

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

            for chunk in self._completion_stream_splitter:
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

            async for chunk in self._completion_stream_splitter:
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
