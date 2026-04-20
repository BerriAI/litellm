# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Literal,
    Optional,
    cast,
)

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


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

    def __init__(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(completion_stream)
        self.model = model
        self.sent_first_chunk = False
        self.sent_content_block_start = False
        self.sent_content_block_finish = False
        self.current_content_block_type: Literal["text", "tool_use", "thinking"] = (
            "text"
        )
        self.sent_last_message = False
        self.holding_chunk = None
        self.holding_stop_reason_chunk = None
        self.queued_usage_chunk = False
        self.current_content_block_index = 0
        self.current_content_block_start = {"type": "text", "text": ""}
        self.chunk_queue: deque = deque()
        self.active_tool_call_index: Optional[int] = None
        self.active_tool_call_id: Optional[str] = None
        self.active_tool_call_name: Optional[str] = None
        self.active_tool_content_block_index: Optional[int] = None
        # Mapping of truncated tool names to original names (for OpenAI's 64-char limit)
        self.tool_name_mapping = tool_name_mapping or {}

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
        try:
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            if self._queue_initial_chunks_if_needed():
                return self.chunk_queue.popleft()

            for chunk in self.completion_stream:
                self._process_stream_chunk(chunk)
                if self.chunk_queue:
                    return self.chunk_queue.popleft()

            self._flush_stream_end()
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            raise StopIteration
        except StopIteration:
            self._flush_stream_end()
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            raise StopIteration
        except Exception as e:
            verbose_logger.error(
                "Anthropic Adapter - {}\n{}".format(e, traceback.format_exc())
            )
            raise StopAsyncIteration

    async def __anext__(self):  # noqa: PLR0915
        try:
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            if self._queue_initial_chunks_if_needed():
                return self.chunk_queue.popleft()

            async for chunk in self.completion_stream:
                self._process_stream_chunk(chunk)
                if self.chunk_queue:
                    return self.chunk_queue.popleft()

            self._flush_stream_end()
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            raise StopIteration

        except StopIteration:
            self._flush_stream_end()
            if self.chunk_queue:
                return self.chunk_queue.popleft()
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

    def _queue_initial_chunks_if_needed(self) -> bool:
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
            return True

        if self.sent_content_block_start is False:
            self.sent_content_block_start = True
            self.sent_content_block_finish = False
            self.chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": self.current_content_block_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            return True

        return False

    def _chunk_has_tool_calls(self, chunk: "ModelResponseStream") -> bool:
        for choice in chunk.choices:
            delta = choice.delta
            tool_calls = (
                delta.get("tool_calls")
                if isinstance(delta, dict)
                else getattr(delta, "tool_calls", None)
            )
            if tool_calls is not None and len(tool_calls) > 0:
                return True
        return False

    def _queue_content_block_stop(self, index: int) -> None:
        self.chunk_queue.append({"type": "content_block_stop", "index": index})

    def _close_active_tool_call(self) -> None:
        if self.active_tool_content_block_index is None:
            return

        self._queue_content_block_stop(self.active_tool_content_block_index)
        self.sent_content_block_finish = True
        self.active_tool_call_index = None
        self.active_tool_call_id = None
        self.active_tool_call_name = None
        self.active_tool_content_block_index = None

    def _build_usage_dict(self, chunk: "ModelResponseStream") -> UsageDelta:
        chunk_usage = getattr(chunk, "usage", None)
        assert chunk_usage is not None

        uncached_input_tokens = chunk_usage.prompt_tokens or 0
        if (
            hasattr(chunk_usage, "prompt_tokens_details")
            and chunk_usage.prompt_tokens_details
        ):
            cached_tokens = (
                getattr(chunk_usage.prompt_tokens_details, "cached_tokens", 0) or 0
            )
            uncached_input_tokens -= cached_tokens

        usage_dict: UsageDelta = {
            "input_tokens": uncached_input_tokens,
            "output_tokens": chunk_usage.completion_tokens or 0,
        }
        if (
            hasattr(chunk_usage, "_cache_creation_input_tokens")
            and chunk_usage._cache_creation_input_tokens > 0
        ):
            usage_dict["cache_creation_input_tokens"] = (
                chunk_usage._cache_creation_input_tokens
            )
        if (
            hasattr(chunk_usage, "_cache_read_input_tokens")
            and chunk_usage._cache_read_input_tokens > 0
        ):
            usage_dict["cache_read_input_tokens"] = chunk_usage._cache_read_input_tokens
        return usage_dict

    def _process_tool_call_chunk(self, chunk: "ModelResponseStream") -> None:
        from .transformation import LiteLLMAnthropicMessagesAdapter

        adapter = LiteLLMAnthropicMessagesAdapter()
        tool_calls = adapter.iter_streaming_tool_calls(chunk.choices)  # type: ignore[arg-type]
        if len(tool_calls) == 0:
            return

        if (
            self.current_content_block_type != "tool_use"
            and not self.sent_content_block_finish
        ):
            self._queue_content_block_stop(self.current_content_block_index)
            self.sent_content_block_finish = True

        self.current_content_block_type = "tool_use"

        for tool_call in tool_calls:
            tool_call_index = tool_call.index or 0
            tool_call_id = tool_call.id
            tool_call_name = None
            if tool_call.function is not None:
                tool_call_name = tool_call.function.name
            tool_call_starts_new_block = bool(tool_call_name)

            tool_call_matches_active = (
                self.active_tool_call_index == tool_call_index
                and (tool_call_id is None or tool_call_id == self.active_tool_call_id)
                and (
                    tool_call_name is None
                    or tool_call_name == self.active_tool_call_name
                )
            )

            if (
                self.active_tool_content_block_index is None
                or tool_call_starts_new_block
                or not tool_call_matches_active
            ):
                if self.active_tool_content_block_index is not None:
                    self._close_active_tool_call()

                self._increment_content_block_index()
                anthropic_index = self.current_content_block_index
                self.active_tool_call_index = tool_call_index
                self.active_tool_call_id = tool_call_id
                self.active_tool_call_name = tool_call_name
                self.active_tool_content_block_index = anthropic_index
                content_block = (
                    adapter.translate_streaming_tool_call_to_anthropic_content_block(
                        tool_call
                    )
                )
                if tool_call_name:
                    content_block_dict = cast(Dict[str, Any], content_block)
                    content_block_dict["name"] = self.tool_name_mapping.get(
                        tool_call_name, tool_call_name
                    )
                self.chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": anthropic_index,
                        "content_block": content_block,
                    }
                )
            else:
                anthropic_index = self.active_tool_content_block_index

            content_block_delta = (
                adapter.translate_streaming_tool_call_to_anthropic_delta(tool_call)
            )
            if content_block_delta is not None:
                self.chunk_queue.append(
                    {
                        "type": "content_block_delta",
                        "index": anthropic_index,
                        "delta": content_block_delta,
                    }
                )

        self.sent_content_block_finish = False

    def _process_non_tool_chunk(self, chunk: "ModelResponseStream") -> None:
        from .transformation import LiteLLMAnthropicMessagesAdapter

        if self.active_tool_content_block_index is not None:
            self._close_active_tool_call()

        should_start_new_block = self._should_start_new_content_block(chunk)
        if should_start_new_block:
            self._increment_content_block_index()

        processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
            response=cast(Any, chunk),
            current_content_block_index=self.current_content_block_index,
        )

        if (
            self.holding_stop_reason_chunk is not None
            and getattr(chunk, "usage", None) is not None
        ):
            merged_chunk = self.holding_stop_reason_chunk.copy()
            if "delta" not in merged_chunk:
                merged_chunk["delta"] = {}
            merged_chunk["usage"] = self._build_usage_dict(chunk)
            self.chunk_queue.append(merged_chunk)
            self.queued_usage_chunk = True
            self.holding_stop_reason_chunk = None
            return

        if should_start_new_block:
            self.chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": self.current_content_block_index,
                    "content_block": self.current_content_block_start,
                }
            )
            self.sent_content_block_finish = False

        if processed_chunk["type"] == "message_delta":
            if self.sent_content_block_finish is False:
                self._queue_content_block_stop(self.current_content_block_index)
                self.sent_content_block_finish = True

            if (
                processed_chunk.get("delta", {}).get("stop_reason") is not None
                and getattr(chunk, "usage", None) is None
            ):
                self.holding_stop_reason_chunk = processed_chunk
            else:
                self.chunk_queue.append(processed_chunk)
            return

        self.chunk_queue.append(processed_chunk)

    def _process_stream_chunk(self, chunk: "ModelResponseStream") -> None:
        if chunk == "None" or chunk is None:
            raise Exception

        self.queued_usage_chunk = False

        if self._chunk_has_tool_calls(chunk):
            self._process_tool_call_chunk(chunk)
            return

        self._process_non_tool_chunk(chunk)

    def _flush_stream_end(self) -> None:
        if self.active_tool_content_block_index is not None:
            self._close_active_tool_call()

        if self.holding_stop_reason_chunk is not None:
            self.chunk_queue.append(self.holding_stop_reason_chunk)
            self.holding_stop_reason_chunk = None

        if self.holding_chunk is not None:
            self.chunk_queue.append(self.holding_chunk)
            self.holding_chunk = None

        if not self.sent_last_message:
            self.sent_last_message = True
            self.chunk_queue.append({"type": "message_stop"})

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
