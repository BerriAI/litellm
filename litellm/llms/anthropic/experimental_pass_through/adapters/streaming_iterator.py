# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Literal, Optional

from litellm import verbose_logger
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

    def __init__(self, completion_stream: Any, model: str):
        super().__init__(completion_stream)
        self.model = model

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    current_content_block_type: Literal["text", "tool_use"] = "text"
    sent_last_message: bool = False
    holding_chunk: Optional[Any] = None
    holding_stop_reason_chunk: Optional[Any] = None
    current_content_block_index: int = 0
    current_content_block_start: ContentBlockContentBlockDict = TextBlock(
        type="text",
        text="",
    )
    pending_new_content_block: bool = False
    chunk_queue: deque = deque()  # Queue for buffering multiple chunks

    def __next__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                return {
                    "type": "message_start",
                    "message": {
                        "id": "msg_{}".format(uuid.uuid4()),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": UsageDelta(input_tokens=0, output_tokens=0),
                    },
                }
            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                return {
                    "type": "content_block_start",
                    "index": self.current_content_block_index,
                    "content_block": {"type": "text", "text": ""},
                }

            # Handle pending new content block start
            if self.pending_new_content_block:
                self.pending_new_content_block = False
                self.sent_content_block_finish = False  # Reset for new block
                return {
                    "type": "content_block_start",
                    "index": self.current_content_block_index,
                    "content_block": self.current_content_block_start,
                }

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

                # Check if we need to start a new content block
                # This is where you'd add your logic to detect when a new content block should start
                # For example, if the chunk indicates a tool call or different content type

                if should_start_new_block and not self.sent_content_block_finish:
                    # End current content block and prepare for new one
                    self.holding_chunk = processed_chunk
                    self.sent_content_block_finish = True
                    self.pending_new_content_block = True
                    return {
                        "type": "content_block_stop",
                        "index": max(self.current_content_block_index - 1, 0),
                    }

                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    self.holding_chunk = processed_chunk
                    self.sent_content_block_finish = True
                    return {
                        "type": "content_block_stop",
                        "index": self.current_content_block_index,
                    }
                elif self.holding_chunk is not None:
                    return_chunk = self.holding_chunk
                    self.holding_chunk = processed_chunk
                    return return_chunk
                else:
                    return processed_chunk
            if self.holding_chunk is not None:
                return_chunk = self.holding_chunk
                self.holding_chunk = None
                return return_chunk
            if self.sent_last_message is False:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopIteration
        except StopIteration:
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
                            "usage": UsageDelta(input_tokens=0, output_tokens=0),
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
                    merged_chunk["usage"] = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }

                    # Queue the merged chunk and reset
                    self.chunk_queue.append(merged_chunk)
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                # Check if this processed chunk has a stop_reason - hold it for next chunk

                if should_start_new_block and not self.sent_content_block_finish:
                    # Queue the sequence: content_block_stop -> content_block_start -> current_chunk

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

                    # 3. Queue the current chunk (don't lose it!)
                    self.chunk_queue.append(processed_chunk)

                    # Reset state for new block
                    self.sent_content_block_finish = False

                    # Return the first queued item
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
                    if processed_chunk.get("delta", {}).get("stop_reason") is not None:

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

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        return False
