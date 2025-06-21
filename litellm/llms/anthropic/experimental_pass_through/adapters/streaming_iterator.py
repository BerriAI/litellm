# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from typing import Any, AsyncIterator, Iterator, Optional

from litellm import verbose_logger
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import AdapterCompletionStreamWrapper


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    sent_last_message: bool = False
    holding_chunk: Optional[Any] = None

    def __next__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                return {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-3-5-sonnet-20240620",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": UsageDelta(input_tokens=0, output_tokens=0),
                    },
                }
            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                return {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }

            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk
                )
                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    self.holding_chunk = processed_chunk
                    self.sent_content_block_finish = True
                    return {
                        "type": "content_block_stop",
                        "index": 0,
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

    async def __anext__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                return {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-3-5-sonnet-20240620",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": UsageDelta(input_tokens=0, output_tokens=0),
                    },
                }
            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                return {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk
                )
                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    self.holding_chunk = processed_chunk
                    self.sent_content_block_finish = True
                    return {
                        "type": "content_block_stop",
                        "index": 0,
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
