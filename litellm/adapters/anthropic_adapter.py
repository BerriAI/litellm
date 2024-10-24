# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import os
import traceback
import uuid
from typing import Any, Literal, Optional

import dotenv
import httpx
from pydantic import BaseModel

import litellm
from litellm import ChatCompletionRequest, verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequest,
    AnthropicResponse,
    ContentBlockDelta,
)
from litellm.types.utils import AdapterCompletionStreamWrapper


class AnthropicAdapter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()

    def translate_completion_input_params(
        self, kwargs
    ) -> Optional[ChatCompletionRequest]:
        """
        - translate params, where needed
        - pass rest, as is
        """
        request_body = AnthropicMessagesRequest(**kwargs)  # type: ignore

        translated_body = litellm.AnthropicExperimentalPassThroughConfig().translate_anthropic_to_openai(
            anthropic_message_request=request_body
        )

        return translated_body

    def translate_completion_output_params(
        self, response: litellm.ModelResponse
    ) -> Optional[AnthropicResponse]:

        return litellm.AnthropicExperimentalPassThroughConfig().translate_openai_response_to_anthropic(
            response=response
        )

    def translate_completion_output_params_streaming(
        self, completion_stream: Any
    ) -> AdapterCompletionStreamWrapper | None:
        return AnthropicStreamWrapper(completion_stream=completion_stream)


anthropic_adapter = AnthropicAdapter()


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
                        "usage": {"input_tokens": 25, "output_tokens": 1},
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

                processed_chunk = litellm.AnthropicExperimentalPassThroughConfig().translate_streaming_openai_response_to_anthropic(
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

    async def __anext__(self):
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
                        "usage": {"input_tokens": 25, "output_tokens": 1},
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
                processed_chunk = litellm.AnthropicExperimentalPassThroughConfig().translate_streaming_openai_response_to_anthropic(
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
