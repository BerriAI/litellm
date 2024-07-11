# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import os
import traceback
import uuid
from typing import Literal, Optional

import dotenv
import httpx
from pydantic import BaseModel

import litellm
from litellm import ChatCompletionRequest, verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.anthropic import AnthropicMessagesRequest, AnthropicResponse


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

        translated_body = litellm.AnthropicConfig().translate_anthropic_to_openai(
            anthropic_message_request=request_body
        )

        return translated_body

    def translate_completion_output_params(
        self, response: litellm.ModelResponse
    ) -> Optional[AnthropicResponse]:

        return litellm.AnthropicConfig().translate_openai_response_to_anthropic(
            response=response
        )

    def translate_completion_output_params_streaming(self) -> Optional[BaseModel]:
        return super().translate_completion_output_params_streaming()


anthropic_adapter = AnthropicAdapter()
