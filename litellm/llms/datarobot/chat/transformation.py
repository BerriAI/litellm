"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as OpenRouter is openai-compatible.

Docs: https://openrouter.ai/docs/parameters
"""

from typing import Any, AsyncIterator, Iterator, Optional, Union, List

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig, LiteLLMLoggingObj
from ..common_utils import DataRobotException


class DataRobotConfig(OpenAIGPTConfig):
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return DataRobotException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Ensure - /v1/chat/completions is at the end of the url

        """
        if api_base is None:
            api_base = "https://app.datarobot.com"

        if not api_base.endswith("/chat/completions/"):
            api_base += "/api/v2/genai/llmgw/chat/completions/"
        return api_base

    def transform_request(self, *args, **kwargs):
        response = super().transform_request(*args, **kwargs)
        return response
