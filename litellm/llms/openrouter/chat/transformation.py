"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as OpenRouter is openai-compatible.

Docs: https://openrouter.ai/docs/parameters
"""

from typing import Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import OpenRouterException


class OpenrouterConfig(OpenAIGPTConfig):

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # OpenRouter-only parameters
        extra_body = {}
        transforms = non_default_params.pop("transforms", None)
        models = non_default_params.pop("models", None)
        route = non_default_params.pop("route", None)
        if transforms is not None:
            extra_body["transforms"] = transforms
        if models is not None:
            extra_body["models"] = models
        if route is not None:
            extra_body["route"] = route
        mapped_openai_params["extra_body"] = (
            extra_body  # openai client supports `extra_body` param
        )
        return mapped_openai_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
