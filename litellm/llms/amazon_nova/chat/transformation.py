"""
Translate from OpenAI's `/v1/chat/completions` to Amazon Nova's `/v1/chat/completions`
"""
from typing import Any, List, Optional, Tuple

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
)
from litellm.types.utils import ModelResponse

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class AmazonNovaChatConfig(OpenAILikeChatConfig):
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    metadata: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    tools: Optional[list] = None
    reasoning_effort: Optional[list] = None

    def __init__(
        self,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        tools: Optional[list] = None,
        reasoning_effort: Optional[list] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "amazon_nova"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Amazon Nova is openai compatible, we just need to set this to custom_openai and have the api_base be Nova's endpoint
        api_base = (
            api_base
            or get_secret_str("AMAZON_NOVA_API_BASE")
            or "https://api.nova.amazon.com/v1"
        )  # type: ignore
        
        # Get API key from multiple sources
        key = (
            api_key
            or litellm.amazon_nova_api_key
            or get_secret_str("AMAZON_NOVA_API_KEY")
            or litellm.api_key
        )
        return api_base, key
    
    def get_supported_openai_params(self, model: str) -> List:
        return [
            "top_p",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "metadata",
            "stop",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "reasoning_effort"
        ]

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        model_response = super().transform_response(
            model=model,
            model_response=model_response,
            raw_response=raw_response,
            messages=messages,
            logging_obj=logging_obj,
            request_data=request_data,
            encoding=encoding,
            optional_params=optional_params,
            json_mode=json_mode,
            litellm_params=litellm_params,
            api_key=api_key,
        )

        # Storing amazon_nova in the model response for easier cost calculation later
        setattr(model_response, "model", "amazon-nova/" + model)

        return model_response