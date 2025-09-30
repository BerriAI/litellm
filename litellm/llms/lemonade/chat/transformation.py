"""
Translate from OpenAI's `/v1/chat/completions` to Lemonade's `/v1/chat/completions`
"""
from typing import Any, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
)
from litellm.types.utils import ModelResponse

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class LemonadeChatConfig(OpenAILikeChatConfig):
    repeat_penalty: Optional[float] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None

    def __init__(
        self,
        repeat_penalty: Optional[float] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "lemonade"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Get available models from Lemonade API.
        
        This method queries the Lemonade /models endpoint to retrieve the list of available models.
        
        Args:
            api_key: Optional API key (Lemonade doesn't require authentication)
            api_base: Optional API base URL (defaults to LEMONADE_API_BASE env var or http://localhost:8000)
            
        Returns:
            List of model names prefixed with "lemonade/"
        """
        api_base, api_key = self._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )
        
        if api_base is None:
            raise ValueError(
                "LEMONADE_API_BASE is not set. Please set the environment variable to query Lemonade's /models endpoint."
            )

        # Getting the list of models from lemonade
        try:
            response = litellm.module_level_client.get(
                url=f"{api_base}/models",
            )
        except Exception as e:
            raise ValueError(
                f"Failed to fetch models from Lemonade. Set Lemonade API Base via `LEMONADE_API_BASE` environment variable. Error: {e}"
            )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Lemonade. Status code: {response.status_code}, Response: {response.text}"
            )

        model_list = response.json().get("data", [])
        return ["lemonade/" + model["id"] for model in model_list]

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # lemonade is openai compatible, we just need to set this to custom_openai and have the api_base be lemonade's endpoint
        api_base = (
            api_base
            or get_secret_str("LEMONADE_API_BASE")
            or "http://localhost:8000/api/v1"
        )  # type: ignore
        # Lemonade doesn't check the key
        key = "lemonade"
        return api_base, key


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

        # Storing lemonade in the model response for easier cost calculation later
        setattr(model_response, "model", "lemonade/" + model)

        return model_response
    