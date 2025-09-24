"""
Translate from OpenAI's `/v1/chat/completions` to Lemonade's `/v1/chat/completions`
"""
from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, cast, overload

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import ModelResponse, ModelInfoBase

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
    top_p: Optional[float] = None
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
        top_p: Optional[float] = None,
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

    def get_model_info(self, model: str) -> ModelInfoBase:
        if model.startswith("lemonade/"):
            model = model.split("/", 1)[1]
        api_base = get_secret_str("LEMONADE_API_BASE") or "http://localhost:8000"

        # Getting the list of models from lemonade to verify the model exists
        try:
            response = litellm.module_level_client.get(
                url=f"{api_base}/api/v1/models",
            )
        except Exception as e:
            raise Exception(
                f"LemonadeError: Error getting model info for {model}. Set Lemonade API Base via `LEMONADE_API_BASE` environment variable. Error: {e}"
            )

        # Making sure the model exists in lemonade
        model_found = False
        model_list = response.json().get("data", [])
        for model_iter in model_list:
            if model_iter['id'] == model:
                model_found = True
                break

        if not model_found:
            raise ValueError(
                f"LemonadeError: Model {model} not found. Available models: {[m['id'] for m in model_list]}"
            )

        # Returning the model if it was found in lemonade. Currently there is no mechanism to report
        # if the model supports function calling or the max tokens so we leave those out
        return ModelInfoBase(
            key=model,
            litellm_provider="lemonade",
            mode="chat",
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
        )

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
    