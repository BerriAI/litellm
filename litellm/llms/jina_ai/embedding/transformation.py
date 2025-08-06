"""
Transformation logic from OpenAI /v1/embeddings format to Jina AI's  `/v1/embeddings` format.

Why separate file? Make it easy to see how transformation works

Docs - https://jina.ai/embeddings/
"""

import types
from typing import List, Optional, Tuple, Union, cast

import httpx

from litellm import LlmProviders
from litellm.secret_managers.main import get_secret_str
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm import BaseEmbeddingConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import is_base64_encoded

from ..common_utils import JinaAIError


class JinaAIEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://jina.ai/embeddings/
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        if "dimensions" in non_default_params:
            optional_params["dimensions"] = non_default_params["dimensions"]
        return optional_params

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Returns:
            Tuple[str, Optional[str], Optional[str]]:
                - custom_llm_provider: str
                - api_base: str
                - dynamic_api_key: str
        """
        api_base = (
            api_base or get_secret_str("JINA_AI_API_BASE") or "https://api.jina.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or (
            get_secret_str("JINA_AI_API_KEY")
            or get_secret_str("JINA_AI_API_KEY")
            or get_secret_str("JINA_AI_API_KEY")
            or get_secret_str("JINA_AI_TOKEN")
        )
        return LlmProviders.JINA_AI.value, api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return (
            f"{api_base}/embeddings"
            if api_base
            else "https://api.jina.ai/v1/embeddings"
        )

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        data = {"model": model, **optional_params}
        input = cast(List[str], input) if isinstance(input, List) else [input]
        if any((is_base64_encoded(x) for x in input)):
            transformed_input = []
            for value in input:
                if isinstance(value, str):
                    if is_base64_encoded(value):
                        img_data = value.split(",")[1]
                        transformed_input.append({"image": img_data})
                    else:
                        transformed_input.append({"text": value})
            data["input"] = transformed_input
        else:
            data["input"] = input
        return data

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        response_json = raw_response.json()
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )
        return EmbeddingResponse(**response_json)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        default_headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        headers = {**default_headers, **headers}
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return JinaAIError(
            status_code=status_code,
            message=error_message,
        )
