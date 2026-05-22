"""
Cohere embedding request/response transformation for SageMaker Marketplace endpoints.

AWS Marketplace Cohere containers expect the native Cohere embed API body
(`texts`, `input_type`, etc.), not HuggingFace TGI `inputs`.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.bedrock.embed.cohere_transformation import (
    BedrockCohereEmbeddingConfig,
)
from litellm.llms.cohere.embed.v1_transformation import CohereEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse

from ..common_utils import SagemakerError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
else:
    LiteLLMLoggingObj = Any


COHERE_EMBED_NAME_MARKERS = (
    "embed-multilingual",
    "embed-english",
    "embed-v4",
    "embed-v3",
)


def is_cohere_sagemaker_embedding_model(model: str) -> bool:
    """
    Detect Cohere embedding endpoints from the SageMaker endpoint / model name.

    The SageMaker endpoint name is opaque, so we match on substrings
    ("cohere" or a Cohere embed model id fragment such as "embed-multilingual-v3").
    Endpoint names should include one of these markers so LiteLLM selects the
    Cohere payload format instead of the HuggingFace TGI default.
    """
    model_lower = model.lower()
    if "cohere" in model_lower:
        return True
    return any(marker in model_lower for marker in COHERE_EMBED_NAME_MARKERS)


class SagemakerCohereEmbeddingConfig(BaseEmbeddingConfig):
    """
    SageMaker invoke payload for self-hosted Cohere embed models
    (AWS Marketplace / JumpStart).
    """

    def __init__(self) -> None:
        self._bedrock_cohere = BedrockCohereEmbeddingConfig()
        self._cohere_v1 = CohereEmbeddingConfig()

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["encoding_format", "dimensions", "input_type"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return self._bedrock_cohere.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return SagemakerError(
            message=error_message, status_code=status_code, headers=headers
        )

    def _normalize_input(self, input: AllEmbeddingInputValues) -> List[str]:
        if isinstance(input, str):
            return [input]
        if isinstance(input, list):
            if input and (isinstance(input[0], list) or isinstance(input[0], int)):
                raise ValueError("Input must be a list of strings")
            return cast(List[str], input)
        return [str(input)]

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        input_list = self._normalize_input(input)
        request = self._bedrock_cohere._transform_request(
            model=model,
            input=input_list,
            inference_params=optional_params,
        )
        return dict(request)

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
        input_value = logging_obj.model_call_details.get("input")
        if input_value is None:
            input_value = (
                request_data.get("texts") or request_data.get("images") or []
            )
        if isinstance(input_value, str):
            input_list = [input_value]
        elif isinstance(input_value, list):
            input_list = input_value
        else:
            input_list = []

        return self._cohere_v1._transform_response(
            response=raw_response,
            api_key=api_key,
            logging_obj=logging_obj,
            data=request_data,
            model_response=model_response,
            model=model,
            encoding=litellm.encoding,
            input=input_list,
        )

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
        return {"Content-Type": "application/json"}
