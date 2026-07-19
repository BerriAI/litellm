"""
Translate from OpenAI's `/v1/embeddings` to Sagemaker's `/invoke`

In the native Cohere embed format for self-hosted Cohere endpoints
(AWS Marketplace / JumpStart). Cohere containers expect
`{"texts": [...], "input_type": "..."}` and reject the HuggingFace TGI shape
`{"inputs": [...]}` with `422 EmbedReqV2.inputs is of type string but should
be of type Object`.

Reference: https://docs.cohere.com/v2/reference/embed
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllEmbeddingInputValues

from httpx._models import Headers, Response

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.bedrock.embed.cohere_transformation import (
    BedrockCohereEmbeddingConfig,
)
from litellm.llms.cohere.embed.v1_transformation import CohereEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

from ..common_utils import SagemakerError


class SagemakerCohereEmbeddingConfig(BaseEmbeddingConfig):
    """
    SageMaker invoke payload for self-hosted Cohere embed models.
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["encoding_format", "dimensions", "input_type"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = BedrockCohereEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
        )
        if "input_type" in non_default_params:
            optional_params["input_type"] = non_default_params["input_type"]
        return optional_params

    def get_error_class(self, error_message: str, status_code: int, headers: Union[dict, Headers]) -> BaseLLMException:
        return SagemakerError(message=error_message, status_code=status_code, headers=headers)

    def transform_embedding_request(
        self,
        model: str,
        input: "AllEmbeddingInputValues",
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request for Cohere models on SageMaker
        """
        if isinstance(input, str):
            input_list: List[str] = [input]
        elif isinstance(input, list):
            if input and (isinstance(input[0], list) or isinstance(input[0], int)):
                raise ValueError("Input must be a list of strings")
            input_list = cast(List[str], input)
        else:
            input_list = [str(input)]

        return dict(
            BedrockCohereEmbeddingConfig()._transform_request(
                model=model,
                input=input_list,
                inference_params=optional_params,
            )
        )

    def transform_embedding_response(
        self,
        model: str,
        raw_response: Response,
        model_response: "EmbeddingResponse",
        logging_obj: Any,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> "EmbeddingResponse":
        """
        Transform embedding response for Cohere models on SageMaker.

        Uses `CohereEmbeddingConfig._populate_embedding_response` (not
        `_transform_response`) so we do not log `post_call` a second time
        — the SageMaker embedding handler already logs `post_call` before
        invoking this transform.
        """
        input_value = (
            logging_obj.model_call_details.get("input") or request_data.get("texts") or request_data.get("images") or []
        )
        if isinstance(input_value, str):
            input_value = [input_value]

        return CohereEmbeddingConfig()._populate_embedding_response(
            response_json=raw_response.json(),
            model_response=model_response,
            model=model,
            encoding=litellm.encoding,
            input=input_value,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment for SageMaker Cohere embeddings
        """
        return {"Content-Type": "application/json"}
