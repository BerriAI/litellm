"""
Translate from OpenAI's `/v1/embeddings` to Sagemaker's `/invoke`

In the native Cohere embed format for self-hosted Cohere endpoints
(AWS Marketplace / JumpStart). Cohere containers expect
`{"texts": [...], "input_type": "..."}` and reject the HuggingFace TGI shape
`{"inputs": [...]}` with `422 EmbedReqV2.inputs is of type string but should
be of type Object`.

Reference: https://docs.cohere.com/v2/reference/embed
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import AllEmbeddingInputValues

from httpx._models import Headers, Response
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, require_attributes, subscript
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.bedrock.embed.cohere_transformation import (
    BedrockCohereEmbeddingConfig,
)
from litellm.llms.cohere.embed.v1_transformation import CohereEmbeddingConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import is_base64_encoded

from ..common_utils import SagemakerError

_DECODED_JSON: Final = TypeAdapter(object)


def _cohere_embed_body(response_json: object) -> dict[str, object]:
    body: Final = as_str_mapping(response_json)
    if isinstance(body, dict):
        return body
    subscript(response_json, "embeddings")
    return require_attributes(response_json, dict[str, object], "get")


def cohere_extra_body_params(
    config: BaseEmbeddingConfig, input: Sequence[object], extra_body: Mapping[str, object]
) -> Mapping[str, object]:
    """The ``extra_body`` ``input_type`` for a Cohere endpoint, passed as a param so the default is skipped while an
    explicit argument still wins. Image input keeps the ``image`` type the transform sets."""
    if not isinstance(config, SagemakerCohereEmbeddingConfig) or "input_type" not in extra_body:
        return {}
    if any(
        isinstance(item, str) and is_base64_encoded(item) for item in ([input] if isinstance(input, str) else input)
    ):
        return {}
    return {"input_type": extra_body["input_type"]}


class SagemakerCohereEmbeddingConfig(BaseEmbeddingConfig):
    """
    SageMaker invoke payload for self-hosted Cohere embed models.
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["encoding_format", "dimensions", "input_type"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
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

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return SagemakerError(message=error_message, status_code=status_code, headers=headers)

    def transform_embedding_request(
        self,
        model: str,
        input: "AllEmbeddingInputValues",
        optional_params: dict[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:
        """
        Transform embedding request for Cohere models on SageMaker
        """
        if isinstance(input, str):
            input_list: list[str] = [input]
        elif isinstance(input, list):
            if input and (isinstance(input[0], list) or isinstance(input[0], int)):
                raise ValueError("Input must be a list of strings")
            input_list = cast(list[str], input)
        else:
            input_list = [str(input)]

        return dict(
            BedrockCohereEmbeddingConfig().transform_request(
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
        logging_obj: "LiteLLMLoggingObj",
        api_key: str | None = None,
        request_data: dict = {},
        optional_params: Mapping[str, object] = {},
        litellm_params: Mapping[str, object] = {},
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

        response_json: Final = _DECODED_JSON.validate_python(raw_response.json())
        return CohereEmbeddingConfig().populate_embedding_response(
            model_response=model_response,
            model=model,
            encoding=litellm.encoding,
            input=input_value,
            response_json=_cohere_embed_body(response_json),
        )

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        """
        Validate environment for SageMaker Cohere embeddings
        """
        return {"Content-Type": "application/json"}
