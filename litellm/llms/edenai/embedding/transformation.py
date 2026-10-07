"""
Support for OpenAI's `/v1/embeddings` endpoint on Eden AI, served at `/v3/embeddings` with the real
per-request cost at the top level of the body.

Docs: https://www.edenai.co/docs/v3/llms/embeddings
"""

from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import EdenAIException, endpoint_url, json_headers, pick, reported_cost

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SUPPORTED_PARAMS: Final = ("dimensions", "encoding_format", "user")


class EdenAIEmbeddingConfig(BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited contract
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        return {**optional_params, **pick(non_default_params, _SUPPORTED_PARAMS)}

    def validate_environment(
        self,
        headers: dict[str, object],
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:
        return json_headers(headers, api_key, model)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        return endpoint_url(api_base, "embeddings")

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict[str, object],
        headers: dict[str, object],
    ) -> dict[str, object]:
        return {"model": model, "input": input, **optional_params}

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: "LiteLLMLoggingObj",
        api_key: str | None,
        request_data: dict[str, object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
    ) -> EmbeddingResponse:
        body: Final = raw_response.json()
        logging_obj.post_call(original_response=body)
        response: Final[EmbeddingResponse] = convert_to_model_response_object(
            response_object=body, model_response_object=model_response, response_type="embedding"
        )
        set_response_cost_in_hidden_params(response, reported_cost(body))
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,
    ) -> BaseLLMException:
        return EdenAIException(message=error_message, status_code=status_code, headers=headers)
