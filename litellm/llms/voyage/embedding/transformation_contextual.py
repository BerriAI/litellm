"""
This module is used to transform the request and response for the Voyage contextualized embeddings API.
This would be used for all the contextualized embeddings models in Voyage.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.voyage.common_utils import get_default_base_url, get_voyage_api_key
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

NO_CONTEXTUAL_DEFAULTS: Final[Mapping[str, str | bool]] = MappingProxyType({})
AUTO_CHUNK_DEFAULTS: Final[Mapping[str, str | bool]] = MappingProxyType(
    {"input_type": "document", "enable_auto_chunking": True}
)


class VoyageError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.voyageai.com/v1/contextualizedembeddings")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class VoyageContextualEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.voyageai.com/reference/embeddings-api
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if api_base:
            if not api_base.endswith("/contextualizedembeddings"):
                api_base = f"{api_base}/contextualizedembeddings"
            return api_base
        return f"{get_default_base_url(api_key)}/contextualizedembeddings"

    def get_supported_openai_params(self, model: str) -> list:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Voyage params

        Reference: https://docs.voyageai.com/reference/contextualized-embeddings-api
        """
        if "encoding_format" in non_default_params:
            optional_params["encoding_format"] = non_default_params["encoding_format"]
        if "dimensions" in non_default_params:
            optional_params["output_dimension"] = non_default_params["dimensions"]
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return {
            "Authorization": f"Bearer {get_voyage_api_key(api_key)}",
        }

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues | list[list[str]],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Shape ``inputs`` and the auto-chunking params to match Voyage's contextualized
        embeddings contract: ``list[list[str]]`` (pre-chunked documents) is always
        valid, while a flat ``list[str]`` or bare ``str`` is valid only as queries
        (``input_type="query"``) or as documents with ``enable_auto_chunking=True``
        and ``input_type="document"``. Caller-set params win.

        Reference: https://docs.voyageai.com/docs/contextualized-chunk-embeddings
        """
        is_prechunked: Final = isinstance(input, list) and len(input) > 0 and isinstance(input[0], list)
        is_query: Final = optional_params.get("input_type") == "query"
        return {
            "inputs": (input,) if isinstance(input, str) else input,
            "model": model,
            **(NO_CONTEXTUAL_DEFAULTS if is_prechunked or is_query else AUTO_CHUNK_DEFAULTS),
            **optional_params,
        }

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        try:
            raw_response_json: Final = raw_response.json()
        except Exception:
            raise VoyageError(message=raw_response.text, status_code=raw_response.status_code)

        # model_response.usage
        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        usage: Final = Usage(
            prompt_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
            total_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
        )
        model_response.usage = usage
        return model_response

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VoyageError(message=error_message, status_code=status_code, headers=headers)

    @staticmethod
    def is_contextualized_embeddings(model: str) -> bool:
        return "context" in model.lower()
