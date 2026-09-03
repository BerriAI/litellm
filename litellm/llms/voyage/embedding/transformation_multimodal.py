"""
Transform request/response for Voyage multimodal embeddings.

Voyage multimodal models use /v1/multimodalembeddings and accept `inputs`
containing content blocks, unlike standard Voyage embeddings which use
/v1/embeddings and a string/list `input` field.
"""

from typing import Any, Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class VoyageMultimodalEmbeddingError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.voyageai.com/v1/multimodalembeddings")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class VoyageMultimodalEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.voyageai.com/reference/multimodal-embeddings-api
    """

    @staticmethod
    def is_multimodal_embeddings(model: str) -> bool:
        return "multimodal" in model.lower()

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
            if not api_base.endswith("/multimodalembeddings"):
                api_base = f"{api_base}/multimodalembeddings"
            return api_base
        return "https://api.voyageai.com/v1/multimodalembeddings"

    def get_supported_openai_params(self, model: str) -> list:
        return ["dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
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
        if api_key is None:
            api_key = (
                get_secret_str("VOYAGE_API_KEY")
                or get_secret_str("VOYAGE_AI_API_KEY")
                or get_secret_str("VOYAGE_AI_TOKEN")
            )
        if not api_key:
            raise ValueError(
                "Voyage API key is required for multimodal embeddings. "
                "Set VOYAGE_API_KEY / VOYAGE_AI_API_KEY / VOYAGE_AI_TOKEN "
                "or pass `api_key` explicitly."
            )
        return {"Authorization": f"Bearer {api_key}"}

    def _normalize_content_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_type: Final = item.get("type")
        if item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if image_url is None:
                raise ValueError(
                    "Voyage multimodal embeddings require a non-empty `image_url`. "
                    "Got an image content block without a `url`."
                )
            if isinstance(image_url, str) and image_url.startswith("data:image/"):
                _, _, encoded = image_url.partition(",")
                return {"type": "image_base64", "image_base64": encoded}
            return {"type": "image_url", "image_url": image_url}
        return item

    def _normalize_input_item(self, item: Any) -> dict[str, Any]:
        if isinstance(item, str):
            return {"content": [{"type": "text", "text": item}]}
        if isinstance(item, dict) and "content" in item:
            content: Final = item.get("content") or []
            return {
                **item,
                "content": [self._normalize_content_item(content_item) for content_item in content],
            }
        return item

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        inputs: Final = input if isinstance(input, list) else [input]
        return {
            "inputs": [self._normalize_input_item(item) for item in inputs],
            "model": model,
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
            raise VoyageMultimodalEmbeddingError(message=raw_response.text, status_code=raw_response.status_code)

        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        usage_payload: Final = raw_response_json.get("usage", {})
        total_tokens: Final = usage_payload.get("total_tokens", 0)
        model_response.usage = Usage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        )
        return model_response

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VoyageMultimodalEmbeddingError(message=error_message, status_code=status_code, headers=headers)
