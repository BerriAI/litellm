import json
from typing import TYPE_CHECKING, Final

import httpx

import litellm
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse

from ..common_utils import (
    BytePlusError,
    get_byteplus_base_url,
    get_byteplus_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class BytePlusEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration class for BytePlus embedding models (Text and Multimodal Vision embeddings).
    Reference: https://docs.byteplus.com/en/docs/ModelArk
    """

    def __init__(
        self,
        encoding_format: str | None = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls) -> "BytePlusEmbeddingConfig":
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list[str]:
        return [
            "encoding_format",
            "user",
            "extra_headers",
            "dimensions",
            "instructions",
            "sparse_embedding",
        ]

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = (
            api_base
            or litellm.api_base
            or get_secret_str("BYTEPLUS_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_byteplus_base_url()
        )
        base_url = base_url.rstrip("/")

        is_multimodal = (
            "vision" in model.lower() or "multimodal" in model.lower() or optional_params.get("is_multimodal", False)
        )

        endpoint = "/embeddings/multimodal" if is_multimodal else "/embeddings"

        if base_url.endswith(endpoint):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}{endpoint}"
        return f"{base_url}/api/v3{endpoint}"

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        optional_params.update({k: v for k, v in non_default_params.items() if k in supported})
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        is_multimodal = (
            "vision" in model.lower() or "multimodal" in model.lower() or optional_params.get("is_multimodal", False)
        )

        raw_input = input if isinstance(input, list) else [input]
        formatted_input: list = []
        if is_multimodal:
            for item in raw_input:
                if isinstance(item, str):
                    formatted_input.append({"type": "text", "text": item})
                else:
                    formatted_input.append(item)
        else:
            formatted_input = list(raw_input)

        data: dict = {
            "model": model,
            "input": formatted_input,
        }

        for key in ["encoding_format", "dimensions", "instructions", "sparse_embedding", "user"]:
            if key in optional_params and optional_params[key] is not None:
                data[key] = optional_params[key]

        if "extra_body" in optional_params and isinstance(optional_params["extra_body"], dict):
            extra_body: Final = {k: v for k, v in optional_params["extra_body"].items() if k not in ("model", "input")}
            data.update(extra_body)

        return data

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: "LiteLLMLoggingObj",
        api_key: str | None,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        try:
            response_json = raw_response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse BytePlus response as JSON: {e}")

        data_raw = response_json.get("data", [])

        if isinstance(data_raw, dict):
            embedding_item: dict = {
                "object": data_raw.get("object", "embedding"),
                "embedding": data_raw.get("embedding", []),
                "index": 0,
            }
            if "sparse_embedding" in data_raw:
                embedding_item["sparse_embedding"] = data_raw["sparse_embedding"]
            data_list = [embedding_item]
        elif isinstance(data_raw, list):
            data_list = data_raw
        else:
            data_list = []

        transformed_response = {
            "object": "list",
            "data": data_list,
            "model": response_json.get("model", model),
            "usage": response_json.get("usage", {}),
        }

        if "id" in response_json:
            transformed_response["id"] = response_json["id"]

        return EmbeddingResponse(**transformed_response)

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
        resolved_api_key: Final = (
            api_key or litellm.api_key or get_secret_str("BYTEPLUS_API_KEY") or get_secret_str("ARK_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError("BytePlus API key is required. Set BYTEPLUS_API_KEY or ARK_API_KEY or pass api_key.")
        return get_byteplus_headers(api_key=resolved_api_key, extra_headers=headers)

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BytePlusError:
        typed_headers: httpx.Headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        return BytePlusError(status_code=status_code, message=error_message, headers=typed_headers)
