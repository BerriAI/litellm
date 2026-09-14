from __future__ import annotations

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


def _extract_embedding_data(data_raw: object) -> list[object]:  # mutable-ok: matches list output
    if isinstance(data_raw, dict):
        embedding_item: Final[dict[str, object]] = {  # mutable-ok: building embedding response dict
            "object": data_raw.get("object", "embedding"),
            "embedding": data_raw.get("embedding", ()),
            "index": 0,
        }
        if "sparse_embedding" in data_raw:
            embedding_item["sparse_embedding"] = data_raw["sparse_embedding"]
        return [embedding_item]  # mutable-ok: single item data list
    if isinstance(data_raw, list):
        return data_raw  # mutable-ok: raw list
    return []  # mutable-ok: default empty list


class BytePlusEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration class for BytePlus embedding models (Text and Multimodal Vision embeddings).
    Reference: https://docs.byteplus.com/en/docs/ModelArk
    """

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: matches BaseEmbeddingConfig interface
        return [  # mutable-ok: matches BaseEmbeddingConfig interface
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
        optional_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        litellm_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        stream: bool | None = None,
    ) -> str:
        base_url: Final = (
            api_base
            or litellm.api_base
            or get_secret_str("BYTEPLUS_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_byteplus_base_url()
        ).rstrip("/")

        is_multimodal: Final = (
            "vision" in model.lower() or "multimodal" in model.lower() or optional_params.get("is_multimodal", False)
        )

        endpoint: Final = "/embeddings/multimodal" if is_multimodal else "/embeddings"

        if base_url.endswith(endpoint):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}{endpoint}"
        return f"{base_url}/api/v3{endpoint}"

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        optional_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: matches BaseEmbeddingConfig interface
        supported: Final = frozenset(self.get_supported_openai_params(model))
        optional_params.update(
            {k: v for k, v in non_default_params.items() if k in supported}  # mutable-ok: update params dict
        )
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        headers: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
    ) -> dict:  # mutable-ok: matches BaseEmbeddingConfig interface
        is_multimodal: Final = (
            "vision" in model.lower() or "multimodal" in model.lower() or optional_params.get("is_multimodal", False)
        )

        raw_input: Final = input if isinstance(input, (list, tuple)) else (input,)
        formatted_input: Final[list] = []  # mutable-ok: building input list for json payload
        if is_multimodal:
            for item in raw_input:
                if isinstance(item, str):
                    formatted_input.append({"type": "text", "text": item})  # mutable-ok: payload dict item
                else:
                    formatted_input.append(item)
        else:
            formatted_input.extend(raw_input)

        data: Final[dict[str, object]] = {  # mutable-ok: request body dictionary
            "model": model,
            "input": formatted_input,
        }

        for key in ("encoding_format", "dimensions", "instructions", "sparse_embedding", "user"):
            if key in optional_params and optional_params[key] is not None:
                data[key] = optional_params[key]

        if "extra_body" in optional_params and isinstance(optional_params["extra_body"], dict):
            extra_body: Final = {  # mutable-ok: extra_body dictionary
                k: v for k, v in optional_params["extra_body"].items() if k not in ("model", "input")
            }
            data.update(extra_body)

        return data

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        optional_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        litellm_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
    ) -> EmbeddingResponse:
        try:
            response_json: Final = raw_response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse BytePlus response as JSON: {e}")

        data_raw: Final = response_json.get("data", ())
        data_list: Final = _extract_embedding_data(data_raw)

        transformed_response: Final[dict[str, object]] = {  # mutable-ok: building response dict
            "object": "list",
            "data": data_list,
            "model": response_json.get("model", model),
            "usage": response_json.get("usage") or {},  # mutable-ok: default empty usage dict
        }

        if "id" in response_json:
            transformed_response["id"] = response_json["id"]

        return EmbeddingResponse(**transformed_response)

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: matches BaseEmbeddingConfig interface
        optional_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        litellm_params: dict,  # mutable-ok: matches BaseEmbeddingConfig interface
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: matches BaseEmbeddingConfig interface
        resolved_api_key: Final = (
            api_key or litellm.api_key or get_secret_str("BYTEPLUS_API_KEY") or get_secret_str("ARK_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError("BytePlus API key is required. Set BYTEPLUS_API_KEY or ARK_API_KEY or pass api_key.")
        return get_byteplus_headers(api_key=resolved_api_key, extra_headers=headers)

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: matches BaseEmbeddingConfig interface
    ) -> BytePlusError:
        typed_headers: Final[httpx.Headers] = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return BytePlusError(status_code=status_code, message=error_message, headers=typed_headers)
