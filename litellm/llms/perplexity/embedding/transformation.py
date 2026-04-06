"""
Perplexity AI Embedding API

Docs: https://docs.perplexity.ai/api-reference/embeddings-post

Supports models:
  - pplx-embed-v1-0.6b  (1024 dims, 32 K context)
  - pplx-embed-v1-4b    (2560 dims, 32 K context)

Perplexity returns embeddings as base64-encoded signed int8 values by default.
This module decodes them into float arrays for OpenAI-compatible responses.
"""

import base64
import struct
from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class PerplexityEmbeddingError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Union[dict, httpx.Headers] = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.perplexity.ai/v1/embeddings"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class PerplexityEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.perplexity.ai/api-reference/embeddings-post
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base:
            if not api_base.endswith("/embeddings"):
                api_base = f"{api_base}/v1/embeddings"
            return api_base
        return "https://api.perplexity.ai/v1/embeddings"

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "dimensions",
            "encoding_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "dimensions":
                optional_params["dimensions"] = v
            elif k == "encoding_format":
                optional_params["encoding_format"] = v
        return optional_params

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
        if api_key is None:
            api_key = get_secret_str("PERPLEXITYAI_API_KEY") or get_secret_str(
                "PERPLEXITY_API_KEY"
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return {
            "model": model,
            "input": input,
            **optional_params,
        }

    @staticmethod
    def _decode_base64_embedding(embedding_value: Any) -> List[float]:
        """
        Decode a Perplexity embedding into a list of floats.

        Perplexity returns base64-encoded signed int8 values by default.
        If the value is already a list of numbers (e.g. from a mock or
        future float format), it is returned as-is.
        """
        if isinstance(embedding_value, list):
            return embedding_value
        if isinstance(embedding_value, str):
            raw_bytes = base64.b64decode(embedding_value)
            count = len(raw_bytes)
            int8_values = struct.unpack(f"{count}b", raw_bytes)
            return [float(v) / 127.0 for v in int8_values]
        return embedding_value

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise PerplexityEmbeddingError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        model_response.model = raw_response_json.get("model", model)
        model_response.object = raw_response_json.get("object", "list")

        raw_data = raw_response_json.get("data", [])
        decoded_data: List[Dict[str, Any]] = []
        for item in raw_data:
            decoded_item = dict(item)
            decoded_item["embedding"] = self._decode_base64_embedding(
                item.get("embedding")
            )
            decoded_data.append(decoded_item)
        model_response.data = decoded_data

        usage_data = raw_response_json.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0)
            or usage_data.get("total_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        model_response.usage = usage
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return PerplexityEmbeddingError(
            message=error_message, status_code=status_code, headers=headers
        )
