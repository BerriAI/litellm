"""
Transformation logic from OpenAI /v1/embeddings format to Isaacus's /v1/embeddings format.

Reference: https://docs.isaacus.com/api-reference/embeddings
"""

from typing import List, Optional, Union, cast

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class IsaacusError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Union[dict, httpx.Headers] = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.isaacus.com/v1/embeddings"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class IsaacusEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.isaacus.com/api-reference/embeddings

    The Isaacus embeddings API provides access to the Kanon 2 Embedder for law.
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
                api_base = f"{api_base}/embeddings"
            return api_base
        return "https://api.isaacus.com/v1/embeddings"

    def get_supported_openai_params(self, model: str) -> list:
        return ["dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Isaacus params

        Reference: https://docs.isaacus.com/api-reference/embeddings
        """
        if "dimensions" in non_default_params:
            optional_params["dimensions"] = non_default_params["dimensions"]
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
            api_key = get_secret_str("ISAACUS_API_KEY")
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
        """
        Transform OpenAI-style embedding request to Isaacus format.

        OpenAI uses 'input' while Isaacus uses 'texts'.
        """
        # Convert input to list of strings if needed
        if isinstance(input, str):
            texts = [input]
        elif isinstance(input, list):
            if len(input) > 0 and isinstance(input[0], (list, int)):
                raise ValueError(
                    "Isaacus does not support token array inputs. Input must be a string or list of strings."
                )
            texts = cast(List[str], input)
        else:
            texts = [input]

        request_data = {
            "model": model,
            "texts": texts,
        }

        # Add optional parameters
        # Isaacus-specific parameters: task, overflow_strategy, dimensions
        if "task" in optional_params:
            request_data["task"] = optional_params["task"]
        if "overflow_strategy" in optional_params:
            request_data["overflow_strategy"] = optional_params["overflow_strategy"]
        if "dimensions" in optional_params:
            request_data["dimensions"] = optional_params["dimensions"]

        return request_data

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
            raise IsaacusError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        # Transform Isaacus response format to OpenAI format
        # Isaacus format: {"embeddings": [{"embedding": [...], "index": 0}, ...], "usage": {"input_tokens": 10}}
        # OpenAI format: {"data": [{"embedding": [...], "index": 0, "object": "embedding"}], "model": "...", "usage": {...}}

        embeddings_data = raw_response_json.get("embeddings", [])
        output_data = []

        for emb_obj in embeddings_data:
            output_data.append(
                {
                    "object": "embedding",
                    "index": emb_obj.get("index", 0),
                    "embedding": emb_obj.get("embedding", []),
                }
            )

        model_response.model = model
        model_response.data = output_data
        model_response.object = "list"

        # Set usage information
        # Isaacus returns usage with "input_tokens"
        usage_data = raw_response_json.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)

        usage = Usage(
            prompt_tokens=input_tokens,
            total_tokens=input_tokens,
        )
        model_response.usage = usage

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return IsaacusError(
            message=error_message, status_code=status_code, headers=headers
        )
