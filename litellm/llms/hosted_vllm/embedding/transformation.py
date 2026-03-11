"""
Hosted VLLM Embedding API Configuration.

This module provides the configuration for hosted VLLM's Embedding API.
VLLM is OpenAI-compatible and supports embeddings via the /v1/embeddings endpoint.

Docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class HostedVLLMEmbeddingError(BaseLLMException):
    """Exception class for Hosted VLLM Embedding errors."""

    pass


class HostedVLLMEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for Hosted VLLM's Embedding API.

    Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

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
        """
        Validate environment and set up headers for Hosted VLLM API.
        """
        if api_key is None:
            api_key = get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"

        default_headers = {
            "Content-Type": "application/json",
        }

        # Only add Authorization header if api_key is not "fake-api-key"
        if api_key and api_key != "fake-api-key":
            default_headers["Authorization"] = f"Bearer {api_key}"

        # Merge with existing headers (user's headers take priority)
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for Hosted VLLM Embedding API endpoint.
        """
        if api_base is None:
            api_base = get_secret_str("HOSTED_VLLM_API_BASE")
            if api_base is None:
                raise ValueError("api_base is required for hosted_vllm embeddings")

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Ensure the URL ends with /embeddings
        if not api_base.endswith("/embeddings"):
            api_base = f"{api_base}/embeddings"

        return api_base

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request to Hosted VLLM format (OpenAI-compatible).
        """
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        # Strip 'hosted_vllm/' prefix if present
        if model.startswith("hosted_vllm/"):
            model = model.replace("hosted_vllm/", "", 1)

        return {
            "model": model,
            "input": input,
            **optional_params,
        }

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
        """
        Transform embedding response from Hosted VLLM format (OpenAI-compatible).
        """
        logging_obj.post_call(original_response=raw_response.text)

        # VLLM returns standard OpenAI-compatible embedding response
        response_json = raw_response.json()

        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get list of supported OpenAI parameters for Hosted VLLM embeddings.
        """
        return [
            "timeout",
            "dimensions",
            "encoding_format",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Hosted VLLM format.
        """
        for param, value in non_default_params.items():
            if param in self.get_supported_openai_params(model):
                optional_params[param] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get the error class for Hosted VLLM errors.
        """
        return HostedVLLMEmbeddingError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
