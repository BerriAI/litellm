"""
OpenRouter Embedding API Configuration.

This module provides the configuration for OpenRouter's Embedding API.
OpenRouter is OpenAI-compatible and supports embeddings via the /v1/embeddings endpoint.

Docs: https://openrouter.ai/docs
"""
from typing import TYPE_CHECKING, Any, Optional

import httpx

from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import OpenRouterException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenrouterEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for OpenRouter's Embedding API.

    Reference: https://openrouter.ai/docs
    """

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set up headers for OpenRouter API.

        OpenRouter requires:
        - Authorization header with Bearer token
        - HTTP-Referer header (site URL)
        - X-Title header (app name)
        """
        from litellm import get_secret

        # Get OpenRouter-specific headers
        openrouter_site_url = get_secret("OR_SITE_URL") or "https://litellm.ai"
        openrouter_app_name = get_secret("OR_APP_NAME") or "liteLLM"

        openrouter_headers = {
            "HTTP-Referer": openrouter_site_url,
            "X-Title": openrouter_app_name,
            "Content-Type": "application/json",
        }

        # Add Authorization header if api_key is provided
        if api_key:
            openrouter_headers["Authorization"] = f"Bearer {api_key}"

        # Merge with existing headers (user's extra_headers take priority)
        merged_headers = {**openrouter_headers, **headers}

        return merged_headers

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
        Get the complete URL for OpenRouter Embedding API endpoint.
        """
        # api_base is already set to https://openrouter.ai/api/v1 in main.py
        # Remove trailing slashes
        if api_base:
            api_base = api_base.rstrip("/")
        else:
            api_base = "https://openrouter.ai/api/v1"

        # Return the embeddings endpoint
        return f"{api_base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request to OpenRouter format (OpenAI-compatible).
        """
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        # OpenRouter expects the full model name (e.g., google/gemini-embedding-001)
        # Strip 'openrouter/' prefix if present
        if model.startswith("openrouter/"):
            model = model.replace("openrouter/", "", 1)

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
        Transform embedding response from OpenRouter format (OpenAI-compatible).
        """
        logging_obj.post_call(original_response=raw_response.text)

        # OpenRouter returns standard OpenAI-compatible embedding response
        response_json = raw_response.json()

        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get list of supported OpenAI parameters for OpenRouter embeddings.
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
        Map OpenAI parameters to OpenRouter format.
        """
        for param, value in non_default_params.items():
            if param in self.get_supported_openai_params(model):
                optional_params[param] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Any
    ) -> Any:
        """
        Get the error class for OpenRouter errors.
        """
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
