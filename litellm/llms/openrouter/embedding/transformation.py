"""
OpenRouter Embedding API Configuration.

This module provides the configuration for OpenRouter's Embedding API.
OpenRouter is OpenAI-compatible and supports embeddings via the /v1/embeddings endpoint.

Docs: https://openrouter.ai/docs
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.litellm_core_utils.url_utils import join_api_base
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
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:
        """
        Validate environment and set up headers for OpenRouter API.

        OpenRouter requires:
        - Authorization header with Bearer token
        - HTTP-Referer header (site URL)
        - X-Title header (app name)
        """
        from litellm import get_secret

        # Get OpenRouter-specific headers
        openrouter_site_url: Final = get_secret("OR_SITE_URL") or "https://litellm.ai"
        openrouter_app_name: Final = get_secret("OR_APP_NAME") or "liteLLM"

        openrouter_headers: Final = {
            "HTTP-Referer": openrouter_site_url,
            "X-Title": openrouter_app_name,
            "Content-Type": "application/json",
        }

        # Add Authorization header if api_key is provided
        if api_key:
            openrouter_headers["Authorization"] = f"Bearer {api_key}"

        # Merge with existing headers (user's extra_headers take priority)
        merged_headers: Final = {**openrouter_headers, **headers}

        return merged_headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete URL for OpenRouter Embedding API endpoint.
        """
        return join_api_base(api_base or "https://openrouter.ai/api/v1", "embeddings")

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:
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
        api_key: str | None,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> EmbeddingResponse:
        """
        Transform embedding response from OpenRouter format (OpenAI-compatible).
        """
        logging_obj.post_call(original_response=raw_response.text)

        # OpenRouter returns standard OpenAI-compatible embedding response
        response_json: Final = raw_response.json()

        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list[str]:
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
        non_default_params: Mapping[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        """
        Map OpenAI parameters to OpenRouter format.
        """
        for param, value in non_default_params.items():
            if param in self.get_supported_openai_params(model):
                optional_params[param] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, str] | httpx.Headers
    ) -> OpenRouterException:
        """
        Get the error class for OpenRouter errors.
        """
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
