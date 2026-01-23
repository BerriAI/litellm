"""
Vercel AI Gateway Embedding API Configuration.

This module provides the configuration for Vercel AI Gateway's Embedding API.
Vercel AI Gateway is OpenAI-compatible and supports embeddings via the /v1/embeddings endpoint.

Docs: https://vercel.com/docs/ai-gateway/openai-compat/embeddings
"""

from typing import TYPE_CHECKING, Any, Optional

import httpx

from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import VercelAIGatewayException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VercelAIGatewayEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for Vercel AI Gateway's Embedding API.

    Reference: https://vercel.com/docs/ai-gateway/openai-compat/embeddings
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
        Validate environment and set up headers for Vercel AI Gateway API.

        Vercel AI Gateway requires:
        - Authorization header with Bearer token (API key or OIDC token)
        """
        vercel_headers = {
            "Content-Type": "application/json",
        }

        # Add Authorization header if api_key is provided
        if api_key:
            vercel_headers["Authorization"] = f"Bearer {api_key}"

        # Merge with existing headers (user's extra_headers take priority)
        merged_headers = {**vercel_headers, **headers}

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
        Get the complete URL for Vercel AI Gateway Embedding API endpoint.
        """
        if api_base:
            api_base = api_base.rstrip("/")
        else:
            api_base = (
                get_secret_str("VERCEL_AI_GATEWAY_API_BASE")
                or "https://ai-gateway.vercel.sh/v1"
            )

        return f"{api_base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request to Vercel AI Gateway format (OpenAI-compatible).
        """
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        # Strip 'vercel_ai_gateway/' prefix if present
        if model.startswith("vercel_ai_gateway/"):
            model = model.replace("vercel_ai_gateway/", "", 1)

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
        Transform embedding response from Vercel AI Gateway format (OpenAI-compatible).
        """
        logging_obj.post_call(original_response=raw_response.text)

        # Vercel AI Gateway returns standard OpenAI-compatible embedding response
        response_json = raw_response.json()

        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get list of supported OpenAI parameters for Vercel AI Gateway embeddings.

        Vercel AI Gateway supports the standard OpenAI embeddings parameters
        and auto-maps 'dimensions' to each provider's expected field.
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
        Map OpenAI parameters to Vercel AI Gateway format.
        """
        for param, value in non_default_params.items():
            if param in self.get_supported_openai_params(model):
                optional_params[param] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Any
    ) -> Any:
        """
        Get the error class for Vercel AI Gateway errors.
        """
        return VercelAIGatewayException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
