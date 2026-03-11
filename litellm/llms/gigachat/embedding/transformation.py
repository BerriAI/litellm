"""
GigaChat Embedding Transformation

Transforms OpenAI /v1/embeddings format to GigaChat format.
API Documentation: https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-embeddings
"""

import types
from typing import List, Optional, Tuple, Union

import httpx

from litellm import LlmProviders
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse

from ..authenticator import get_access_token

# GigaChat API endpoint
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"


class GigaChatEmbeddingError(BaseLLMException):
    """GigaChat Embedding API error."""

    pass


class GigaChatEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration class for GigaChat Embeddings API.

    GigaChat embeddings endpoint: POST /api/v1/embeddings
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        """GigaChat embeddings don't support additional parameters."""
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Map OpenAI params to GigaChat format (no special mapping needed)."""
        return optional_params

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Returns provider info for GigaChat.

        Returns:
            Tuple of (custom_llm_provider, api_base, dynamic_api_key)
        """
        api_base = api_base or GIGACHAT_BASE_URL
        return LlmProviders.GIGACHAT.value, api_base, api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Get the complete URL for embeddings endpoint."""
        base = api_base or GIGACHAT_BASE_URL
        return f"{base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI embedding request to GigaChat format.

        GigaChat format:
        {
            "model": "Embeddings",
            "input": ["text1", "text2", ...]
        }
        """
        # Normalize input to list
        if isinstance(input, str):
            input_list: list = [input]
        elif isinstance(input, list):
            input_list = input
        else:
            input_list = [input]

        # Remove gigachat/ prefix from model if present
        if model.startswith("gigachat/"):
            model = model[9:]

        return {
            "model": model,
            "input": input_list,
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
        Transform GigaChat embedding response to OpenAI format.

        GigaChat returns:
        {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [...], "index": 0, "usage": {...}}],
            "model": "Embeddings"
        }
        """
        response_json = raw_response.json()

        # Log response
        logging_obj.post_call(
            input=request_data.get("input"),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )

        # Calculate total tokens from individual embeddings
        total_tokens = 0
        if "data" in response_json:
            for emb in response_json["data"]:
                if "usage" in emb and "prompt_tokens" in emb["usage"]:
                    total_tokens += emb["usage"]["prompt_tokens"]
                # Remove usage from individual embeddings (not part of OpenAI format)
                if "usage" in emb:
                    del emb["usage"]

        # Set overall usage
        response_json["usage"] = {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }

        return EmbeddingResponse(**response_json)

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
        Set up headers with OAuth token for GigaChat.
        """
        # Get access token via OAuth
        access_token = get_access_token(api_key)

        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        return {**default_headers, **headers}

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """Return GigaChat-specific error class."""
        return GigaChatEmbeddingError(
            status_code=status_code,
            message=error_message,
        )
