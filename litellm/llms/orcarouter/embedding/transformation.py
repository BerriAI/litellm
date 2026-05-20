"""
OrcaRouter Embedding API Configuration.

OrcaRouter is OpenAI-compatible and exposes embeddings via the `/v1/embeddings`
endpoint. Currently 5 embedding models are routed (3 OpenAI + 2 Google).

Docs: https://docs.orcarouter.ai
"""

from typing import TYPE_CHECKING, Any, Optional

import httpx

import litellm
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import OrcaRouterException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


_DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"


class OrcaRouterEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for OrcaRouter's `/v1/embeddings` endpoint."""

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
        """Set Authorization + attribution headers; user headers take priority."""
        resolved_key = (
            api_key
            or litellm.api_key
            or litellm.orcarouter_key
            or get_secret_str("ORCAROUTER_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "OrcaRouter API key is required. Set ORCAROUTER_API_KEY environment variable or pass api_key parameter."
            )

        orcarouter_headers = {
            "Authorization": f"Bearer {resolved_key}",
            "HTTP-Referer": get_secret_str("ORCAROUTER_SITE_URL")
            or "https://www.orcarouter.ai/",
            "X-Title": get_secret_str("ORCAROUTER_APP_NAME") or "liteLLM",
            "Content-Type": "application/json",
        }

        return {**orcarouter_headers, **headers}

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = (api_base or _DEFAULT_API_BASE).rstrip("/")
        return f"{base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        if isinstance(input, str):
            input = [input]

        # Strip the LiteLLM routing prefix so OrcaRouter sees the native model id.
        if model.startswith("orcarouter/"):
            model = model[len("orcarouter/") :]

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
        logging_obj.post_call(original_response=raw_response.text)

        return convert_to_model_response_object(
            response_object=raw_response.json(),
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list:
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
        supported = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported:
                optional_params[param] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Any
    ) -> Any:
        return OrcaRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
