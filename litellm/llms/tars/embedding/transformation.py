"""
Support for OpenAI's `/v1/embeddings` endpoint.

TARS (Tetrate Agent Router Service) is OpenAI-compatible for embeddings.

Docs: https://router.tetrate.ai
API: https://api.router.tetrate.ai/v1
"""

from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.embedding.transformation import OpenAIEmbeddingConfig
from litellm.secret_managers.main import get_secret_str

from ..common_utils import TarsException


class TarsEmbeddingConfig(OpenAIEmbeddingConfig):
    """
    Configuration for TARS embeddings.
    
    TARS supports embeddings through OpenAI-compatible API.
    """

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return TarsException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        """
        Get TARS API base URL from parameter or environment variable.
        Override to use TARS-specific defaults instead of OpenAI defaults.
        """
        return api_base or get_secret_str("TARS_API_BASE") or "https://api.router.tetrate.ai/v1"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get TARS API key from parameter or environment variable.
        Override to use TARS-specific API key instead of OpenAI key.
        """
        return api_key or get_secret_str("TARS_API_KEY")

