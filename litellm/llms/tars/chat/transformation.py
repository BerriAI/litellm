"""
Support for OpenAI's `/v1/chat/completions` endpoint.

TARS (Tetrate Agent Router Service) is OpenAI-compatible.

Docs: https://router.tetrate.ai
API: https://api.router.tetrate.ai/v1
"""

from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

from ..common_utils import TarsException, TarsModelInfo


class TarsConfig(OpenAIGPTConfig, TarsModelInfo):
    """
    Configuration for TARS (Tetrate Agent Router Service).
    
    TARS is OpenAI-compatible and routes to multiple LLM providers.
    Supports dynamic model fetching from the TARS API.
    """

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return TarsException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            api_base = "https://api.router.tetrate.ai/v1"
        
        endpoint = "chat/completions"
        api_base = api_base.rstrip("/")
        
        if endpoint in api_base:
            result = api_base
        else:
            result = f"{api_base}/{endpoint}"
        
        return result

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Override OpenAIGPTConfig.get_models() to use TARS API instead of OpenAI API.
        """
        # Use TarsModelInfo.get_models() method instead of OpenAIGPTConfig.get_models()
        return TarsModelInfo.get_models(self, api_key=api_key, api_base=api_base)

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        """
        Override OpenAIGPTConfig.get_api_base() to use TARS API base instead of OpenAI API base.
        """
        return TarsModelInfo.get_api_base(api_base)

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Override OpenAIGPTConfig.get_api_key() to use TARS API key instead of OpenAI API key.
        """
        return TarsModelInfo.get_api_key(api_key)

