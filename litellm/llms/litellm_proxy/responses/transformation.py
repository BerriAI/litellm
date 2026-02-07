"""
Responses API transformation for LiteLLM Proxy provider.

LiteLLM Proxy supports the OpenAI Responses API natively when the underlying model supports it.
This config enables pass-through behavior to the proxy's /v1/responses endpoint.
"""

from typing import Optional

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders


class LiteLLMProxyResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for LiteLLM Proxy Responses API support.
    
    Extends OpenAI's config since the proxy follows OpenAI's API spec,
    but uses LITELLM_PROXY_API_BASE for the base URL.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.LITELLM_PROXY

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the endpoint for LiteLLM Proxy responses API.
        
        Uses LITELLM_PROXY_API_BASE environment variable if api_base is not provided.
        """
        api_base = api_base or get_secret_str("LITELLM_PROXY_API_BASE")
        
        if api_base is None:
            raise ValueError(
                "api_base not set for LiteLLM Proxy responses API. "
                "Set via api_base parameter or LITELLM_PROXY_API_BASE environment variable"
            )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"
