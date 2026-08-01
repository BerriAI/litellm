"""
apiToken.sale chat transformation config - extends AnthropicConfig for
apiToken.sale's Anthropic-compatible API.

apiToken.sale serves the Anthropic Messages API at https://api.apitoken.sale.
The request/response protocol is identical to Anthropic's; only the API base
URL and the API key environment variable differ.
"""

from typing import Dict, List, Optional

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.apitoken.common_utils import APITOKEN_API_BASE, build_messages_url
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class ApiTokenChatConfig(AnthropicConfig):
    """
    apiToken.sale configuration that extends AnthropicConfig.

    apiToken.sale provides an Anthropic-compatible API at:
    - https://api.apitoken.sale

    Supported models:
    - claude-opus-4-8
    - claude-opus-4-7
    - claude-sonnet-5
    - claude-sonnet-4-6
    - claude-haiku-4-5
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "apitoken"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """Get apiToken.sale API key from environment or parameters."""
        return api_key or get_secret_str("APITOKEN_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        """Get apiToken.sale API base URL. Defaults to https://api.apitoken.sale."""
        return api_base or get_secret_str("APITOKEN_API_BASE") or APITOKEN_API_BASE

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
        Get the complete URL for the apiToken.sale request.
        Override to ensure we use apiToken.sale's endpoint, not Anthropic's.
        """
        return build_messages_url(self.get_api_base(api_base=api_base))

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        """
        Resolve the apiToken.sale key/base, then reuse the Anthropic header logic
        (x-api-key + anthropic-version) unchanged.
        """
        api_key = self.get_api_key(api_key)
        if api_key is None:
            raise litellm.AuthenticationError(
                message=(
                    "Missing apiToken.sale API key - A call is being made to apitoken but no key is set "
                    "either in the environment variables or via params. Please set `APITOKEN_API_KEY` "
                    "in your environment vars"
                ),
                llm_provider="apitoken",
                model=model,
            )
        api_base = self.get_api_base(api_base)
        return super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
