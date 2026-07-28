"""
apiToken.sale Anthropic transformation config - extends AnthropicMessagesConfig for
apiToken.sale's Anthropic-compatible API
"""

from typing import Optional

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str


class ApiTokenMessagesConfig(AnthropicMessagesConfig):
    """
    apiToken.sale Anthropic configuration that extends AnthropicMessagesConfig.
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

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get apiToken.sale API key from environment or parameters.
        """
        return api_key or get_secret_str("APITOKEN_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> str:
        """
        Get apiToken.sale API base URL.
        Defaults to https://api.apitoken.sale/v1/messages
        """
        return api_base or get_secret_str("APITOKEN_API_BASE") or "https://api.apitoken.sale/v1/messages"

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
        Get the complete URL for the apiToken.sale API.
        Override to ensure we use apiToken.sale's endpoint, not Anthropic's.
        """
        base_url = self.get_api_base(api_base=api_base)

        # If the base URL already includes the full path, return it
        if base_url.endswith("/v1/messages"):
            return base_url

        # Otherwise append the messages endpoint
        if base_url.endswith("/"):
            return f"{base_url}v1/messages"
        else:
            return f"{base_url}/v1/messages"
