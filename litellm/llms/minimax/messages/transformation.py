"""
MiniMax Anthropic transformation config - extends AnthropicConfig for MiniMax's Anthropic-compatible API
"""

from urllib.parse import urlsplit, urlunsplit

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str


class MinimaxMessagesConfig(AnthropicMessagesConfig):
    """
    MiniMax Anthropic configuration that extends AnthropicConfig.
    MiniMax provides an Anthropic-compatible API at:
    - International: https://api.minimax.io/anthropic
    - China: https://api.minimaxi.com/anthropic

    Supported models:
    - MiniMax-M2.1
    - MiniMax-M2.1-lightning
    - MiniMax-M2
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "minimax"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        """
        Get MiniMax API key from environment or parameters.
        """
        return api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str:
        """
        Get MiniMax API base URL.
        Defaults to international endpoint: https://api.minimax.io/anthropic
        For China, set to: https://api.minimaxi.com/anthropic
        """
        return api_base or get_secret_str("MINIMAX_API_BASE") or "https://api.minimax.io/anthropic/v1/messages"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete URL for MiniMax API.
        Override to ensure we use MiniMax's endpoint, not Anthropic's.
        """
        base_url = self.get_api_base(api_base=api_base)
        parsed_url = urlsplit(base_url)
        path_parts = tuple(part for part in parsed_url.path.split("/") if part)
        base_path_parts = (
            path_parts[:-2]
            if path_parts[-2:] == ("v1", "messages")
            else path_parts[:-1]
            if path_parts[-1:] in (("v1",), ("messages",))
            else path_parts
        )
        provider_path_parts = (
            base_path_parts if base_path_parts[-1:] == ("anthropic",) else (*base_path_parts, "anthropic")
        )
        path = "/" + "/".join((*provider_path_parts, "v1", "messages"))
        return urlunsplit(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                path,
                parsed_url.query,
                parsed_url.fragment,
            )
        )
