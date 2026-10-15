"""
MiniMax Anthropic transformation config - extends AnthropicConfig for MiniMax's Anthropic-compatible API
"""

from typing import Any  # noqa: TID251  # override below must mirror the legacy base signature

import litellm
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.pass_through.messages.transformation import (
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

    def validate_anthropic_messages_environment(
        self,
        headers: dict,  # mutable-ok: mirrors the legacy base override signature
        model: str,
        messages: list[Any],  # mutable-ok: mirrors the legacy base override signature
        optional_params: dict,  # mutable-ok: mirrors the legacy base override signature
        litellm_params: dict,  # mutable-ok: mirrors the legacy base override signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:  # mutable-ok: mirrors the legacy base override signature
        return super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=self.get_api_key(api_key=api_key),
            api_base=api_base,
        )

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
        return join_api_base(self.get_api_base(api_base=api_base), "v1/messages")
