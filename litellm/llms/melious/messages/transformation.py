"""
Melious Anthropic-compatible messages transformation config.

Melious serves the Anthropic Messages API at https://api.melious.ai/v1/messages
alongside its OpenAI-compatible endpoint, so the Anthropic payload is forwarded
untranslated. It rejects Anthropic's billing-attribution system blocks.
"""

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import MELIOUS_API_BASE, anthropic_messages_url


class MeliousAnthropicMessagesConfig(AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "melious"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def _resolve_api_key(api_key: str | None) -> str | None:
        return api_key or get_secret_str("MELIOUS_API_KEY") or litellm.api_key

    @staticmethod
    def _resolve_api_base(api_base: str | None) -> str:
        return api_base or get_secret_str("MELIOUS_API_BASE") or MELIOUS_API_BASE

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseAnthropicMessagesConfig signature
        model: str,
        messages: list[object],  # mutable-ok: BaseAnthropicMessagesConfig signature
        optional_params: dict[str, object],  # mutable-ok: BaseAnthropicMessagesConfig signature
        litellm_params: dict[str, object],  # mutable-ok: BaseAnthropicMessagesConfig signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: BaseAnthropicMessagesConfig signature
        return super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=self._resolve_api_key(api_key),
            api_base=api_base,
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: BaseAnthropicMessagesConfig signature
        litellm_params: dict[str, object],  # mutable-ok: BaseAnthropicMessagesConfig signature
        stream: bool | None = None,
    ) -> str:
        return anthropic_messages_url(self._resolve_api_base(api_base))
