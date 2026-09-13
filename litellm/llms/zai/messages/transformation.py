"""
Z.AI Anthropic-compatible messages transformation config.
"""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str


class ZAIAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    Z.AI exposes an Anthropic-compatible Messages API at
    https://api.z.ai/api/anthropic (see
    https://docs.z.ai/guides/llm/glm-5.3).

    The endpoint accepts the native Anthropic Messages conversation shape
    and authenticates with the Z.AI API key sent as the Anthropic
    ``x-api-key`` header.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("ZAI_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or "https://api.z.ai/api/anthropic"

    def validate_anthropic_messages_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: list[Mapping[str, object]],  # mutable-ok: matches the pass-through handler's message list contract
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: the handler owns and mutates the returned headers dict
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
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        raw_base_url: Final = self.get_api_base(api_base=api_base).rstrip("/")
        root_url: Final = raw_base_url.removesuffix("/v1/messages").removesuffix("/v1").removesuffix("/beta")

        if root_url.endswith("/anthropic") or "/anthropic/" in root_url:
            return f"{root_url}/v1/messages"
        return f"{root_url}/anthropic/v1/messages"
