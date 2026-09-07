"""
Z.AI Anthropic-compatible messages transformation config.
"""

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
        headers: dict[str, str],
        model: str,
        messages: list[dict[str, object]],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:
        dynamic_api_key: Final = self.get_api_key(api_key=api_key)
        header_names: Final = {header_name.lower() for header_name in headers}

        if "x-api-key" not in header_names and "authorization" not in header_names and dynamic_api_key is not None:
            headers["x-api-key"] = dynamic_api_key

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
            custom_llm_provider=self.custom_llm_provider or "zai",
        )

        return headers, api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages"):
            return base_url
        base_url = base_url.removesuffix("/v1/messages")
        base_url = base_url.removesuffix("/v1")
        base_url = base_url.removesuffix("/beta")

        if not base_url.endswith("/anthropic") and "/anthropic/" not in base_url:
            base_url = f"{base_url}/anthropic"

        return f"{base_url}/v1/messages"
