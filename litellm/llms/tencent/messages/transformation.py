"""
Tencent Anthropic-compatible messages transformation config.

Tencent TokenHub exposes an Anthropic-compatible Messages API endpoint
alongside its standard OpenAI-compatible chat completions endpoint.
"""

from typing import Any, Optional

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str


class TencentAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    Tencent TokenHub exposes an Anthropic-compatible Messages API.

    Unlike the chat completions endpoint (which uses /v1), the Anthropic
    endpoint may use a different base URL. Configure via
    TENCENT_ANTHROPIC_API_BASE or TENCENT_API_BASE.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "tencent"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("TENCENT_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        return (
            api_base
            or get_secret_str("TENCENT_ANTHROPIC_API_BASE")
            or get_secret_str("TENCENT_API_BASE")
            or "https://tokenhub-intl.tencentcloudmaas.com"
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> tuple[dict, Optional[str]]:
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
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages"):
            return base_url

        if base_url.endswith("/v1/chat/completions"):
            base_url = base_url[: -len("/v1/chat/completions")]
        elif base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]

        return f"{base_url}/v1/messages"
