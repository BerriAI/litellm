"""
Tencent Anthropic-compatible messages transformation config.

Tencent TokenHub exposes an Anthropic-compatible Messages API endpoint
alongside its standard OpenAI-compatible chat completions endpoint.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import litellm
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str, normalize_nonempty_secret_str


class TencentAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    Tencent TokenHub exposes an Anthropic-compatible Messages API.

    Unlike the chat completions endpoint (which uses /v1), the Anthropic
    endpoint may use a different base URL. Configure via
    TENCENT_ANTHROPIC_API_BASE or TENCENT_API_BASE.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "tencent"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return (
            normalize_nonempty_secret_str(api_key)
            or normalize_nonempty_secret_str(get_secret_str("TENCENT_API_KEY"))
            or normalize_nonempty_secret_str(litellm.api_key)
        )

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return (
            api_base
            or get_secret_str("TENCENT_ANTHROPIC_API_BASE")
            or get_secret_str("TENCENT_API_BASE")
            or "https://tokenhub-intl.tencentcloudmaas.com"
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, object],  # mutable-ok: mirrors the legacy base override signature
        model: str,
        messages: Sequence[object],
        optional_params: dict[str, object],  # mutable-ok: mirrors the legacy base override signature
        litellm_params: dict[str, object],  # mutable-ok: mirrors the legacy base override signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:  # mutable-ok: mirrors the legacy base override signature
        resolved_api_key: Final = self.get_api_key(api_key=api_key)
        if resolved_api_key is None:
            present_headers: Final = {name.lower() for name in headers}
            if "x-api-key" not in present_headers and "authorization" not in present_headers:
                raise AuthenticationError(
                    message=(
                        "Missing Tencent API key. Set TENCENT_API_KEY or pass api_key explicitly. "
                        "Anthropic credentials are never sent to a Tencent endpoint."
                    ),
                    llm_provider="tencent",
                    model=model,
                )
        return super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=resolved_api_key,
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
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages"):
            return base_url

        if base_url.endswith("/v1/chat/completions"):
            base_url = base_url[: -len("/v1/chat/completions")]
        elif base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]

        return join_api_base(base_url, "v1/messages")
