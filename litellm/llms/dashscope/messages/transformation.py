"""
DashScope Anthropic-compatible Messages transformation config.
"""

from typing import ClassVar, Final

import litellm
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str, normalize_nonempty_secret_str

from ..common_utils import (
    dashscope_base_selects_anthropic_messages,
    resolve_dashscope_anthropic_api_base,
)


class DashScopeAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DashScope exposes an Anthropic-compatible Messages API at
    ``{origin}/apps/anthropic`` for every DashScope domain, including
    ``dashscope.aliyuncs.com`` and ``{workspace}.{region}.maas.aliyuncs.com``.
    """

    _workload_identity_eligible: ClassVar[bool] = False

    @property
    def custom_llm_provider(self) -> str | None:
        return "dashscope"

    def should_strip_billing_metadata(self) -> bool:
        return True

    def handles_web_search_natively(self) -> bool:
        return False

    def serves_api_base(self, model: str, api_base: str | None) -> bool:
        return dashscope_base_selects_anthropic_messages(api_base)

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return (
            normalize_nonempty_secret_str(api_key)
            or normalize_nonempty_secret_str(get_secret_str("DASHSCOPE_API_KEY"))
            or normalize_nonempty_secret_str(litellm.api_key)
        )

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return resolve_dashscope_anthropic_api_base(api_base)

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors the legacy base override signature
        model: str,
        messages: list[object],  # mutable-ok: mirrors the legacy base override signature
        optional_params: dict[str, object],  # mutable-ok: mirrors the legacy base override signature
        litellm_params: dict[str, object],  # mutable-ok: mirrors the legacy base override signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: mirrors the base signature
        resolved_api_key: Final = self.get_api_key(api_key=api_key)
        if resolved_api_key is None:
            present_headers: Final = {name.lower() for name in headers}
            if "x-api-key" not in present_headers and "authorization" not in present_headers:
                raise AuthenticationError(
                    message=(
                        "Missing DashScope API key. Set DASHSCOPE_API_KEY or pass api_key explicitly. "
                        "Anthropic credentials are never sent to a DashScope endpoint."
                    ),
                    llm_provider="dashscope",
                    model=model,
                )
        return super().validate_anthropic_messages_environment(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportReturnType]  # base signature uses an unparameterized dict
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
        optional_params: dict[str, object],  # mutable-ok: mirrors the base get_complete_url signature
        litellm_params: dict[str, object],  # mutable-ok: mirrors the base get_complete_url signature
        stream: bool | None = None,
    ) -> str:
        return ensure_api_route(self.get_api_base(api_base=api_base), "v1/messages")
