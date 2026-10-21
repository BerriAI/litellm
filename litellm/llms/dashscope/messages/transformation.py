"""
DashScope Anthropic-compatible Messages transformation config.
"""

from typing import Any  # noqa: TID251  # override below must mirror the legacy base signature

import litellm
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import resolve_dashscope_anthropic_api_base


class DashScopeAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DashScope exposes an Anthropic-compatible Messages API at
    ``{origin}/apps/anthropic`` for every DashScope domain, including
    ``dashscope.aliyuncs.com`` and ``{workspace}.{region}.maas.aliyuncs.com``.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "dashscope"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("DASHSCOPE_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return resolve_dashscope_anthropic_api_base(api_base)

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
        optional_params: dict,  # mutable-ok: mirrors the base get_complete_url signature
        litellm_params: dict,  # mutable-ok: mirrors the base get_complete_url signature
        stream: bool | None = None,
    ) -> str:
        return ensure_api_route(self.get_api_base(api_base=api_base), "v1/messages")
