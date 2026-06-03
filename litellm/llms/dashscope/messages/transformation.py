"""
DashScope Anthropic transformation config - extends AnthropicMessagesConfig for
DashScope's Anthropic-compatible Messages API.
"""

from typing import Any, List, Optional, Tuple

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str

DEFAULT_DASHSCOPE_ANTHROPIC_MESSAGES_API_BASE = "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
_COMPATIBLE_MODE_CHAT_COMPLETIONS_SUFFIX = "/compatible-mode/v1/chat/completions"
_COMPATIBLE_MODE_SUFFIX = "/compatible-mode/v1"
_ANTHROPIC_MESSAGES_SUFFIX = "/apps/anthropic"
_MESSAGES_SUFFIX = "/v1/messages"


class DashScopeAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DashScope Anthropic configuration that extends AnthropicMessagesConfig.

    DashScope (Alibaba Cloud) exposes an Anthropic-compatible Messages API at:
    - China:         https://dashscope.aliyuncs.com/apps/anthropic/v1/messages
    - International: https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages

    The OpenAI-compatible base URLs (``.../compatible-mode/v1`` and
    ``.../compatible-mode/v1/chat/completions``) are also recognized and
    rewritten to the Anthropic Messages endpoint, since callers commonly reuse
    the chat-completions base URL.

    Supported models: any DashScope-hosted model that is exposed via the
    Anthropic Messages endpoint (e.g. ``qwen3-max``, ``qwen-plus``,
    ``qwen-max``).
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "dashscope"

    @staticmethod
    def _get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("DASHSCOPE_API_KEY")

    @staticmethod
    def _get_anthropic_messages_api_base(api_base: Optional[str] = None) -> str:
        if not api_base:
            return DEFAULT_DASHSCOPE_ANTHROPIC_MESSAGES_API_BASE

        base_url = api_base.rstrip("/")
        if base_url.endswith(_MESSAGES_SUFFIX):
            return base_url
        if base_url.endswith(_COMPATIBLE_MODE_CHAT_COMPLETIONS_SUFFIX):
            root = base_url[: -len(_COMPATIBLE_MODE_CHAT_COMPLETIONS_SUFFIX)]
            return f"{root}{_ANTHROPIC_MESSAGES_SUFFIX}{_MESSAGES_SUFFIX}"
        if base_url.endswith(_COMPATIBLE_MODE_SUFFIX):
            root = base_url[: -len(_COMPATIBLE_MODE_SUFFIX)]
            return f"{root}{_ANTHROPIC_MESSAGES_SUFFIX}{_MESSAGES_SUFFIX}"
        if base_url.endswith(f"{_ANTHROPIC_MESSAGES_SUFFIX}/v1"):
            return f"{base_url}/messages"
        if base_url.endswith(_ANTHROPIC_MESSAGES_SUFFIX):
            return f"{base_url}{_MESSAGES_SUFFIX}"

        return f"{base_url}{_MESSAGES_SUFFIX}"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return self._get_anthropic_messages_api_base(api_base=api_base)

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        dynamic_api_key = self._get_api_key(api_key=api_key)
        if dynamic_api_key and "x-api-key" not in headers:
            headers["x-api-key"] = dynamic_api_key
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        # DashScope's Anthropic-compatible endpoint does not accept Anthropic
        # version/beta headers. Keep this provider-specific so other Anthropic
        # Messages providers retain their existing behavior.
        headers.pop("anthropic-version", None)
        headers.pop("anthropic-beta", None)
        return headers, api_base
