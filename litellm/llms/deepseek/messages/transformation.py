"""
DeepSeek Anthropic-compatible messages transformation config.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

_TOOL_LIST: Final = TypeAdapter(list[object])
_REQUEST_FIELDS: Final = TypeAdapter(dict[str, object])
_TOOL_FIELDS: Final = TypeAdapter(dict[object, object])


def _custom_tool_without_type(tool: object) -> Mapping[object, object] | None:
    if not isinstance(tool, dict):
        return None
    fields: Final = _TOOL_FIELDS.validate_python(tool)
    if fields.get("type") == "custom":
        fields.pop("type", None)
        return fields
    return None


def _sanitized_tool(tool: object) -> object:
    stripped: Final = _custom_tool_without_type(tool)
    return tool if stripped is None else stripped


def sanitize_tools_for_deepseek(tools: Sequence[object]) -> Sequence[object]:
    return [_sanitized_tool(tool) for tool in tools]


def deepseek_anthropic_base(api_base: str | None) -> str:
    configured: Final = DeepSeekAnthropicMessagesConfig.get_api_base(api_base=api_base).rstrip("/")
    if configured.endswith("/v1/messages") and "/anthropic/" in configured:
        return configured.removesuffix("/v1/messages")
    stripped: Final = configured.removesuffix("/v1/messages").removesuffix("/v1").removesuffix("/beta")
    if stripped.endswith("/anthropic") or "/anthropic/" in stripped:
        return stripped
    return join_api_base(stripped, "anthropic")


class DeepSeekAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DeepSeek exposes an Anthropic-compatible Messages API at
    https://api.deepseek.com/anthropic.

    It accepts the native Anthropic Messages conversation shape, including
    thinking blocks in assistant history, but rejects Anthropic's explicit
    custom-tool discriminator (`{"type": "custom"}`).
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "deepseek"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("DEEPSEEK_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return (
            api_base
            or get_secret_str("DEEPSEEK_ANTHROPIC_API_BASE")
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/anthropic"
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: Sequence[object],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        dynamic_api_key: Final = self.get_api_key(api_key=api_key)

        if "x-api-key" not in headers and "authorization" not in headers and dynamic_api_key is not None:
            headers["x-api-key"] = dynamic_api_key

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
            custom_llm_provider=self.custom_llm_provider or "deepseek",
            messages=messages,
        )

        return headers, api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return join_api_base(deepseek_anthropic_base(api_base), "v1/messages")

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        anthropic_messages_request: Final = _REQUEST_FIELDS.validate_python(
            super().transform_anthropic_messages_request(
                model=model,
                messages=messages,
                anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        )
        tools: Final = anthropic_messages_request.get("tools")
        if not isinstance(tools, list):
            return anthropic_messages_request
        sanitized: Final = {
            **anthropic_messages_request,
            "tools": sanitize_tools_for_deepseek(_TOOL_LIST.validate_python(tools)),
        }
        return sanitized
