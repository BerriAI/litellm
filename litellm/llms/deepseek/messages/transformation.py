"""
DeepSeek Anthropic-compatible messages transformation config.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.common_utils import AnthropicError, AnthropicModelInfo
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ModelInfoLookupMiss, lookup_catalog_model_info

_TOOL_LIST: Final = TypeAdapter(list[object])
_TOOL_FIELDS: Final = TypeAdapter(dict[object, object])
_FIELDS: Final = TypeAdapter(dict[str, object])
DEEPSEEK_EFFORT_LEVELS: Final = ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")


@dataclass(frozen=True, slots=True)
class _InvalidEffort:
    effort: object


def _without(params: Mapping[str, object], key: str) -> Mapping[str, object]:
    return {name: value for name, value in params.items() if name != key}


def _with_output_config_effort(
    params: Mapping[str, object], alias_effort: str | None, drop_params: bool
) -> Mapping[str, object] | _InvalidEffort:
    output_config: Final = params.get("output_config")
    if output_config is not None and not isinstance(output_config, dict):
        return params
    fields: Final = _FIELDS.validate_python(output_config or {})
    effort: Final = fields.get("effort", alias_effort)
    if effort is None:
        return params
    if effort in DEEPSEEK_EFFORT_LEVELS:
        return {**params, "output_config": {**fields, "effort": effort}}
    if not drop_params:
        return _InvalidEffort(effort=effort)
    remaining: Final = _without(fields, "effort")
    return {**_without(params, "output_config"), **({"output_config": remaining} if remaining else {})}


def _deepseek_effort_params(params: Mapping[str, object], drop_params: bool) -> Mapping[str, object] | _InvalidEffort:
    alias: Final = params.get("reasoning_effort")
    if not isinstance(alias, str):
        return _with_output_config_effort(params, None, drop_params)
    if alias in DEEPSEEK_EFFORT_LEVELS:
        return _with_output_config_effort(_without(params, "reasoning_effort"), alias, drop_params)
    if not drop_params:
        return _InvalidEffort(effort=alias)
    return _with_output_config_effort(_without(params, "reasoning_effort"), None, drop_params)


def _listed_without_reasoning(model: str, custom_llm_provider: str) -> bool:
    if AnthropicModelInfo.supports_model_capability(model, "supports_reasoning", custom_llm_provider):
        return False
    return not isinstance(lookup_catalog_model_info(model, custom_llm_provider), ModelInfoLookupMiss)


def _deepseek_request_params(
    model: str, custom_llm_provider: str, params: Mapping[str, object], drop_params: bool
) -> Mapping[str, object] | _InvalidEffort:
    if _listed_without_reasoning(model, custom_llm_provider):
        return params
    return _deepseek_effort_params(params, drop_params)


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


class DeepSeekAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DeepSeek exposes an Anthropic-compatible Messages API at
    https://api.deepseek.com/anthropic.

    It accepts the native Anthropic Messages conversation shape, including
    thinking blocks in assistant history, but rejects Anthropic's explicit
    custom-tool discriminator (`{"type": "custom"}`).

    Its reasoning models read `thinking` and `output_config.effort` natively
    (ignoring `budget_tokens`) and accept any `temperature`, so for them and
    for models the cost map does not list yet the Claude-specific effort and
    temperature reshaping is skipped, the `reasoning_effort` alias becomes
    `output_config.effort`, and a level DeepSeek does not accept is a 400, or
    dropped under `drop_params`. Models the cost map lists without reasoning
    keep the base handling unchanged.
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
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages") and "/anthropic/" in base_url:
            return base_url
        base_url = base_url.removesuffix("/v1/messages")
        base_url = base_url.removesuffix("/v1")
        base_url = base_url.removesuffix("/beta")

        if not base_url.endswith("/anthropic") and "/anthropic/" not in base_url:
            base_url = join_api_base(base_url, "anthropic")

        return join_api_base(base_url, "v1/messages")

    @staticmethod
    def _translate_adaptive_effort_for_non_adaptive_model(
        model: str,
        optional_params: dict[str, object],  # mutable-ok: the base hook edits the request params in place
        max_tokens: int | None,
        custom_llm_provider: str,
    ) -> None:
        if not _listed_without_reasoning(model, custom_llm_provider):
            return None
        return AnthropicMessagesConfig._translate_adaptive_effort_for_non_adaptive_model(  # pyright: ignore[reportUnknownMemberType]  # the base hook takes a bare dict
            model=model,
            optional_params=optional_params,
            max_tokens=max_tokens,
            custom_llm_provider=custom_llm_provider,
        )

    @staticmethod
    def _drop_incompatible_temperature_for_thinking(
        model: str,
        optional_params: dict[str, object],  # mutable-ok: the base hook edits the request params in place
        custom_llm_provider: str,
    ) -> None:
        if not _listed_without_reasoning(model, custom_llm_provider):
            return None
        return AnthropicMessagesConfig._drop_incompatible_temperature_for_thinking(  # pyright: ignore[reportUnknownMemberType]  # the base hook takes a bare dict
            model=model,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
        )

    @staticmethod
    def _sanitize_tools_for_deepseek(tools: object) -> object:
        if not isinstance(tools, list):
            return tools

        return [_sanitized_tool(tool) for tool in _TOOL_LIST.validate_python(tools)]

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        effort_params: Final = _deepseek_request_params(
            model,
            self._resolved_provider,
            _FIELDS.validate_python(anthropic_messages_optional_request_params),
            drops_unsupported_params(litellm_params.drop_params),
        )
        if isinstance(effort_params, _InvalidEffort):
            raise AnthropicError(
                message=(
                    f"Invalid effort: {effort_params.effort!r}. DeepSeek accepts one of: "
                    f"{', '.join(repr(level) for level in DEEPSEEK_EFFORT_LEVELS)}"
                ),
                status_code=400,
            )
        anthropic_messages_request: Final = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=dict(effort_params),
            litellm_params=litellm_params,
            headers=headers,
        )
        if "tools" in anthropic_messages_request:
            anthropic_messages_request["tools"] = self._sanitize_tools_for_deepseek(anthropic_messages_request["tools"])
        return anthropic_messages_request
