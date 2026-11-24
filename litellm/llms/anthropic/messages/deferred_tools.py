from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from litellm.litellm_core_utils.core_helpers import as_str_mapping

DeferredToolsMode: TypeAlias = Literal["passthrough", "native", "emulate", "drop"]
CLAUDE_CODE_TOOL_SEARCH_NAME: Final = "ToolSearch"
CLIENT_TOOL_SEARCH_PROVIDERS: Final = frozenset({"azure", "fireworks_ai", "openai"})
DEFERRAL_NEEDS_TOOL_SEARCH_PROVIDERS: Final = frozenset({"openai"})


@dataclass(frozen=True, slots=True)
class UnknownDeferredTools:
    value: object


def deferred_tools_setting(model_info: object) -> DeferredToolsMode | UnknownDeferredTools:
    match (as_str_mapping(model_info) or {}).get("deferred_tools"):
        case None:
            return "drop"
        case "passthrough":
            return "passthrough"
        case "native":
            return "native"
        case "emulate":
            return "emulate"
        case "drop":
            return "drop"
        case value:
            return UnknownDeferredTools(value)


def deferred_tools_mode(model_info: object) -> DeferredToolsMode:
    setting: Final = deferred_tools_setting(model_info)
    return "drop" if isinstance(setting, UnknownDeferredTools) else setting


def unknown_deferred_tools_message(setting: UnknownDeferredTools) -> str:
    return (
        f"model_info.deferred_tools must be one of passthrough, native, emulate, drop, got {setting.value!r}. "
        "Fix the deployment, or set drop_params to send deferred tools the default drop way"
    )


def unknown_deferred_tools_refusal(model_info: object, drop_params: bool) -> str | None:
    setting: Final = deferred_tools_setting(model_info)
    if not isinstance(setting, UnknownDeferredTools) or drop_params:
        return None
    return unknown_deferred_tools_message(setting)


def usable_deferred_tools_mode(model_info: object, provider: str | None) -> DeferredToolsMode:
    mode: Final = deferred_tools_mode(model_info)
    if mode == "native" and provider not in CLIENT_TOOL_SEARCH_PROVIDERS:
        return "drop"
    if mode == "passthrough" and provider in DEFERRAL_NEEDS_TOOL_SEARCH_PROVIDERS:
        return "drop"
    return mode


def unusable_deferred_tools_refusal(model_info: object, provider: str | None, drop_params: bool) -> str | None:
    mode: Final = deferred_tools_mode(model_info)
    if drop_params or usable_deferred_tools_mode(model_info, provider) == mode:
        return None
    if mode == "native":
        return (
            f"model_info.deferred_tools native needs a Responses API with client tool search, which provider "
            f"{provider} lacks. Pick emulate, passthrough or drop for this deployment, or set drop_params to send "
            "deferred tools the drop way"
        )
    return (
        f"model_info.deferred_tools passthrough sends deferred tools as Claude Code sent them, which provider "
        f"{provider} rejects without its own tool search. Pick native or emulate for this deployment, or set "
        "drop_params to send deferred tools the drop way"
    )


def native_messages_deferred_tools_refusal(model_info: object, provider: str | None, drop_params: bool) -> str | None:
    setting: Final = (as_str_mapping(model_info) or {}).get("deferred_tools")
    if setting is None or setting == "passthrough" or drop_params:
        return None
    return (
        f"model_info.deferred_tools {setting} applies only where LiteLLM translates /v1/messages to another API. "
        f"Provider {provider} takes /v1/messages natively and gets deferred tools as sent, so remove the setting or "
        "set it to passthrough, or set drop_params to ignore it"
    )


def is_deferred(tool: Mapping[str, object]) -> bool:
    return tool.get("defer_loading") is True


def client_tool_search_name(tools: Iterable[object] | None) -> str | None:
    names: Final = frozenset((as_str_mapping(tool) or {}).get("name") for tool in tools or ())
    return CLAUDE_CODE_TOOL_SEARCH_NAME if CLAUDE_CODE_TOOL_SEARCH_NAME in names else None


def native_tool_search_name(model_info: object, provider: str | None, tools: Iterable[object] | None) -> str | None:
    return client_tool_search_name(tools) if usable_deferred_tools_mode(model_info, provider) == "native" else None
