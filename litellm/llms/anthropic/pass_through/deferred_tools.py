from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from litellm.litellm_core_utils.core_helpers import as_str_mapping

DeferredToolsMode: TypeAlias = Literal["passthrough", "native", "emulate", "drop"]
CLAUDE_CODE_TOOL_SEARCH_NAME: Final = "ToolSearch"
CLIENT_TOOL_SEARCH_PROVIDERS: Final = frozenset({"azure", "fireworks_ai", "openai"})


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


def is_deferred(tool: Mapping[str, object]) -> bool:
    return tool.get("defer_loading") is True


def client_tool_search_name(tools: Iterable[object] | None) -> str | None:
    names: Final = frozenset((as_str_mapping(tool) or {}).get("name") for tool in tools or ())
    return CLAUDE_CODE_TOOL_SEARCH_NAME if CLAUDE_CODE_TOOL_SEARCH_NAME in names else None


def native_tool_search_name(model_info: object, tools: Iterable[object] | None) -> str | None:
    return client_tool_search_name(tools) if deferred_tools_mode(model_info) == "native" else None
