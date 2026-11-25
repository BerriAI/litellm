from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping

DeferredToolsMode: TypeAlias = Literal["passthrough", "native", "emulate", "drop"]
CLAUDE_CODE_TOOL_SEARCH_NAME: Final = "ToolSearch"
CLIENT_TOOL_SEARCH_PROVIDERS: Final = frozenset({"azure", "openai"})
DEFERRAL_REJECTING_CHAT_PROVIDERS: Final = CLIENT_TOOL_SEARCH_PROVIDERS | {"mistral"}


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


def unknown_deferred_tools_refusal(model_info: object) -> str | None:
    setting: Final = deferred_tools_setting(model_info)
    return unknown_deferred_tools_message(setting) if isinstance(setting, UnknownDeferredTools) else None


def refusal_unless_dropped(refusal: str | None, drop_params: bool) -> str | None:
    if refusal is None or not drop_params:
        return refusal
    verbose_logger.warning("drop_params is on, so LiteLLM falls back instead of answering 400: %s", refusal)
    return None


def usable_deferred_tools_mode(model_info: object, provider: str | None) -> DeferredToolsMode:
    mode: Final = deferred_tools_mode(model_info)
    if mode == "native" and provider not in CLIENT_TOOL_SEARCH_PROVIDERS:
        return "drop"
    if mode == "passthrough" and provider in DEFERRAL_REJECTING_CHAT_PROVIDERS:
        return "drop"
    return mode


def unusable_deferred_tools_refusal(model_info: object, provider: str | None) -> str | None:
    mode: Final = deferred_tools_mode(model_info)
    if usable_deferred_tools_mode(model_info, provider) == mode:
        return None
    if mode == "native":
        return (
            f"model_info.deferred_tools native needs a Responses API whose client tool search keeps deferred tools "
            f"out of the prompt, which provider {provider} lacks. Pick emulate, passthrough or drop for this "
            "deployment, or set drop_params to send deferred tools the drop way"
        )
    other_modes: Final = "native or emulate" if provider in CLIENT_TOOL_SEARCH_PROVIDERS else "emulate or drop"
    return (
        f"model_info.deferred_tools passthrough sends deferred tools as Claude Code sent them, which provider "
        f"{provider} rejects. Pick {other_modes} for this deployment, or set drop_params to send deferred tools "
        "the drop way"
    )


def native_messages_deferred_tools_refusal(model_info: object, provider: str | None) -> str | None:
    setting: Final = (as_str_mapping(model_info) or {}).get("deferred_tools")
    if setting is None or setting == "passthrough":
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
