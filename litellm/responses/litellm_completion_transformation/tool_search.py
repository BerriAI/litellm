from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.prompt_templates.factory import is_anthropic_tool_search_use
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES, AnthropicToolSearchToolRegex
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.responses.main import OutputToolSearchCall
from litellm.types.utils import ChatCompletionMessageToolCall, LlmProviders, Message

from .custom_tools import openai_shaped_tool_call_item_id

__all__ = [
    "ANTHROPIC_TOOL_SEARCH_TOOL_NAME",
    "ANTHROPIC_TOOL_SEARCH_TOOL_TYPE",
    "CompletionRoute",
    "ToolSearchCallStatus",
    "anthropic_tool_search_chat_tool",
    "build_tool_search_call",
    "has_server_tool_search",
    "holds_tool_search_use",
    "is_anthropic_tool_search_chat_tool",
    "is_anthropic_tool_search_use",
    "is_server_tool_search_tool",
    "provider_builds_anthropic_request",
    "without_tool_search_uses",
]

ANTHROPIC_TOOL_SEARCH_TOOL_TYPE: Final = "tool_search_tool_regex_20251119"
ANTHROPIC_TOOL_SEARCH_TOOL_NAME: Final = "tool_search_tool_regex"
ToolSearchCallStatus: TypeAlias = Literal["in_progress", "completed", "incomplete"]

_ARGUMENTS_ADAPTER: Final = TypeAdapter(dict[str, object])


class _ToolSearchFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    execution: str | None = None


def is_server_tool_search_tool(tool: object) -> bool:
    try:
        parsed: Final = _ToolSearchFields.model_validate(tool)
    except ValidationError:
        return False
    return parsed.type == "tool_search" and parsed.execution != "client"


def has_server_tool_search(tools: Sequence[object] | None) -> bool:
    return any(is_server_tool_search_tool(tool) for tool in tools or ())


def anthropic_tool_search_chat_tool() -> AnthropicToolSearchToolRegex:
    return AnthropicToolSearchToolRegex(type=ANTHROPIC_TOOL_SEARCH_TOOL_TYPE, name=ANTHROPIC_TOOL_SEARCH_TOOL_NAME)


def is_anthropic_tool_search_chat_tool(tool: Mapping[str, object]) -> bool:
    return tool.get("type") in ANTHROPIC_TOOL_SEARCH_TOOL_TYPES


def provider_builds_anthropic_request(model: str, custom_llm_provider: str | None) -> bool:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.utils import ProviderConfigManager

    _, provider, _, _ = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    try:
        llm_provider: Final = LlmProviders(provider)
    except ValueError:
        return False
    chat_config: Final = ProviderConfigManager.get_provider_chat_config(model=model, provider=llm_provider)
    return isinstance(chat_config, AnthropicConfig)


class CompletionRoute(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    custom_llm_provider: str | None = None


def _is_tool_search_call(tool_call: object) -> bool:
    return isinstance(tool_call, ChatCompletionMessageToolCall) and is_anthropic_tool_search_use(
        tool_call.id or "", tool_call.function.name or ""
    )


def holds_tool_search_use(message: object) -> bool:
    return isinstance(message, Message) and any(
        _is_tool_search_call(tool_call) for tool_call in message.tool_calls or ()
    )


def without_tool_search_uses(message: Message) -> Message | None:
    tool_calls: Final = message.tool_calls or []
    kept: Final = tuple(tool_call for tool_call in tool_calls if not _is_tool_search_call(tool_call))
    if len(kept) == len(tool_calls):
        return message
    if not kept and not message.content:
        return None
    return message.model_copy(update={"tool_calls": list(kept) or None})


def _parsed_arguments(arguments_json: str) -> Mapping[str, object] | None:
    try:
        return _ARGUMENTS_ADAPTER.validate_json(arguments_json or "{}")
    except ValidationError:
        return None


def build_tool_search_call(tool_id: str, arguments_json: str, status: ToolSearchCallStatus) -> OutputToolSearchCall:
    arguments: Final = _parsed_arguments(arguments_json)
    return OutputToolSearchCall(
        id=openai_shaped_tool_call_item_id("tool_search_call", tool_id),
        call_id=tool_id,
        arguments={} if arguments is None else arguments,
        execution="server",
        status="incomplete" if arguments is None else status,
        type="tool_search_call",
    )
