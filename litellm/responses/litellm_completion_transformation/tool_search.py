from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import is_anthropic_tool_search_use
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES, AnthropicToolSearchToolRegex
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.responses.main import OutputToolSearchCall
from litellm.types.utils import LlmProviders, Message

from .custom_tools import openai_shaped_tool_call_item_id

__all__ = [
    "ANTHROPIC_TOOL_SEARCH_TOOL_NAME",
    "ANTHROPIC_TOOL_SEARCH_TOOL_TYPE",
    "CompletionRoute",
    "ToolSearchCallStatus",
    "answers_tool_search",
    "anthropic_tool_search_chat_tool",
    "build_tool_search_call",
    "has_server_tool_search",
    "holds_tool_search_use",
    "is_anthropic_tool_search_chat_tool",
    "is_anthropic_tool_search_use",
    "is_server_tool_search_tool",
    "provider_builds_anthropic_request",
    "tool_search_call_ids",
    "without_tool_search_calls",
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
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
    from litellm.utils import ProviderConfigManager

    _, provider, _, _ = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    try:
        llm_provider: Final = LlmProviders(provider)
    except ValueError:
        return False
    chat_config: Final = ProviderConfigManager.get_provider_chat_config(model=model, provider=llm_provider)
    return isinstance(chat_config, AnthropicConfig) and not isinstance(chat_config, OpenAIGPTConfig)


class CompletionRoute(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    custom_llm_provider: str | None = None


class _ToolCallFunctionFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    name: str | None = None


class _ToolCallFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: str | None = None
    function: _ToolCallFunctionFields | None = None


class _SessionMessageFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    role: str | None = None
    tool_calls: tuple[object, ...] | None = None
    tool_call_id: str | None = None


def _session_message_fields(message: object) -> _SessionMessageFields:
    try:
        return _SessionMessageFields.model_validate(message)
    except ValidationError:
        return _SessionMessageFields()


def _tool_search_call_id(tool_call: object) -> str | None:
    try:
        parsed: Final = _ToolCallFields.model_validate(tool_call)
    except ValidationError:
        return None
    tool_id: Final = parsed.id or ""
    tool_name: Final = (parsed.function.name if parsed.function is not None else None) or ""
    return tool_id if is_anthropic_tool_search_use(tool_id, tool_name) else None


def _is_tool_search_call(tool_call: object) -> bool:
    return _tool_search_call_id(tool_call) is not None


def _assistant_tool_calls(message: object) -> tuple[object, ...]:
    fields: Final = _session_message_fields(message)
    return (fields.tool_calls or ()) if fields.role == "assistant" else ()


def holds_tool_search_use(message: object) -> bool:
    return any(_is_tool_search_call(tool_call) for tool_call in _assistant_tool_calls(message))


def _tool_search_call_ids_of(messages: Iterable[object]) -> Iterator[str]:
    for message in messages:
        for tool_call in _assistant_tool_calls(message):
            if (tool_id := _tool_search_call_id(tool_call)) is not None:
                yield tool_id


def tool_search_call_ids(messages: Iterable[object]) -> frozenset[str]:
    return frozenset(_tool_search_call_ids_of(messages))


def answers_tool_search(message: object, search_call_ids: Set[str]) -> bool:
    fields: Final = _session_message_fields(message)
    return fields.role == "tool" and fields.tool_call_id in search_call_ids


def without_tool_search_calls(message: object) -> tuple[object, ...]:
    return tuple(tool_call for tool_call in _assistant_tool_calls(message) if not _is_tool_search_call(tool_call))


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
