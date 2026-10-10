from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.types.llms.anthropic import AnthropicToolSearchToolRegex
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.responses.main import ResponseToolSearchCall

from .custom_tools import openai_shaped_tool_call_item_id

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
    return tool.get("type") == ANTHROPIC_TOOL_SEARCH_TOOL_TYPE


def is_tool_search_call_name(tool_name: str) -> bool:
    return tool_name == ANTHROPIC_TOOL_SEARCH_TOOL_NAME


def _parsed_arguments(arguments_json: str) -> Mapping[str, object] | None:
    try:
        return _ARGUMENTS_ADAPTER.validate_json(arguments_json or "{}")
    except ValidationError:
        return None


def build_tool_search_call(tool_id: str, arguments_json: str, status: ToolSearchCallStatus) -> ResponseToolSearchCall:
    arguments: Final = _parsed_arguments(arguments_json)
    return ResponseToolSearchCall(
        id=openai_shaped_tool_call_item_id("tool_search_call", tool_id),
        call_id=tool_id,
        arguments={} if arguments is None else arguments,
        execution="server",
        status="incomplete" if arguments is None else status,
        type="tool_search_call",
    )
