from collections.abc import Mapping
from typing import Final, Literal

from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS: Final = frozenset({"openai", "azure", "litellm_proxy"})

CHAT_ALLOWED_TOOLS_LAYOUT: Final = (
    '{"type": "allowed_tools", "allowed_tools": {"mode": "auto" | "required", "tools": [<at least one tool>]}}'
)


class ChatAllowedTools(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["auto", "required"]
    tools: tuple[Mapping[str, object], ...] = Field(min_length=1)


class ChatAllowedToolsToolChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["allowed_tools"]
    allowed_tools: ChatAllowedTools


_CHAT_ALLOWED_TOOLS_TOOL_CHOICE: Final = TypeAdapter(ChatAllowedToolsToolChoice)


class _NamedTool(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)


_NAMED_TOOL: Final = TypeAdapter(_NamedTool)


def is_allowed_tools_tool_choice(tool_choice: Mapping[str, object] | str | None) -> bool:
    return isinstance(tool_choice, Mapping) and tool_choice.get("type") == "allowed_tools"


def parse_chat_allowed_tools_tool_choice(tool_choice: Mapping[str, object]) -> ChatAllowedToolsToolChoice | None:
    try:
        return _CHAT_ALLOWED_TOOLS_TOOL_CHOICE.validate_python(tool_choice)
    except ValidationError:
        return None


def _responses_allowed_tool(tool: Mapping[str, object]) -> Mapping[str, object] | None:
    tool_type: Final = tool.get("type")
    if tool_type not in ("function", "custom"):
        return tool
    if isinstance(tool.get("name"), str):
        return tool
    try:
        nested: Final = _NAMED_TOOL.validate_python(tool.get(tool_type))
    except ValidationError:
        return None
    return {"type": tool_type, "name": nested.name}


def responses_allowed_tools_tool_choice(choice: ChatAllowedToolsToolChoice) -> ToolChoiceAllowedParam | None:
    """The flat Responses API layout of a chat ``allowed_tools`` choice, or ``None`` when an entry is not one the
    chat layout defines, so the caller sends the original rather than a rewrite that could mean something looser."""
    flattened: Final = tuple(_responses_allowed_tool(tool) for tool in choice.allowed_tools.tools)
    if any(tool is None for tool in flattened):
        return None
    return ToolChoiceAllowedParam(
        type="allowed_tools",
        mode=choice.allowed_tools.mode,
        tools=[dict(tool) for tool in flattened if tool is not None],
    )
