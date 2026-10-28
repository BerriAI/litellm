from collections.abc import Mapping
from typing import Final, Literal, TypeAlias

from openai.types.chat.chat_completion_allowed_tool_choice_param import ChatCompletionAllowedToolChoiceParam
from openai.types.chat.chat_completion_allowed_tools_param import ChatCompletionAllowedToolsParam
from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from pydantic import BaseModel, ConfigDict, TypeAdapter

PROVIDERS_ACCEPTING_ALLOWED_TOOLS: Final = frozenset({"openai", "azure", "litellm_proxy"})


class _AllowedTools(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["auto", "required"]
    tools: tuple[Mapping[str, object], ...]


class _ChatAllowedToolChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["allowed_tools"]
    allowed_tools: _AllowedTools


class _ResponsesAllowedToolChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["allowed_tools"]
    mode: Literal["auto", "required"]
    tools: tuple[Mapping[str, object], ...]


AllowedToolChoice: TypeAlias = _ChatAllowedToolChoice | _ResponsesAllowedToolChoice

ALLOWED_TOOL_CHOICE_ADAPTER: Final[TypeAdapter[AllowedToolChoice]] = TypeAdapter(AllowedToolChoice)


def _allowed_tools(choice: AllowedToolChoice) -> _AllowedTools:
    match choice:
        case _ChatAllowedToolChoice(allowed_tools=allowed_tools):
            return allowed_tools
        case _ResponsesAllowedToolChoice(mode=mode, tools=tools):
            return _AllowedTools(mode=mode, tools=tools)


def _chat_tool(tool: Mapping[str, object]) -> Mapping[str, object]:
    match tool:
        case {"type": "function", "name": str() as function_name}:
            return {"type": "function", "function": {"name": function_name}}
        case {"type": "custom", "name": str() as custom_name}:
            return {"type": "custom", "custom": {"name": custom_name}}
        case _:
            return tool


def _responses_tool(tool: Mapping[str, object]) -> Mapping[str, object]:
    match tool:
        case {"type": "function", "function": {"name": str() as function_name}}:
            return {"type": "function", "name": function_name}
        case {"type": "custom", "custom": {"name": str() as custom_name}}:
            return {"type": "custom", "name": custom_name}
        case _:
            return tool


def to_chat_allowed_tool_choice(choice: AllowedToolChoice) -> ChatCompletionAllowedToolChoiceParam:
    allowed_tools: Final = _allowed_tools(choice)
    return ChatCompletionAllowedToolChoiceParam(
        type="allowed_tools",
        allowed_tools=ChatCompletionAllowedToolsParam(
            mode=allowed_tools.mode, tools=[dict(_chat_tool(tool)) for tool in allowed_tools.tools]
        ),
    )


def to_responses_allowed_tool_choice(choice: AllowedToolChoice) -> ToolChoiceAllowedParam:
    allowed_tools: Final = _allowed_tools(choice)
    return ToolChoiceAllowedParam(
        type="allowed_tools",
        mode=allowed_tools.mode,
        tools=[dict(_responses_tool(tool)) for tool in allowed_tools.tools],
    )
