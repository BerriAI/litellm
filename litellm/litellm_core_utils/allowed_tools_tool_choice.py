from collections.abc import Mapping, Sequence, Set
from typing import Final, Literal, TypeVar

from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.litellm_core_utils.tool_fields import first_validation_error

PROVIDERS_ACCEPTING_ALLOWED_TOOLS: Final = frozenset({"openai", "azure", "litellm_proxy"})


class _AllowedTools(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    mode: Literal["auto", "required"]
    tools: tuple[Mapping[str, JsonValue], ...] = Field(min_length=1)


class ChatAllowedToolChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["allowed_tools"]
    allowed_tools: _AllowedTools


class ResponsesAllowedToolChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["allowed_tools"]
    mode: Literal["auto", "required"]
    tools: tuple[Mapping[str, JsonValue], ...] = Field(min_length=1)


class _AllowedToolChoiceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_choice: ChatAllowedToolChoice | ResponsesAllowedToolChoice


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])
_ToolChoiceT: Final = TypeVar("_ToolChoiceT")


def parse_allowed_tool_choice(tool_choice: object) -> ChatAllowedToolChoice | ResponsesAllowedToolChoice:
    match tool_choice:
        case {"allowed_tools": object()}:
            return ChatAllowedToolChoice.model_validate(tool_choice)
        case _:
            return ResponsesAllowedToolChoice.model_validate(tool_choice)


def _json_choice(choice: ChatAllowedToolChoice | ResponsesAllowedToolChoice) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_python(choice.model_dump(mode="json"))


def _chat_tool(tool: JsonValue) -> JsonValue:
    match tool:
        case {"type": "function", "name": str() as name, **fields}:
            return {**fields, "type": "function", "function": {"name": name}}
        case {"type": "custom", "name": str() as name, **fields}:
            return {**fields, "type": "custom", "custom": {"name": name}}
        case _:
            return tool


def _responses_tool(tool: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    match tool:
        case {"type": "function", "function": {"name": str() as name, **inner_fields}, **fields}:
            return {**fields, **inner_fields, "type": "function", "name": name}
        case {"type": "custom", "custom": {"name": str() as name, **inner_fields}, **fields}:
            return {**fields, **inner_fields, "type": "custom", "name": name}
        case _:
            return tool


def chat_allowed_tool_choice(
    allowed_tool_choice: ChatAllowedToolChoice | ResponsesAllowedToolChoice,
) -> Mapping[str, JsonValue]:
    choice: Final = _json_choice(allowed_tool_choice)
    match choice:
        case {"type": choice_type, "mode": mode, "tools": list() as tools, **allowed_tools_fields}:
            return {
                "type": choice_type,
                "allowed_tools": {**allowed_tools_fields, "mode": mode, "tools": [_chat_tool(tool) for tool in tools]},
            }
        case _:
            return choice


def responses_allowed_tool_choice(
    allowed_tool_choice: ChatAllowedToolChoice | ResponsesAllowedToolChoice,
) -> ToolChoiceAllowedParam:
    allowed_tools: Final = (
        allowed_tool_choice.allowed_tools
        if isinstance(allowed_tool_choice, ChatAllowedToolChoice)
        else allowed_tool_choice
    )
    return {
        "type": "allowed_tools",
        "mode": allowed_tools.mode,
        "tools": [{**_responses_tool(tool)} for tool in allowed_tools.tools],
    }


def is_allowed_tools_choice(tool_choice: object) -> bool:
    match tool_choice:
        case {"type": "allowed_tools"}:
            return True
        case _:
            return False


def _accepts_allowed_tools(custom_llm_provider: str, model: str) -> bool:
    match custom_llm_provider:
        case "azure_ai":
            from litellm.llms.azure_ai.common_utils import azure_ai_routes_to_openai_chat

            return azure_ai_routes_to_openai_chat(model)
        case _:
            return custom_llm_provider in PROVIDERS_ACCEPTING_ALLOWED_TOOLS


def tool_choice_for_provider(
    tool_choice: _ToolChoiceT, *, custom_llm_provider: str, model: str, drop_params: bool
) -> _ToolChoiceT | Mapping[str, JsonValue] | None:
    if not is_allowed_tools_choice(tool_choice):
        return tool_choice
    try:
        allowed_tool_choice: Final = parse_allowed_tool_choice(tool_choice)
    except ValidationError as exc:
        if drop_params:
            return None
        raise BadRequestError(
            message=f"Invalid allowed_tools tool_choice, {first_validation_error(exc, prefix=('tool_choice',))}. To drop it, set `litellm.drop_params=True`",
            model=model,
            llm_provider=custom_llm_provider,
        ) from exc
    if _accepts_allowed_tools(custom_llm_provider, model):
        return dict(chat_allowed_tool_choice(allowed_tool_choice))
    if drop_params:
        return None
    raise UnsupportedParamsError(
        message=f"{custom_llm_provider} does not support an allowed_tools tool_choice, for model={model}. To drop it, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n",
        model=model,
        llm_provider=custom_llm_provider,
    )


def function_or_custom_tool_name(tool: object) -> str | None:
    match tool:
        case (
            {"type": "function", "function": {"name": str() as name}}
            | {"type": "custom", "custom": {"name": str() as name}}
            | {"type": "function" | "custom", "name": str() as name}
        ):
            return name
        case _:
            return None


def _allowed_tools_list(choice: Mapping[str, JsonValue]) -> Sequence[JsonValue]:
    match choice:
        case {"allowed_tools": {"tools": list() as tools}} | {"tools": list() as tools}:
            return tools
        case _:
            return []


def _with_allowed_tools(choice: Mapping[str, JsonValue], tools: Sequence[JsonValue]) -> JsonValue:
    match choice:
        case {"allowed_tools": dict() as allowed_tools}:
            return {**choice, "allowed_tools": {**allowed_tools, "tools": [*tools]}}
        case _:
            return {**choice, "tools": [*tools]}


def pruned_allowed_tool_choice(request: object, removed_tool_names: Set[str]) -> JsonValue | None:
    try:
        choice: Final = _json_choice(_AllowedToolChoiceRequest.model_validate(request).tool_choice)
    except ValidationError:
        return None
    tools: Final = _allowed_tools_list(choice)
    kept: Final = [tool for tool in tools if function_or_custom_tool_name(tool) not in removed_tool_names]
    if len(kept) == len(tools):
        return None
    return _with_allowed_tools(choice, kept) if kept else "none"
