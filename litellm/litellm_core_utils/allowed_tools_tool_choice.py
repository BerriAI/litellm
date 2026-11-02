from collections.abc import Mapping
from typing import Final, Literal

from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError

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


_SELECTOR: Final = TypeAdapter(dict[str, object])


class _ToolChoiceType(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


_TOOL_CHOICE_TYPE: Final = TypeAdapter(_ToolChoiceType)


def is_allowed_tools_tool_choice(tool_choice: object) -> bool:
    try:
        return _TOOL_CHOICE_TYPE.validate_python(tool_choice).type == "allowed_tools"
    except ValidationError:
        return False


def raise_if_allowed_tools_tool_choice_unsupported(
    tool_choice: object,
    custom_llm_provider: str,
    model: str,
    *,
    bridges_to_interactions_api: bool = False,
) -> None:
    """Providers outside the set would silently send the request with no tool restriction at all, so the call stops
    here with a 400. ``drop_params`` is deliberately not consulted: dropping a restriction widens what the model may call."""
    if (
        bridges_to_interactions_api
        or not is_allowed_tools_tool_choice(tool_choice)
        or custom_llm_provider in ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS
    ):
        return
    raise BadRequestError(
        message=(
            "tool_choice type 'allowed_tools' is only supported for "
            f"{', '.join(sorted(ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS))}, got custom_llm_provider={custom_llm_provider}. "
            "drop_params does not drop it, since that would let the model call tools the request excluded. "
            "Send only the allowed tools in `tools` with tool_choice='auto' or 'required' instead."
        ),
        model=model,
        llm_provider=custom_llm_provider,
    )


def parse_chat_allowed_tools_tool_choice(tool_choice: object) -> ChatAllowedToolsToolChoice | None:
    try:
        return _CHAT_ALLOWED_TOOLS_TOOL_CHOICE.validate_python(tool_choice)
    except ValidationError:
        return None


def _has_name(selector: Mapping[str, object]) -> bool:
    name: Final = selector.get("name")
    return isinstance(name, str) and name != ""


def _responses_allowed_tool(tool: Mapping[str, object]) -> Mapping[str, object] | None:
    tool_type: Final = tool.get("type")
    if tool_type not in ("function", "custom"):
        return tool
    if tool_type not in tool:
        return tool if _has_name(tool) else None
    try:
        nested: Final = _SELECTOR.validate_python(tool[tool_type])
    except ValidationError:
        return None
    outer: Final = {key: value for key, value in tool.items() if key != tool_type}
    if not _has_name(nested) or outer.keys() & nested.keys():
        return None
    return {**outer, **nested}


def responses_allowed_tools_tool_choice(choice: ChatAllowedToolsToolChoice) -> ToolChoiceAllowedParam | None:
    """The flat Responses API layout of a chat ``allowed_tools`` choice, or ``None`` when an entry is malformed or
    carries conflicting keys, so the caller sends the original rather than a rewrite that could mean something looser."""
    flattened: Final = tuple(_responses_allowed_tool(tool) for tool in choice.allowed_tools.tools)
    if any(tool is None for tool in flattened):
        return None
    return ToolChoiceAllowedParam(
        type="allowed_tools",
        mode=choice.allowed_tools.mode,
        tools=[dict(tool) for tool in flattened if tool is not None],
    )
