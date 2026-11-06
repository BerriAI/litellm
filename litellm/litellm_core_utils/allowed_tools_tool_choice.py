from collections.abc import Iterable, Mapping
from typing import Final, Literal

from openai.types.chat.chat_completion_allowed_tool_choice_param import ChatCompletionAllowedToolChoiceParam
from openai.types.chat.chat_completion_allowed_tools_param import ChatCompletionAllowedToolsParam
from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError

ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS: Final = frozenset({"openai", "azure", "litellm_proxy"})
_FIRST_AZURE_API_VERSION_WITH_TOOL_CHOICE: Final = "2023-12-01"
_TOOL_TYPES_THE_CALLER_RUNS: Final = frozenset({"function", "custom"})

CHAT_ALLOWED_TOOLS_LAYOUT: Final = (
    '{"type": "allowed_tools", "allowed_tools": {"mode": "auto" | "required", "tools": [<at least one tool>]}}'
)
RESPONSES_ALLOWED_TOOLS_LAYOUT: Final = (
    '{"type": "allowed_tools", "mode": "auto" | "required", "tools": [<at least one tool>]}'
)


class ChatAllowedTools(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["auto", "required"]
    tools: tuple[Mapping[str, object], ...] = Field(min_length=1)


class ChatAllowedToolsToolChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["allowed_tools"]
    allowed_tools: ChatAllowedTools


class ResponsesAllowedToolsToolChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["allowed_tools"]
    mode: Literal["auto", "required"]
    tools: tuple[Mapping[str, object], ...] = Field(min_length=1)


_ALLOWED_TOOLS_TOOL_CHOICE: Final[TypeAdapter[ChatAllowedToolsToolChoice | ResponsesAllowedToolsToolChoice]] = (
    TypeAdapter(ChatAllowedToolsToolChoice | ResponsesAllowedToolsToolChoice)
)


_SELECTOR: Final = TypeAdapter(dict[str, object])


class _ToolChoiceType(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


_TOOL_CHOICE_TYPE: Final = TypeAdapter(_ToolChoiceType)


def _declared_type(value: object) -> str | None:
    try:
        return _TOOL_CHOICE_TYPE.validate_python(value).type
    except ValidationError:
        return None


def is_allowed_tools_tool_choice(tool_choice: object) -> bool:
    return _declared_type(tool_choice) == "allowed_tools"


def caller_runs_every_tool(tools: Iterable[object] | None) -> bool:
    return all(_declared_type(tool) in _TOOL_TYPES_THE_CALLER_RUNS for tool in tools or ())


def drops_unsupported_allowed_tools_tool_choice(
    tool_choice: object,
    custom_llm_provider: str,
    model: str,
    drop_params: bool,
    tools: Iterable[object] | None,
) -> bool:
    if not is_allowed_tools_tool_choice(tool_choice) or custom_llm_provider in ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS:
        return False
    if drop_params and caller_runs_every_tool(tools):
        return True
    if drop_params:
        raise _a_choice_drop_params_keeps(custom_llm_provider, model)
    raise BadRequestError(
        message=(
            "tool_choice type 'allowed_tools' is only supported for "
            f"{', '.join(sorted(ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS))}, got custom_llm_provider={custom_llm_provider}. "
            "Send only the allowed tools in `tools` with tool_choice='auto' or 'required', or set drop_params=True "
            "to drop the tool_choice, which lets the model call any tool in `tools`."
        ),
        model=model,
        llm_provider=custom_llm_provider,
    )


def _has_name(selector: Mapping[str, object]) -> bool:
    name: Final = selector.get("name")
    return isinstance(name, str) and name != ""


def _chat_allowed_tool(tool: Mapping[str, object]) -> Mapping[str, object] | None:
    tool_type: Final = tool.get("type")
    if not isinstance(tool_type, str) or tool_type not in ("function", "custom"):
        return tool
    if tool_type in tool:
        return None if "name" in tool else tool
    if not _has_name(tool):
        return None
    return {**{key: value for key, value in tool.items() if key != "name"}, tool_type: {"name": tool["name"]}}


def _chat_layout(choice: ResponsesAllowedToolsToolChoice) -> ChatAllowedToolsToolChoice | None:
    nested: Final = tuple(_chat_allowed_tool(tool) for tool in choice.tools)
    if any(tool is None for tool in nested):
        return None
    return ChatAllowedToolsToolChoice(
        type="allowed_tools",
        allowed_tools=ChatAllowedTools(mode=choice.mode, tools=tuple(tool for tool in nested if tool is not None)),
    )


def parse_allowed_tools_tool_choice(tool_choice: object) -> ChatAllowedToolsToolChoice | None:
    try:
        parsed: Final = _ALLOWED_TOOLS_TOOL_CHOICE.validate_python(tool_choice)
    except ValidationError:
        return None
    match parsed:
        case ChatAllowedToolsToolChoice():
            return parsed
        case ResponsesAllowedToolsToolChoice():
            return _chat_layout(parsed)


def azure_api_version_takes_tool_choice(api_version: str) -> bool:
    return api_version.count("-") < 2 or api_version >= _FIRST_AZURE_API_VERSION_WITH_TOOL_CHOICE


def _azure_request_takes_tool_choice(api_version: str | None) -> bool:
    import litellm
    from litellm.secret_managers.main import get_secret_str

    return azure_api_version_takes_tool_choice(
        api_version or litellm.api_version or get_secret_str("AZURE_API_VERSION") or litellm.AZURE_DEFAULT_API_VERSION
    )


def _enforces_allowed_tools(custom_llm_provider: str | None, api_version: str | None) -> bool:
    if custom_llm_provider == "azure":
        return _azure_request_takes_tool_choice(api_version)
    return custom_llm_provider in ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS


def unenforced_allowed_tools_tool_choice(
    tool_choice: object, custom_llm_provider: str | None, api_version: str | None = None
) -> bool:
    return is_allowed_tools_tool_choice(tool_choice) and (
        not _enforces_allowed_tools(custom_llm_provider, api_version)
        or parse_allowed_tools_tool_choice(tool_choice) is None
    )


def _a_choice_drop_params_keeps(custom_llm_provider: str | None, model: str) -> BadRequestError:
    return BadRequestError(
        message=(
            f"This allowed_tools tool_choice can't be enforced on custom_llm_provider={custom_llm_provider}, and the "
            "caller does not run every tool on this request, so drop_params does not drop it. Send only the allowed "
            "tools in `tools` with tool_choice='auto' or 'required'."
        ),
        model=model,
        llm_provider=custom_llm_provider or "",
    )


def refuse_unenforced_allowed_tools_tool_choice(
    tool_choice: object, custom_llm_provider: str | None, model: str, api_version: str | None = None
) -> None:
    if unenforced_allowed_tools_tool_choice(tool_choice, custom_llm_provider, api_version):
        raise _a_choice_drop_params_keeps(custom_llm_provider, model)


def _allowed_tool_names(tool_choice: object) -> frozenset[object]:
    parsed: Final = parse_allowed_tools_tool_choice(tool_choice)
    if parsed is None:
        return frozenset()
    flattened: Final = tuple(_responses_allowed_tool(tool) for tool in parsed.allowed_tools.tools)
    return frozenset(tool.get("name") for tool in flattened if tool is not None)


def refuse_tool_calls_the_choice_excludes(tool_choice: object, tool_names: Iterable[str], model: str) -> None:
    if not is_allowed_tools_tool_choice(tool_choice):
        return
    allowed: Final = _allowed_tool_names(tool_choice)
    excluded: Final = tuple(name for name in tool_names if name not in allowed)
    if excluded:
        raise BadRequestError(
            message=(
                f"The model called {', '.join(excluded)}, which the allowed_tools tool_choice excludes, because this "
                "request did not reach a model that enforces the choice. LiteLLM did not run the call. Send only the "
                "allowed tools in `tools` with tool_choice='auto' or 'required'."
            ),
            model=model,
            llm_provider="",
        )


def chat_allowed_tools_tool_choice(choice: ChatAllowedToolsToolChoice) -> ChatCompletionAllowedToolChoiceParam:
    return ChatCompletionAllowedToolChoiceParam(
        type="allowed_tools",
        allowed_tools=ChatCompletionAllowedToolsParam(
            mode=choice.allowed_tools.mode, tools=[dict(tool) for tool in choice.allowed_tools.tools]
        ),
    )


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
