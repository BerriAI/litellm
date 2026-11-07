from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Final, Literal

from openai.types.chat.chat_completion_allowed_tool_choice_param import ChatCompletionAllowedToolChoiceParam
from openai.types.chat.chat_completion_allowed_tools_param import ChatCompletionAllowedToolsParam
from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError

ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS: Final = frozenset({"openai", "azure", "litellm_proxy"})
_FIRST_AZURE_API_VERSION_WITH_TOOL_CHOICE: Final = "2023-12-01"

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


def is_allowed_tools_tool_choice(tool_choice: object) -> bool:
    try:
        return _TOOL_CHOICE_TYPE.validate_python(tool_choice).type == "allowed_tools"
    except ValidationError:
        return False


def drops_unsupported_allowed_tools_tool_choice(
    tool_choice: object,
    custom_llm_provider: str,
    model: str,
    drop_params: bool,
) -> bool:
    if not is_allowed_tools_tool_choice(tool_choice) or custom_llm_provider in ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS:
        return False
    if drop_params:
        return True
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


def _names_a_tool_or_nothing(tool: Mapping[str, object]) -> bool:
    tool_type: Final = tool.get("type")
    if not isinstance(tool_type, str) or tool_type not in ("function", "custom"):
        return tool.get("name") is None or _has_name(tool)
    match tool.get(tool_type):
        case {"name": str()}:
            return True
        case {"name": object()}:
            return False
        case _:
            return True


def _names_are_strings(tools: Iterable[Mapping[str, object]]) -> bool:
    return all(map(_names_a_tool_or_nothing, tools))


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
            return parsed if _names_are_strings(parsed.allowed_tools.tools) else None
        case ResponsesAllowedToolsToolChoice():
            return _chat_layout(parsed) if _names_are_strings(parsed.tools) else None


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


def _allowed_tool_names(tool_choice: object) -> frozenset[object]:
    parsed: Final = parse_allowed_tools_tool_choice(tool_choice)
    if parsed is None:
        return frozenset()
    flattened: Final = tuple(_responses_allowed_tool(tool) for tool in parsed.allowed_tools.tools)
    return frozenset(tool.get("name") for tool in flattened if tool is not None and _server_entry(tool) is None)


def refuse_a_malformed_allowed_tools_tool_choice(tool_choice: object, model: str) -> None:
    if is_allowed_tools_tool_choice(tool_choice) and parse_allowed_tools_tool_choice(tool_choice) is None:
        raise BadRequestError(
            message=(
                f"Invalid tool choice, tool_choice={tool_choice}. An allowed_tools tool_choice must be "
                f"{CHAT_ALLOWED_TOOLS_LAYOUT} or {RESPONSES_ALLOWED_TOOLS_LAYOUT}"
            ),
            model=model,
            llm_provider="",
        )


def _server_entry(tool: Mapping[str, object]) -> tuple[str, str | None] | None:
    label: Final = tool.get("server_label")
    name: Final = tool.get("name")
    if tool.get("type") != "mcp" or not isinstance(label, str):
        return None
    return label, name if isinstance(name, str) else None


def _server_entries(tool_choice: object) -> frozenset[tuple[str, str | None]]:
    parsed: Final = parse_allowed_tools_tool_choice(tool_choice)
    entries: Final = () if parsed is None else tuple(map(_server_entry, parsed.allowed_tools.tools))
    return frozenset(entry for entry in entries if entry is not None)


def _a_server_entry_allows(entries: frozenset[tuple[str, str | None]], name: str, labels: frozenset[str]) -> bool:
    return any(label in labels and entry_name in (None, name) for label, entry_name in entries)


def tool_calls_the_choice_excludes(
    tool_choice: object,
    tool_names: Iterable[str],
    server_labels_by_tool: Mapping[str, frozenset[str]] = MappingProxyType({}),
) -> tuple[str, ...]:
    if not is_allowed_tools_tool_choice(tool_choice):
        return ()
    allowed: Final = _allowed_tool_names(tool_choice)
    server_entries: Final = _server_entries(tool_choice)
    return tuple(
        name
        for name in tool_names
        if name not in allowed
        and not _a_server_entry_allows(server_entries, name, server_labels_by_tool.get(name, frozenset()))
    )


def refuse_tool_calls_the_choice_excludes(tool_choice: object, tool_names: Iterable[str], model: str) -> None:
    excluded: Final = tool_calls_the_choice_excludes(tool_choice, tool_names)
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


def _with_builtin_as(
    tool: Mapping[str, object], is_builtin: Callable[[Mapping[str, object]], bool], function_entry: Mapping[str, object]
) -> Mapping[str, object]:
    return function_entry if tool.get("type") not in ("function", "custom", "mcp") and is_builtin(tool) else tool


def allowed_tools_tool_choice_with_builtin_as_function(
    tool_choice: object, is_builtin: Callable[[Mapping[str, object]], bool], function_name: str
) -> object:
    try:
        parsed: Final = _ALLOWED_TOOLS_TOOL_CHOICE.validate_python(tool_choice)
    except ValidationError:
        return tool_choice
    match parsed:
        case ChatAllowedToolsToolChoice():
            nested: Final = {"type": "function", "function": {"name": function_name}}
            return {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": parsed.allowed_tools.mode,
                    "tools": [dict(_with_builtin_as(tool, is_builtin, nested)) for tool in parsed.allowed_tools.tools],
                },
            }
        case ResponsesAllowedToolsToolChoice():
            flat: Final = {"type": "function", "name": function_name}
            return {
                "type": "allowed_tools",
                "mode": parsed.mode,
                "tools": [dict(_with_builtin_as(tool, is_builtin, flat)) for tool in parsed.tools],
            }


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
