from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from openai.types.chat.chat_completion_allowed_tool_choice_param import ChatCompletionAllowedToolChoiceParam
from openai.types.chat.chat_completion_allowed_tools_param import ChatCompletionAllowedToolsParam
from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.exceptions import BadRequestError

ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS: Final = frozenset({"openai", "azure", "litellm_proxy"})
_FIRST_AZURE_API_VERSION_WITH_TOOL_CHOICE: Final = "2023-12-01"

HOSTED_TOOL_CHOICE_TYPES: Final = frozenset(
    {
        "file_search",
        "tool_search",
        "web_search",
        "web_search_preview",
        "web_search_preview_2025_03_11",
        "computer",
        "computer_use",
        "computer_use_preview",
        "code_interpreter",
        "image_generation",
        "mcp",
        "shell",
        "apply_patch",
    }
)

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
    *,
    bridges_to_interactions_api: bool = False,
) -> bool:
    if (
        bridges_to_interactions_api
        or not is_allowed_tools_tool_choice(tool_choice)
        or custom_llm_provider in ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS
    ):
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


def hosted_tool_choice_type(tool_choice: object) -> str | None:
    """The hosted tool a Responses API ``tool_choice`` forces, such as ``web_search``, or ``None`` for any other choice."""
    try:
        choice_type: Final = _TOOL_CHOICE_TYPE.validate_python(tool_choice).type
    except ValidationError:
        return None
    return choice_type if choice_type in HOSTED_TOOL_CHOICE_TYPES else None


def _forced_mcp_server(tool_choice: object) -> str | None:
    match tool_choice:
        case {"type": "mcp", "server_label": str() as server_label}:
            return server_label
        case _:
            return None


HostedMcpToolHandling: TypeAlias = Literal["forwards", "runs", "drops"]


@dataclass(frozen=True, slots=True)
class HostedMcpCall:
    handling: HostedMcpToolHandling
    sent_servers: frozenset[str]
    drops_unsupported_params: bool


_REMOVE_THE_MCP_CHOICE: Final = "Remove the tool_choice or set it to 'auto'."
_WHY_DROP_PARAMS_KEEPS_A_HOSTED_CHOICE: Final = (
    "drop_params does not drop it, since that would let the model answer without the tool the request required."
)


def _why_a_dropping_route_keeps_the_mcp_choice(server: str | None, call: HostedMcpCall) -> str | None:
    if server is None:
        return "This tool_choice names no server_label, so no hosted MCP tool drops with it. Remove the tool_choice."
    if server not in call.sent_servers:
        return (
            f"This request sends no hosted MCP tool with server_label `{server}`. "
            "Send that tool or remove the tool_choice."
        )
    if not call.drops_unsupported_params:
        return (
            f"Set drop_params to true to drop this tool_choice together with the hosted MCP tool `{server}`, "
            "or remove both."
        )
    return None


def _why_the_mcp_choice_stays(tool_choice: object, route: str, call: HostedMcpCall) -> str | None:
    match call.handling:
        case "runs":
            return f"{route} runs hosted MCP tools itself, but LiteLLM cannot force one there. {_REMOVE_THE_MCP_CHOICE}"
        case "forwards":
            return f"LiteLLM cannot force a hosted MCP tool on {route}. {_REMOVE_THE_MCP_CHOICE}"
        case "drops":
            return _why_a_dropping_route_keeps_the_mcp_choice(_forced_mcp_server(tool_choice), call)
        case _:
            assert_never(call.handling)


def raise_if_hosted_tool_choice_unsupported(
    tool_choice: object,
    custom_llm_provider: str,
    model: str,
    mcp: HostedMcpCall,
) -> None:
    """Anthropic drops a tool_choice dict it does not recognise and Bedrock sends a tool named "", so a forced hosted
    tool stops here with a 400 on providers that cannot run it. The one exception is an ``mcp`` choice on a route that
    drops hosted MCP tools, naming a server the request sent, under drop_params: the caller drops it with that tool."""
    hosted: Final = hosted_tool_choice_type(tool_choice)
    if hosted is None or custom_llm_provider in ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS:
        return
    why: Final = (
        _why_the_mcp_choice_stays(tool_choice, custom_llm_provider, mcp)
        if hosted == "mcp"
        else _WHY_DROP_PARAMS_KEEPS_A_HOSTED_CHOICE
    )
    if why is None:
        return
    raise BadRequestError(
        message=(
            f"tool_choice type '{hosted}' forces an OpenAI hosted tool and is only supported for "
            f"{', '.join(sorted(ALLOWED_TOOLS_TOOL_CHOICE_PROVIDERS))}, got custom_llm_provider={custom_llm_provider}. "
            f"{why}"
        ),
        model=model,
        llm_provider=custom_llm_provider,
    )


def raise_if_mcp_tool_choice_forces_a_gateway_server(
    tool_choice: object, gateway_tools: Iterable[Mapping[str, object]], model: str
) -> None:
    server: Final = _forced_mcp_server(tool_choice)
    if server is None or server not in {tool.get("server_label") for tool in gateway_tools}:
        return
    raise BadRequestError(
        message=(
            f"tool_choice type 'mcp' cannot force `{server}`, which runs on the LiteLLM MCP gateway. The gateway sends "
            "its tools to the model as functions, so force one by its function name or remove the tool_choice."
        ),
        model=model,
        llm_provider="",
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


def parse_allowed_tools_tool_choice_or_refuse(
    tool_choice: object, model: str, drop_params: bool
) -> ChatAllowedToolsToolChoice | None:
    parsed: Final = parse_allowed_tools_tool_choice(tool_choice)
    if parsed is not None or drop_params:
        return parsed
    raise BadRequestError(
        message=(
            f"Invalid tool choice, tool_choice={tool_choice}. An allowed_tools tool_choice must be "
            f"{CHAT_ALLOWED_TOOLS_LAYOUT} or {RESPONSES_ALLOWED_TOOLS_LAYOUT}"
        ),
        model=model,
        llm_provider="",
    )


def allowed_tools_tool_choice_in_effect(tool_choice: object, model: str, drop_params: bool) -> object:
    if not is_allowed_tools_tool_choice(tool_choice):
        return tool_choice
    return None if parse_allowed_tools_tool_choice_or_refuse(tool_choice, model, drop_params) is None else tool_choice


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


@dataclass(frozen=True, slots=True)
class GatewayTool:
    names: frozenset[str]
    server_labels: frozenset[str]


@dataclass(frozen=True, slots=True)
class GatewayTools:
    server_labels: frozenset[str]
    tools: Mapping[str, GatewayTool]


NO_GATEWAY_TOOLS: Final = GatewayTools(server_labels=frozenset(), tools=MappingProxyType({}))


def _server_entry_allows(entry: tuple[str, str | None], tool: GatewayTool | None) -> bool:
    label, name = entry
    return tool is not None and label in tool.server_labels and (name is None or name in tool.names)


def _a_server_entry_allows(entries: Iterable[tuple[str, str | None]], tool: GatewayTool | None) -> bool:
    return any(_server_entry_allows(entry, tool) for entry in entries)


def tool_calls_the_choice_excludes(
    tool_choice: object, tool_names: Iterable[str], gateway: GatewayTools = NO_GATEWAY_TOOLS
) -> tuple[str, ...]:
    if not is_allowed_tools_tool_choice(tool_choice):
        return ()
    allowed: Final = _allowed_tool_names(tool_choice)
    server_entries: Final = _server_entries(tool_choice)
    return tuple(
        name
        for name in tool_names
        if name not in allowed and not _a_server_entry_allows(server_entries, gateway.tools.get(name))
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


def _function_entry(
    layout: ChatAllowedToolsToolChoice | ResponsesAllowedToolsToolChoice, name: str
) -> Mapping[str, object]:
    match layout:
        case ChatAllowedToolsToolChoice():
            return {"type": "function", "function": {"name": name}}
        case ResponsesAllowedToolsToolChoice():
            return {"type": "function", "name": name}


def _entries_of(
    layout: ChatAllowedToolsToolChoice | ResponsesAllowedToolsToolChoice,
) -> tuple[Mapping[str, object], ...]:
    match layout:
        case ChatAllowedToolsToolChoice():
            return layout.allowed_tools.tools
        case ResponsesAllowedToolsToolChoice():
            return layout.tools


def _with_entries(
    layout: ChatAllowedToolsToolChoice | ResponsesAllowedToolsToolChoice, tools: Iterable[Mapping[str, object]]
) -> Mapping[str, object]:
    entries: Final = [dict(tool) for tool in tools]
    match layout:
        case ChatAllowedToolsToolChoice():
            return {"type": "allowed_tools", "allowed_tools": {"mode": layout.allowed_tools.mode, "tools": entries}}
        case ResponsesAllowedToolsToolChoice():
            return {"type": "allowed_tools", "mode": layout.mode, "tools": entries}


def _callers_layout(tool_choice: object) -> ChatAllowedToolsToolChoice | ResponsesAllowedToolsToolChoice | None:
    if parse_allowed_tools_tool_choice(tool_choice) is None:
        return None
    return _ALLOWED_TOOLS_TOOL_CHOICE.validate_python(tool_choice)


def allowed_tools_tool_choice_with_builtin_as_function(
    tool_choice: object, is_builtin: Callable[[Mapping[str, object]], bool], function_name: str
) -> object:
    layout: Final = _callers_layout(tool_choice)
    if layout is None:
        return tool_choice
    entry: Final = _function_entry(layout, function_name)
    return _with_entries(layout, (_with_builtin_as(tool, is_builtin, entry) for tool in _entries_of(layout)))


def allowed_tools_tool_choice_with_function(tool_choice: object, function_name: str) -> object:
    layout: Final = _callers_layout(tool_choice)
    if layout is None or function_name in _allowed_tool_names(tool_choice):
        return tool_choice
    return _with_entries(layout, (*_entries_of(layout), _function_entry(layout, function_name)))


def allowed_tools_tool_choice_in_auto_mode(tool_choice: object) -> object:
    layout: Final = _callers_layout(tool_choice)
    match layout:
        case None:
            return tool_choice
        case ChatAllowedToolsToolChoice():
            return layout.model_copy(
                update={"allowed_tools": layout.allowed_tools.model_copy(update={"mode": "auto"})}
            ).model_dump(mode="json")
        case ResponsesAllowedToolsToolChoice():
            return layout.model_copy(update={"mode": "auto"}).model_dump(mode="json")


def allowed_tools_tool_choice_without(
    tool_choice: object, is_removed: Callable[[Mapping[str, object]], bool]
) -> object:
    layout: Final = _callers_layout(tool_choice)
    if layout is None:
        return tool_choice
    kept: Final = tuple(tool for tool in _entries_of(layout) if not is_removed(tool))
    return _with_entries(layout, kept) if kept else "none"


def allowed_tools_tool_choice_kept_on_follow_up(tool_choice: object) -> Mapping[str, object]:
    if not is_allowed_tools_tool_choice(tool_choice):
        return MappingProxyType({})
    return MappingProxyType({"tool_choice": allowed_tools_tool_choice_in_auto_mode(tool_choice)})


def _gateway_entry(tool: Mapping[str, object], gateway_labels: frozenset[str]) -> tuple[str, str | None] | None:
    entry: Final = _server_entry(tool)
    return entry if entry is not None and entry[0] in gateway_labels else None


def allowed_tools_tool_choice_with_gateway_tools_as_functions(tool_choice: object, gateway: GatewayTools) -> object:
    layout: Final = _callers_layout(tool_choice)
    entries: Final = () if layout is None else _entries_of(layout)
    gateway_entries: Final = tuple(
        entry for entry in (_gateway_entry(tool, gateway.server_labels) for tool in entries) if entry is not None
    )
    if layout is None or not gateway_entries:
        return tool_choice
    kept: Final = tuple(tool for tool in entries if _gateway_entry(tool, gateway.server_labels) is None)
    allowed: Final = tuple(
        name for name, tool in gateway.tools.items() if _a_server_entry_allows(gateway_entries, tool)
    )
    if not kept and not allowed:
        return "none"
    return _with_entries(layout, (*kept, *(_function_entry(layout, name) for name in allowed)))


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
