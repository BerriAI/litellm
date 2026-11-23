import hashlib
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypeAlias, TypeVar

from pydantic import BeforeValidator, ConfigDict, Json, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    HostedMcpCall,
    HostedMcpToolHandling,
    raise_if_hosted_tool_choice_unsupported,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params, normalize_drop_params
from litellm.types.utils import (
    ChatCompletionDeltaCustomToolCall,
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

OPENAI_MAX_TOOL_NAME_LENGTH: Final = 64
TOOL_NAME_HASH_LENGTH: Final = 8
TOOL_NAME_PREFIX_LENGTH: Final = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1


BRIDGED_TOOL_NAMES_KWARG: Final = "litellm_bridged_tool_names"

ROUTES_WITH_TOOLS: Final = frozenset(
    {
        "acompletion",
        "aresponses",
        "acompact_responses",
        "anthropic_messages",
        "agenerate_content",
        "agenerate_content_stream",
        "acreate_interaction",
    }
)
TOOL_FIELDS: Final = frozenset({"tools", "tool_choice", "parallel_tool_calls", "toolConfig", "tool_config"})
_TOOLS: Final = TypeAdapter(list[object])
_FIELD_LIST: Final = TypeAdapter(list[object], config=ConfigDict(strict=True))
_JSON_OBJECT: Final = TypeAdapter(Json[dict[str, object]])


def _schema_as_read(value: object) -> object:
    if not value:
        return None
    if not isinstance(value, str):
        return value
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return value


_Schema: TypeAlias = Annotated[dict[str, object] | None, BeforeValidator(_schema_as_read)]


class _ToolFunction(TypedDict, total=False):
    parameters: ReadOnly[_Schema]


class _ToolEntry(TypedDict, total=False):
    function: ReadOnly[_ToolFunction | None]
    parameters: ReadOnly[_Schema]
    input_schema: ReadOnly[_Schema]
    vector_store_ids: ReadOnly[list[str] | None]


_TOOL_ENTRY: Final = TypeAdapter(_ToolEntry)
_WEB_SEARCH_OPTIONS: Final = TypeAdapter(dict[str, object] | None, config=ConfigDict(strict=True))


def declared_function_name(tool: object) -> str | None:
    fields: Final = as_str_mapping(tool)
    if fields is None:
        return None
    function: Final = as_str_mapping(fields.get("function"))
    name: Final = (fields if function is None else function).get("name")
    return name if isinstance(name, str) and name else None


def qualified_member_name(namespace: str, name: str) -> str:
    return f"{namespace}__{name}"


@dataclass(frozen=True, slots=True)
class BridgedTool:
    namespace: str
    name: str

    @property
    def full_name(self) -> str:
        return qualified_member_name(self.namespace, self.name)


@dataclass(frozen=True, slots=True)
class BridgedToolNames:
    """The namespace tool the caller gets back for each name a bridged request let the provider call."""

    tool_by_sent_name: tuple[tuple[str, BridgedTool], ...]

    def as_mapping(self) -> Mapping[str, str]:
        return {sent_name: tool.full_name for sent_name, tool in self.tool_by_sent_name}

    def tools(self) -> Mapping[str, BridgedTool]:
        return MappingProxyType(dict(self.tool_by_sent_name))


_NO_BRIDGED_TOOLS: Final = MappingProxyType[str, BridgedTool]({})


def shortened_full_names(tools: Mapping[str, BridgedTool]) -> Mapping[str, str]:
    """The full name behind each name the bridge had to shorten, which is all logs and spend tracking rename."""
    return MappingProxyType(
        {
            sent_name: tool.full_name
            for sent_name, tool in tools.items()
            if sent_name == chat_member_name(tool.namespace, tool.name) != tool.full_name
        }
    )


def recorded_bridged_tools(call_details: object) -> Mapping[str, BridgedTool]:
    """The bridged tools a request's logging object recorded, which a client cannot set from its own JSON."""
    fields: Final = as_str_mapping(call_details)
    recorded: Final = None if fields is None else fields.get(BRIDGED_TOOL_NAMES_KWARG)
    return recorded.tools() if isinstance(recorded, BridgedToolNames) else _NO_BRIDGED_TOOLS


@dataclass(frozen=True, slots=True)
class SentToolNames:
    member_chat_names: Mapping[str, str]
    self_named_namespace_members: Container[str]


_RenamableResponse = TypeVar("_RenamableResponse", ModelResponse, ModelResponseStream)
_RenamableToolCall = TypeVar(
    "_RenamableToolCall",
    ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
    ChatCompletionDeltaToolCall | ChatCompletionDeltaCustomToolCall,
)


def with_renamed_tool_calls(response: _RenamableResponse, new_names: Mapping[str, str]) -> _RenamableResponse:
    if not new_names:
        return response
    return response.model_copy(
        update={"choices": [_choice_with_renamed_tool_calls(choice, new_names) for choice in response.choices]}
    )


def _choice_with_renamed_tool_calls(
    choice: Choices | StreamingChoices, new_names: Mapping[str, str]
) -> Choices | StreamingChoices:
    if isinstance(choice, StreamingChoices):
        return _streaming_choice_with_renamed_tool_calls(choice, new_names)
    if not choice.message.tool_calls:
        return choice
    tool_calls: Final = [_renamed_tool_call(tool_call, new_names) for tool_call in choice.message.tool_calls]
    return choice.model_copy(update={"message": choice.message.model_copy(update={"tool_calls": tool_calls})})


def _streaming_choice_with_renamed_tool_calls(
    choice: StreamingChoices, new_names: Mapping[str, str]
) -> StreamingChoices:
    if not choice.delta.tool_calls:
        return choice
    tool_calls: Final = [_renamed_tool_call(tool_call, new_names) for tool_call in choice.delta.tool_calls]
    return choice.model_copy(update={"delta": choice.delta.model_copy(update={"tool_calls": tool_calls})})


def _renamed_tool_call(tool_call: _RenamableToolCall, new_names: Mapping[str, str]) -> _RenamableToolCall:
    if not isinstance(tool_call, ChatCompletionMessageToolCall | ChatCompletionDeltaToolCall):
        return tool_call
    new_name: Final = new_names.get(tool_call.function.name or "")
    if new_name is None or new_name == tool_call.function.name:
        return tool_call
    return tool_call.model_copy(update={"function": tool_call.function.model_copy(update={"name": new_name})})


def _function_call_names(choice: Choices | StreamingChoices) -> tuple[str, ...]:
    if not isinstance(choice, Choices):
        return ()
    return tuple(
        tool_call.function.name
        for tool_call in choice.message.tool_calls or ()
        if isinstance(tool_call, ChatCompletionMessageToolCall) and tool_call.function.name
    )


def declared_function_names(tools: object) -> frozenset[str]:
    try:
        entries: Final = _TOOLS.validate_python(tools)
    except ValidationError:
        return frozenset()
    return frozenset(name for name in map(declared_function_name, entries) if name)


def with_tool_calls_named_as_declared(response: ModelResponse, declared_names: frozenset[str]) -> ModelResponse:
    called_names: Final = frozenset(chain.from_iterable(map(_function_call_names, response.choices)))
    return with_renamed_tool_calls(
        response,
        MappingProxyType(
            {
                name: truncate_tool_name(name)
                for name in called_names
                if name not in declared_names and truncate_tool_name(name) in declared_names
            }
        ),
    )


def truncate_tool_name(name: str) -> str:
    """
    Truncate tool names that exceed OpenAI's 64-character limit.

    Uses format: {55-char-prefix}_{8-char-hash} to avoid collisions
    when multiple tools have similar long names.

    Args:
        name: The original tool name

    Returns:
        The original name if <= 64 chars, otherwise truncated with hash
    """
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name

    name_hash: Final = hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{name_hash}"


def chat_member_name(namespace: str, name: str) -> str:
    return truncate_tool_name(qualified_member_name(namespace, name))


@dataclass(frozen=True, slots=True)
class DropPolicy:
    drop_params: bool
    additional_drop_params: frozenset[str]

    def drops(self, field: str) -> bool:
        return self.drop_params or field in self.additional_drop_params


@dataclass(frozen=True, slots=True)
class MalformedField:
    field: str
    message: str
    dropped: Callable[[Mapping[str, object]], Mapping[str, object]]


def _field_names(value: object) -> frozenset[str]:
    try:
        return frozenset(name for name in _FIELD_LIST.validate_python(value) if isinstance(name, str))
    except ValidationError:
        return frozenset()


def drop_policy(params: Mapping[str, object]) -> DropPolicy:
    return DropPolicy(
        drop_params=litellm.drop_params is True or normalize_drop_params(params.get("drop_params")) is True,
        additional_drop_params=_field_names(params.get("additional_drop_params")),
    )


def first_validation_error(e: ValidationError, prefix: tuple[str | int, ...] = ()) -> str:
    error: Final = e.errors(include_url=False)[0]
    return f"{'.'.join(str(part) for part in (*prefix, *error['loc']))}: {error['msg']}"


def _tool_entry_error(index: int, entry: object) -> str | None:
    try:
        _TOOL_ENTRY.validate_python(entry, strict=True)
    except ValidationError as e:
        return first_validation_error(e, prefix=("tools", index))
    return None


def _without_tool_fields(data: Mapping[str, object]) -> Mapping[str, object]:
    return {name: value for name, value in data.items() if name not in TOOL_FIELDS}


def _malformed_tools(data: Mapping[str, object]) -> MalformedField | None:
    tools: Final = data.get("tools")
    if tools is None:
        return None
    try:
        entries: Final = _TOOLS.validate_python(tools)
    except ValidationError as e:
        return MalformedField("tools", first_validation_error(e, prefix=("tools",)), _without_tool_fields)
    entry_errors: Final = tuple(_tool_entry_error(index, entry) for index, entry in enumerate(entries))
    first_error: Final = next((error for error in entry_errors if error is not None), None)
    if first_error is None:
        return None
    kept: Final = [entry for entry, error in zip(entries, entry_errors) if error is None]
    return MalformedField(
        "tools", first_error, lambda fields: {**fields, "tools": kept} if kept else _without_tool_fields(fields)
    )


def _without_web_search_options(data: Mapping[str, object]) -> Mapping[str, object]:
    return {name: value for name, value in data.items() if name != "web_search_options"}


def _malformed_web_search_options(data: Mapping[str, object]) -> MalformedField | None:
    try:
        _WEB_SEARCH_OPTIONS.validate_python(data.get("web_search_options"))
    except ValidationError as e:
        return MalformedField(
            "web_search_options",
            first_validation_error(e, prefix=("web_search_options",)),
            _without_web_search_options,
        )
    return None


def resolve_tool_fields(
    data: Mapping[str, object], candidates: Callable[[], tuple[DropPolicy, ...]]
) -> Mapping[str, object] | MalformedField:
    checked: Final = (_malformed_tools(data), _malformed_web_search_options(data))
    malformed: Final = tuple(field for field in checked if field is not None)
    if not malformed:
        return data
    policies: Final = candidates()
    rejected: Final = next(
        (field for field in malformed if not any(policy.drops(field.field) for policy in policies)), None
    )
    if rejected is not None:
        return rejected
    return reduce(
        lambda resolved, field: field.dropped(resolved),
        (field for field in malformed if all(policy.drops(field.field) for policy in policies)),
        data,
    )


def resolve_deployment_tool_fields(
    kwargs: Mapping[str, object], route: object
) -> Mapping[str, object] | MalformedField:
    if "proxy_server_request" not in kwargs or route not in ROUTES_WITH_TOOLS:
        return kwargs
    return resolve_tool_fields(kwargs, lambda: (drop_policy(kwargs),))


_ToolT = TypeVar("_ToolT")

HOSTED_MCP_DROP_WARNING: Final = (
    "Dropping the hosted MCP tool(s) %s for model=%s on %s (drop_params=True) because this route cannot run a "
    "hosted MCP tool."
)


def _hosted_mcp_server_label(tool: object) -> str | None:
    match tool:
        case {"type": "mcp", "server_label": str() as server_label}:
            return server_label
        case {"type": "mcp"}:
            return "unlabeled"
        case _:
            return None


def _drop_params_hint(dropped: str) -> str:
    return (
        f"To drop {dropped} instead, set `litellm.drop_params=True` or for proxy:\n\n"
        "`litellm_settings:\n drop_params: true`\n"
    )


def _one_log_line(text: str) -> str:
    return text.replace("\r", "").replace("\n", "")


def _unrunnable_mcp_error(message: str, model: str, route: str) -> UnsupportedParamsError:
    return UnsupportedParamsError(message=message, model=model, llm_provider=route, status_code=400)


_HOSTED_MCP_GATEWAY_REMEDY: Final = (
    "To have LiteLLM run the MCP server for this model, register it on the LiteLLM MCP gateway and send "
    '`server_url: "litellm_proxy/mcp/<server name>"`. ' + _drop_params_hint("hosted MCP tools")
)


def hosted_mcp_batch_row_remedy(row_drop_params: object) -> str:
    if row_drop_params is not None:
        return (
            "A batch row cannot reach the LiteLLM MCP gateway, and this row sets `drop_params` to a value that does "
            "not turn dropping on, which wins over the router default and the deployment. Remove the tool from the "
            "row, or set `drop_params: true` on the row to drop it."
        )
    return (
        "A batch row cannot reach the LiteLLM MCP gateway, so remove the tool from the row, or set `drop_params: true` "
        "on the row, in `router_settings.default_litellm_params`, or on the deployment the upload names, to drop it."
    )


def tools_without_hosted_mcp(
    tools: Sequence[_ToolT], model: str, route: str, drop_params: bool, remedy: str = _HOSTED_MCP_GATEWAY_REMEDY
) -> Sequence[_ToolT]:
    server_labels: Final = tuple(label for label in map(_hosted_mcp_server_label, tools) if label is not None)
    if not server_labels:
        return tools
    named_servers: Final = ", ".join(f"`{label}`" for label in server_labels)
    if not drops_unsupported_params(drop_params):
        raise _unrunnable_mcp_error(
            f'{model} on {route} cannot run the hosted MCP tool {named_servers} (`type: "mcp"`). {remedy}',
            model=model,
            route=route,
        )
    verbose_logger.warning(
        HOSTED_MCP_DROP_WARNING, _one_log_line(named_servers), _one_log_line(model), _one_log_line(route)
    )
    return [tool for tool in tools if _hosted_mcp_server_label(tool) is None]


MCP_SERVERS_DROP_WARNING: Final = (
    "Dropping mcp_servers %s and their mcp_toolset tools for model=%s on %s (drop_params=True) because LiteLLM "
    "cannot pass mcp_servers to this route."
)

ROUTES_WITHOUT_MCP_SERVERS: Final = frozenset(
    {"bedrock", "databricks", "deepseek", "gemini", "gigachat", "ollama", "ollama_chat", "vertex_ai", "vertex_ai_beta"}
)


def _mcp_server_name(server: object) -> str:
    match server:
        case {"name": str() as name}:
            return name
        case _:
            return "unnamed"


def _mcp_server_names(mcp_servers: object) -> tuple[str, ...]:
    if not mcp_servers:
        return ()
    if not isinstance(mcp_servers, list):
        return ("unnamed",)
    return tuple(map(_mcp_server_name, _TOOLS.validate_python(mcp_servers)))


def _is_mcp_toolset(tool: object) -> bool:
    match tool:
        case {"type": "mcp_toolset"}:
            return True
        case _:
            return False


def drops_mcp_servers(request_kwargs: Mapping[str, object]) -> bool:
    additional_drop_params: Final = request_kwargs.get("additional_drop_params")
    return drops_unsupported_params(request_kwargs.get("drop_params")) or (
        isinstance(additional_drop_params, list) and "mcp_servers" in additional_drop_params
    )


def tools_without_mcp_servers(
    mcp_servers: object, tools: Sequence[_ToolT], model: str, route: str, drop_params: bool
) -> Sequence[_ToolT]:
    server_names: Final = _mcp_server_names(mcp_servers)
    if not server_names:
        return tools
    named_servers: Final = ", ".join(f"`{name}`" for name in server_names)
    if not drops_unsupported_params(drop_params):
        raise _unrunnable_mcp_error(
            f"LiteLLM cannot pass `mcp_servers` to {model} on {route}, so the MCP server {named_servers} would not "
            'run. To have LiteLLM run it for this model, register it on the LiteLLM MCP gateway and send a `type: "mcp"` '
            'tool with `server_url: "litellm_proxy/mcp/<server name>"`. ' + _drop_params_hint("`mcp_servers`"),
            model=model,
            route=route,
        )
    verbose_logger.warning(
        MCP_SERVERS_DROP_WARNING, _one_log_line(named_servers), _one_log_line(model), _one_log_line(route)
    )
    return [tool for tool in tools if not _is_mcp_toolset(tool)]


_TOOL_SELECTION_PARAMS: Final = frozenset({"tools", "tool_choice", "parallel_tool_calls"})
_ParamsT = TypeVar("_ParamsT", bound=dict[str, object])


def _forced_function_name(tool_choice: object) -> str | None:
    match tool_choice:
        case {"type": "function", "function": {"name": str() as name}}:
            return name
        case _:
            return None


def _labeled_mcp_server(tool: object) -> str | None:
    match tool:
        case {"type": "mcp", "server_label": str() as server_label}:
            return server_label
        case _:
            return None


def _forces_a_dropped_mcp_tool(
    tool_choice: object, sent_tools: Sequence[object], kept_tools: Sequence[object], functions: object
) -> bool:
    if _hosted_mcp_server_label(tool_choice) is not None:
        return True
    dropped_servers: Final = frozenset(label for label in map(_labeled_mcp_server, sent_tools) if label is not None)
    forced: Final = _forced_function_name(tool_choice)
    if forced not in dropped_servers:
        return False
    legacy_functions: Final = _TOOLS.validate_python(functions) if isinstance(functions, list) else []
    return forced not in map(declared_function_name, (*kept_tools, *legacy_functions))


def _cleared_tool_selection_params(kept_tools: Sequence[object], functions: object) -> frozenset[str]:
    if kept_tools:
        return frozenset()
    return frozenset({"tools"}) if functions else _TOOL_SELECTION_PARAMS


def _allowed_tool_entries(tool_choice: Mapping[str, object]) -> Sequence[object]:
    if tool_choice.get("type") != "allowed_tools":
        return ()
    entries: Final = (as_str_mapping(tool_choice.get("allowed_tools")) or tool_choice).get("tools")
    return _TOOLS.validate_python(entries) if isinstance(entries, list) else ()


def _allowed_tools_choice(tool_choice: Mapping[str, object], entries: Sequence[object]) -> Mapping[str, object]:
    nested: Final = as_str_mapping(tool_choice.get("allowed_tools"))
    if nested is None:
        return {**tool_choice, "tools": list(entries)}
    return {**tool_choice, "allowed_tools": {**nested, "tools": list(entries)}}


def _tool_params_narrowed_to_allowed_tools(
    params: _ParamsT, kept_tools: Sequence[object], tool_choice: Mapping[str, object], allowed: Sequence[object]
) -> _ParamsT:
    if not kept_tools:
        return type(params)({key: value for key, value in params.items() if key not in _TOOL_SELECTION_PARAMS})
    narrowed_choice: Final = {"tool_choice": _allowed_tools_choice(tool_choice, allowed)} if allowed else {}
    return type(params)(
        {
            **{key: value for key, value in params.items() if key != "tool_choice"},
            "tools": list(kept_tools),
            **narrowed_choice,
        }
    )


def _sent_tools(tools: object) -> Sequence[object]:
    return _TOOLS.validate_python(tools) if isinstance(tools, list) else ()


def _hosted_mcp_call(sent_tools: Sequence[object], drop_params: bool, handling: HostedMcpToolHandling) -> HostedMcpCall:
    return HostedMcpCall(
        handling=handling,
        sent_servers=frozenset(label for label in map(_labeled_mcp_server, sent_tools) if label is not None),
        drops_unsupported_params=drops_unsupported_params(drop_params),
    )


class HostedMcpToolHandler(Protocol):
    def hosted_mcp_tool_handling(self, model: str) -> HostedMcpToolHandling: ...


def raise_if_a_forced_hosted_tool_stays(
    params: Mapping[str, object],
    model: str,
    route: str,
    drop_params: bool,
    provider_config: HostedMcpToolHandler | None,
) -> None:
    raise_if_hosted_tool_choice_unsupported(
        params.get("tool_choice"),
        custom_llm_provider=route,
        model=model,
        mcp=_hosted_mcp_call(
            _sent_tools(params.get("tools")),
            drop_params,
            "forwards" if provider_config is None else provider_config.hosted_mcp_tool_handling(model),
        ),
    )


def _refuse_an_mcp_choice_drop_params_keeps(
    tool_choice: object,
    sent_tools: Sequence[object],
    model: str,
    route: str,
    drop_params: bool,
    handling: HostedMcpToolHandling,
) -> None:
    if _hosted_mcp_server_label(tool_choice) is not None:
        raise_if_hosted_tool_choice_unsupported(
            tool_choice, custom_llm_provider=route, model=model, mcp=_hosted_mcp_call(sent_tools, drop_params, handling)
        )


def tool_params_without_hosted_mcp(
    params: _ParamsT,
    model: str,
    route: str,
    drop_params: bool,
    handling: HostedMcpToolHandling,
    remedy: str = _HOSTED_MCP_GATEWAY_REMEDY,
) -> _ParamsT:
    sent_tools: Final = _sent_tools(params.get("tools"))
    sends_hosted_mcp: Final = any(_hosted_mcp_server_label(tool) is not None for tool in sent_tools)
    if sends_hosted_mcp and handling == "runs":
        raise_if_hosted_tool_choice_unsupported(
            params.get("tool_choice"),
            custom_llm_provider=route,
            model=model,
            mcp=_hosted_mcp_call(sent_tools, drop_params, handling),
        )
        return params
    _refuse_an_mcp_choice_drop_params_keeps(params.get("tool_choice"), sent_tools, model, route, drop_params, handling)
    if not sends_hosted_mcp or handling != "drops":
        return params
    kept_tools: Final = tools_without_hosted_mcp(
        tools=sent_tools, model=model, route=route, drop_params=drop_params, remedy=remedy
    )
    if len(kept_tools) == len(sent_tools):
        return params
    tool_choice: Final = as_str_mapping(params.get("tool_choice")) or {}
    allowed_entries: Final = _allowed_tool_entries(tool_choice)
    still_allowed: Final = tuple(entry for entry in allowed_entries if _hosted_mcp_server_label(entry) is None)
    if len(still_allowed) < len(allowed_entries):
        return _tool_params_narrowed_to_allowed_tools(params, kept_tools, tool_choice, still_allowed)
    functions: Final = params.get("functions")
    cleared: Final = _cleared_tool_selection_params(kept_tools, functions)
    removed: Final = (
        cleared | {"tool_choice"}
        if _forces_a_dropped_mcp_tool(params.get("tool_choice"), sent_tools, kept_tools, functions)
        else cleared
    )
    return type(params)(
        {key: (list(kept_tools) if key == "tools" else value) for key, value in params.items() if key not in removed}
    )
