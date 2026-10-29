import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # loaded tool specs are forwarded as the client sent them

from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
)
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import (
    ALL_RESPONSES_API_TOOL_PARAMS,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ResponseInputParam,
)

from .custom_tools import function_chat_tool
from .tool_search import CLIENT_TOOL_SEARCH_FUNCTION_NAME, has_client_tool_search

__all__ = [
    "ClientToolSearchBridge",
    "bridge_client_tool_search",
    "client_tool_search_chat_tool",
    "client_tool_search_loaded_tools",
]

_DEFAULT_SEARCH_DESCRIPTION: Final = (
    "Search the deferred tools by keyword. The matching tools are loaded and become callable on the next turn."
)
_DEFAULT_SEARCH_PARAMETERS: Final = MappingProxyType(
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for deferred tools."},
            "limit": {"type": "integer", "description": "Maximum number of tools to return. Defaults to 8."},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
)


class _ClientToolSearchItem(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    execution: str | None = None
    call_id: str | None = None
    arguments: object = None
    tools: tuple[Mapping[str, object], ...] = ()


class _ToolSpec(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    name: str = ""
    description: str = ""
    defer_loading: object = None
    tools: tuple[Mapping[str, object], ...] = ()


class _ClientToolSearchTool(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    description: str = ""
    parameters: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _LoadedTool:
    namespace: str | None
    namespace_description: str
    spec: Mapping[str, object]

    @property
    def callable_name(self) -> str:
        name: Final = str(self.spec.get("name") or "")
        return name if self.namespace is None else f"{self.namespace}__{name}"


@dataclass(frozen=True, slots=True)
class ClientToolSearchBridge:
    input: str | ResponseInputParam
    tools: tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]
    bridged: bool


def client_tool_search_chat_tool(tool: Mapping[str, object]) -> ChatCompletionToolParam:
    spec: Final = _ClientToolSearchTool.model_validate(tool)
    function: Final = ChatCompletionToolParamFunctionChunk(
        name=CLIENT_TOOL_SEARCH_FUNCTION_NAME,
        description=spec.description or _DEFAULT_SEARCH_DESCRIPTION,
        parameters=dict(spec.parameters or _DEFAULT_SEARCH_PARAMETERS),
    )
    return function_chat_tool(function, None, False)


def _client_tool_search_item(item: object) -> _ClientToolSearchItem | None:
    try:
        parsed: Final = _ClientToolSearchItem.model_validate(item)
    except ValidationError:
        return None
    if parsed.type not in ("tool_search_call", "tool_search_output") or parsed.execution != "client":
        return None
    return parsed


_INPUT_ITEMS: Final = TypeAdapter(tuple[object, ...])


def _searched_tools(item: object) -> tuple[Mapping[str, object], ...]:
    parsed: Final = _client_tool_search_item(item)
    if parsed is None or parsed.type != "tool_search_output" or not parsed.call_id:
        return ()
    return parsed.tools


def client_tool_search_loaded_tools(input: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(input, (list, tuple)):
        return ()
    return tuple(chain.from_iterable(map(_searched_tools, _INPUT_ITEMS.validate_python(input))))


def _tool_spec(tool: object) -> _ToolSpec | None:
    try:
        return _ToolSpec.model_validate(tool)
    except ValidationError:
        return None


def _as_tool_param(spec: Mapping[str, object]) -> ALL_RESPONSES_API_TOOL_PARAMS:
    return cast("ALL_RESPONSES_API_TOOL_PARAMS", spec)  # cast-ok: client tool specs are forwarded as sent


def _without_deferral(spec: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in spec.items() if key != "defer_loading"}


def _loaded_tools(tool: Mapping[str, object]) -> tuple[_LoadedTool, ...]:
    spec: Final = _tool_spec(tool)
    if spec is None:
        return ()
    if spec.type != "namespace":
        return (_LoadedTool(namespace=None, namespace_description="", spec=_without_deferral(tool)),)
    return tuple(
        _LoadedTool(namespace=spec.name, namespace_description=spec.description, spec=_without_deferral(member))
        for member in spec.tools
    )


def _all_loaded_tools(tools: Iterable[Mapping[str, object]]) -> tuple[_LoadedTool, ...]:
    return tuple(chain.from_iterable(map(_loaded_tools, tools)))


def _callable_names(tool: object) -> tuple[str, ...]:
    spec: Final = _tool_spec(tool)
    if spec is None:
        return ()
    if spec.type != "namespace":
        return (spec.name,)
    return tuple(f"{spec.name}__{member.get('name') or ''}" for member in spec.tools)


def _first_occurrences(loaded: Sequence[_LoadedTool], taken: frozenset[str]) -> tuple[_LoadedTool, ...]:
    first_index: Final = {tool.callable_name: index for index, tool in reversed(tuple(enumerate(loaded)))}
    return tuple(
        tool
        for index, tool in enumerate(loaded)
        if first_index[tool.callable_name] == index and tool.callable_name not in taken
    )


def _regrouped(loaded: Sequence[_LoadedTool]) -> tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]:
    namespaces: Final = {tool.namespace: tool.namespace_description for tool in reversed(loaded) if tool.namespace}
    ordered_namespaces: Final = tuple(dict.fromkeys(tool.namespace for tool in loaded if tool.namespace))
    top_level: Final = tuple(tool.spec for tool in loaded if tool.namespace is None)
    grouped: Final = tuple(
        {
            "type": "namespace",
            "name": namespace,
            "description": namespaces[namespace],
            "tools": tuple(tool.spec for tool in loaded if tool.namespace == namespace),
        }
        for namespace in ordered_namespaces
    )
    return tuple(_as_tool_param(tool) for tool in (*top_level, *grouped))


def _is_deferred(tool: object) -> bool:
    spec: Final = _tool_spec(tool)
    return spec is not None and spec.defer_loading is True


def _without_deferred(tool: ALL_RESPONSES_API_TOOL_PARAMS) -> ALL_RESPONSES_API_TOOL_PARAMS | None:
    spec: Final = _tool_spec(tool)
    if spec is None:
        return tool
    if spec.type != "namespace":
        return None if spec.defer_loading is True else tool
    eager_members: Final = tuple(member for member in spec.tools if not _is_deferred(member))
    if len(eager_members) == len(spec.tools):
        return tool
    if not eager_members:
        return None
    return _as_tool_param({**tool, "tools": eager_members})


def _search_output_text(loaded: Sequence[_LoadedTool]) -> str:
    if not loaded:
        return "No tools matched the search."
    return "Loaded these tools, which are now callable: " + ", ".join(tool.callable_name for tool in loaded)


def _search_arguments_json(arguments: object) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments or {})


def _bridged_item(item: ResponseInputItemParam) -> ResponseInputItemParam | None:
    parsed: Final = _client_tool_search_item(item)
    if parsed is None:
        return item
    if not parsed.call_id:
        return None
    if parsed.type == "tool_search_call":
        return ResponseFunctionToolCallParam(
            type="function_call",
            call_id=parsed.call_id,
            name=CLIENT_TOOL_SEARCH_FUNCTION_NAME,
            arguments=_search_arguments_json(parsed.arguments),
        )
    return FunctionCallOutput(
        type="function_call_output", call_id=parsed.call_id, output=_search_output_text(_all_loaded_tools(parsed.tools))
    )


def bridge_client_tool_search(
    input: str | ResponseInputParam,
    tools: Sequence[ALL_RESPONSES_API_TOOL_PARAMS] | None,
    loaded_earlier: Sequence[Mapping[str, object]] = (),
) -> ClientToolSearchBridge:
    existing: Final = tuple(tools or ())
    searches_on_client: Final = has_client_tool_search(existing)
    items: Final = () if isinstance(input, str) else tuple(input)
    search_items: Final = tuple(parsed for item in items if (parsed := _client_tool_search_item(item)) is not None)
    if not searches_on_client and not search_items:
        return ClientToolSearchBridge(input=input, tools=existing, bridged=False)
    eager: Final = (
        tuple(kept for tool in existing if (kept := _without_deferred(tool)) is not None)
        if searches_on_client
        else existing
    )
    loaded: Final = _all_loaded_tools((*loaded_earlier, *client_tool_search_loaded_tools(input)))
    taken: Final = frozenset(chain.from_iterable(map(_callable_names, eager)))
    bridged_input: Final = (
        input if isinstance(input, str) else [bridged for item in input if (bridged := _bridged_item(item)) is not None]
    )
    return ClientToolSearchBridge(
        input=bridged_input,
        tools=(*eager, *_regrouped(_first_occurrences(loaded, taken))),
        bridged=True,
    )
