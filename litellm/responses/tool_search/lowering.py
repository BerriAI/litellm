import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.types.llms.openai import ResponseInputParam, ToolChoice

TOOL_SEARCH_FUNCTION_NAME: Final = "tool_search"
_DEFAULT_TOOL_SEARCH_DESCRIPTION: Final = (
    "Search the client tool catalog and load the matching tools for the next call."
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
# a replayed search may only load tools the client runs itself, never hosted ones the proxy would run
_LOADABLE_TOOL_TYPES: Final = frozenset({"function", "custom", "namespace"})
_LOADABLE_MEMBER_TYPES: Final = frozenset({"function", "custom"})

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _Typed(BaseModel):
    type: str = ""
    name: str | None = None


class _ClientToolSearchDeclaration(BaseModel):
    type: Literal["tool_search"]
    execution: Literal["client"]
    description: str | None = None
    parameters: object = None


class _ReplayedToolSearchCall(BaseModel):
    type: Literal["tool_search_call"]
    call_id: str | None = None
    arguments: object = None
    status: str | None = None


class _ReplayedToolSearchOutput(BaseModel):
    type: Literal["tool_search_output"]
    call_id: str | None = None
    tools: tuple[dict[str, object], ...] = ()


class _Namespace(BaseModel):
    type: Literal["namespace"]
    tools: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class LoweredToolSearchRequest:
    input: str | Sequence[object]
    tools: tuple[object, ...]
    tool_choice: ToolChoice | None


@dataclass(frozen=True, slots=True)
class ToolSearchFunctionNameTaken:
    pass


ToolSearchLowering: TypeAlias = LoweredToolSearchRequest | ToolSearchFunctionNameTaken


def _plain(value: object) -> object:
    return value.model_dump(exclude_none=True) if isinstance(value, BaseModel) else value


def _parsed(model: type[_ModelT], value: object) -> _ModelT | None:
    try:
        return model.model_validate(_plain(value))
    except ValidationError:
        return None


def _json_object(value: object) -> dict[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(_plain(value))
    except ValidationError:
        return None


def _is_tool_search_item(item: object) -> bool:
    return _parsed(_ReplayedToolSearchCall, item) is not None or _parsed(_ReplayedToolSearchOutput, item) is not None


def _declares_client_tool_search(tools: Sequence[object]) -> bool:
    return any(_parsed(_ClientToolSearchDeclaration, tool) is not None for tool in tools)


def needs_tool_search_lowering(input: str | ResponseInputParam, tools: Sequence[object] | None) -> bool:
    if _declares_client_tool_search(tools or ()):
        return True
    return not isinstance(input, str) and any(_is_tool_search_item(item) for item in input)


def _object_schema(parameters: object) -> dict[str, object]:
    schema: Final = _json_object(parameters)
    if schema is None:
        return {"type": "object", "properties": {}}
    if "type" in schema:
        return schema
    return {"type": "object", **schema}


def _lowered_tool(tool: object) -> object:
    declaration: Final = _parsed(_ClientToolSearchDeclaration, tool)
    if declaration is None:
        return tool
    return {
        "type": "function",
        "name": TOOL_SEARCH_FUNCTION_NAME,
        "description": declaration.description or _DEFAULT_TOOL_SEARCH_DESCRIPTION,
        "parameters": (
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            if declaration.parameters is None
            else _object_schema(declaration.parameters)
        ),
        "strict": False,
    }


def _loaded_definition(tool: dict[str, object]) -> dict[str, object]:
    loaded: Final = {key: value for key, value in tool.items() if key != "defer_loading"}
    kind: Final = _parsed(_Typed, tool)
    if kind is not None and kind.type == "function":
        return {**loaded, "parameters": _object_schema(tool.get("parameters"))}
    return loaded


def _has_type(tool: object, types: frozenset[str]) -> bool:
    kind: Final = _parsed(_Typed, tool)
    return kind is not None and kind.type in types


def _loaded_tool(tool: dict[str, object]) -> dict[str, object]:
    namespace: Final = _parsed(_Namespace, tool)
    if namespace is None:
        return _loaded_definition(tool)
    members: Final = [
        _loaded_definition(member) for member in namespace.tools if _has_type(member, _LOADABLE_MEMBER_TYPES)
    ]
    return {**_loaded_definition(tool), "tools": members}


def _loadable_tools(output: _ReplayedToolSearchOutput) -> tuple[dict[str, object], ...]:
    return tuple(_loaded_tool(tool) for tool in output.tools if _has_type(tool, _LOADABLE_TOOL_TYPES))


def _visible_loaded_tool(tool: dict[str, object]) -> dict[str, object]:
    if _parsed(_Namespace, tool) is None:
        return tool
    return {key: value for key, value in tool.items() if key in ("type", "name", "description")}


def _lowered_item(item: object) -> object:
    call: Final = _parsed(_ReplayedToolSearchCall, item)
    if call is not None:
        return {
            "type": "function_call",
            "call_id": call.call_id,
            "name": TOOL_SEARCH_FUNCTION_NAME,
            "arguments": json.dumps({} if call.arguments is None else call.arguments),
            **({} if call.status is None else {"status": call.status}),
        }
    output: Final = _parsed(_ReplayedToolSearchOutput, item)
    if output is None:
        return item
    visible_tools: Final = [_visible_loaded_tool(tool) for tool in _loadable_tools(output)]
    return {
        "type": "function_call_output",
        "call_id": output.call_id,
        "output": json.dumps({"tools": visible_tools}, separators=(",", ":")),
    }


def _loaded_tools(items: Sequence[object]) -> tuple[dict[str, object], ...]:
    outputs: Final = tuple(_parsed(_ReplayedToolSearchOutput, item) for item in items)
    return tuple(chain.from_iterable(_loadable_tools(output) for output in outputs if output is not None))


def _merge_key(index: int, tool: object) -> tuple[str, str]:
    kind: Final = _parsed(_Typed, tool)
    if kind is not None and kind.type in ("function", "custom", "namespace") and kind.name is not None:
        return kind.type, kind.name
    return "position", str(index)


def _namespace_members(group: Sequence[object]) -> tuple[dict[str, object], ...]:
    namespaces: Final = (_parsed(_Namespace, tool) for tool in group)
    return tuple(chain.from_iterable(namespace.tools for namespace in namespaces if namespace is not None))


def _merged_group(kind: str, group: Sequence[object]) -> object:
    latest: Final = group[-1]
    latest_object: Final = _json_object(latest) if kind == "namespace" and len(group) > 1 else None
    if latest_object is None:
        return latest
    return {**latest_object, "tools": list(_merged_tools(_namespace_members(group)))}


def _merged_tools(tools: Sequence[object]) -> tuple[object, ...]:
    keys: Final = tuple(_merge_key(index, tool) for index, tool in enumerate(tools))
    keyed: Final = tuple(zip(keys, tools, strict=True))
    return tuple(
        _merged_group(key[0], tuple(tool for tool_key, tool in keyed if tool_key == key)) for key in dict.fromkeys(keys)
    )


def _lowered_tool_choice(tool_choice: ToolChoice | None) -> ToolChoice | None:
    kind: Final = None if isinstance(tool_choice, str) else _parsed(_Typed, tool_choice)
    if kind is None or kind.type != "tool_search":
        return tool_choice
    return {"type": "function", "name": TOOL_SEARCH_FUNCTION_NAME}


def _declares_function(tools: Sequence[object] | None, name: str) -> bool:
    return ("function", name) in (_merge_key(index, tool) for index, tool in enumerate(tools or ()))


def lower_tool_search_request(
    input: str | ResponseInputParam,
    tools: Sequence[object] | None,
    tool_choice: ToolChoice | None,
) -> ToolSearchLowering:
    declared: Final = tuple(tools or ())
    if _declares_client_tool_search(declared) and _declares_function(declared, TOOL_SEARCH_FUNCTION_NAME):
        return ToolSearchFunctionNameTaken()
    items: Final = () if isinstance(input, str) else tuple(input)
    lowered_tools: Final = _merged_tools((*(_lowered_tool(tool) for tool in declared), *_loaded_tools(items)))
    return LoweredToolSearchRequest(
        input=input if isinstance(input, str) else [_lowered_item(item) for item in items],
        tools=lowered_tools,
        tool_choice=_lowered_tool_choice(tool_choice),
    )
