from collections.abc import Iterable, Mapping, Sequence
from itertools import accumulate, chain, groupby
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.responses.litellm_completion_transformation.transformation import (
    NAMESPACE_DESCRIPTION_SEPARATOR,
    LiteLLMCompletionResponsesConfig,
)

Tool: TypeAlias = Mapping[str, object]
IndexedKey: TypeAlias = tuple[str, int]

_TOOL_ADAPTER: Final = TypeAdapter(dict[str, object])
_CHAT_TOOL_TOP_LEVEL_KEYS: Final = frozenset({"type", "function"})


def _as_tool(value: object) -> Tool | None:
    try:
        return _TOOL_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _validated_tools(values: Iterable[object]) -> tuple[Tool, ...]:
    validated: Final = tuple(map(_as_tool, values))
    dropped: Final = sum(tool is None for tool in validated)
    if dropped:
        verbose_logger.warning("Dropping %d guardrail-returned tools that are not objects", dropped)
    return tuple(tool for tool in validated if tool is not None)


def _is_function(tool: Tool) -> bool:
    return tool.get("type") == "function"


def _chat_tool_key(tool: Tool) -> str:
    tool_type: Final = str(tool.get("type") or "")
    function: Final = _as_tool(tool.get("function"))
    if function is not None:
        return f"{tool_type}:{function.get('name') or ''}"
    return f"{tool_type}:{tool.get('server_label') or tool.get('name') or ''}"


def _indexed_keys(tools: Sequence[Tool]) -> tuple[IndexedKey, ...]:
    keys: Final = tuple(_chat_tool_key(tool) for tool in tools)
    positions_by_key: Final = groupby(sorted(range(len(keys)), key=keys.__getitem__), key=keys.__getitem__)
    ordinal_by_position: Final = MappingProxyType(
        {position: ordinal for _, positions in positions_by_key for ordinal, position in enumerate(positions)}
    )
    return tuple((key, ordinal_by_position[position]) for position, key in enumerate(keys))


def _namespace_members(namespace: Tool) -> tuple[Tool, ...]:
    members: Final = namespace.get("tools")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        return ()
    return tuple(member for member in map(_as_tool, members) if member is not None)


def _function_fields(tool: Tool) -> Tool:
    function: Final = _as_tool(tool.get("function"))
    return function if function is not None else MappingProxyType({})


def _without_namespace_prefix(key: str, value: object, prefix: str) -> object:
    if key != "description" or not isinstance(value, str) or not value.startswith(prefix):
        return value
    return value[len(prefix) :]


def _rebuilt_member(member: Tool, flattened: Tool, guardrailed: Tool, namespace_description: str) -> Tool:
    flattened_function: Final = _function_fields(flattened)
    prefix: Final = f"{namespace_description}{NAMESPACE_DESCRIPTION_SEPARATOR}" if namespace_description else ""
    changed_function: Final = MappingProxyType(
        {
            key: _without_namespace_prefix(key, value, prefix)
            for key, value in _function_fields(guardrailed).items()
            if flattened_function.get(key) != value
        }
    )
    changed_extras: Final = MappingProxyType(
        {
            key: value
            for key, value in guardrailed.items()
            if key not in _CHAT_TOOL_TOP_LEVEL_KEYS and flattened.get(key) != value
        }
    )
    return {**member, **changed_extras, **changed_function}  # mutable-ok: json.dumps rejects MappingProxyType


def _rebuilt_function_members(
    function_members: Sequence[Tool],
    flattened_group: Sequence[Tool],
    group_keys: Sequence[IndexedKey],
    guardrailed_by_key: Mapping[IndexedKey, Tool],
    namespace_description: str,
) -> tuple[Tool | None, ...]:
    return tuple(
        None
        if key not in guardrailed_by_key
        else member
        if guardrailed_by_key[key] == flattened
        else _rebuilt_member(member, flattened, guardrailed_by_key[key], namespace_description)
        for member, flattened, key in zip(function_members, flattened_group, group_keys)
    )


def _rebuilt_namespace(
    original: Tool,
    members: Sequence[Tool],
    flattened_group: Sequence[Tool],
    group_keys: Sequence[IndexedKey],
    guardrailed_by_key: Mapping[IndexedKey, Tool],
) -> tuple[Tool, ...]:
    namespace_description: Final = str(original.get("description") or "")
    rebuilt_functions: Final = iter(
        _rebuilt_function_members(
            tuple(member for member in members if _is_function(member)),
            flattened_group,
            group_keys,
            guardrailed_by_key,
            namespace_description,
        )
    )
    rebuilt_members: Final = tuple(
        rebuilt
        for rebuilt in (next(rebuilt_functions) if _is_function(member) else member for member in members)
        if rebuilt is not None
    )
    if not rebuilt_members:
        return ()
    return ({**original, "tools": list(rebuilt_members)},)  # mutable-ok: json.dumps needs a plain dict and list


def _merged_original(
    original: Tool,
    flattened_group: Sequence[Tool],
    group_keys: Sequence[IndexedKey],
    guardrailed_by_key: Mapping[IndexedKey, Tool],
) -> tuple[Tool, ...]:
    if not group_keys:
        return (original,)
    guardrailed_group: Final = tuple(guardrailed_by_key[key] for key in group_keys if key in guardrailed_by_key)
    if guardrailed_group == tuple(flattened_group):
        return (original,)
    members: Final = _namespace_members(original) if original.get("type") == "namespace" else ()
    if members and sum(map(_is_function, members)) == len(flattened_group):
        return _rebuilt_namespace(original, members, flattened_group, group_keys, guardrailed_by_key)
    if not guardrailed_group:
        return ()
    return tuple(
        LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(guardrailed_group)
    )


def merge_guardrailed_tools(
    original_tools: Sequence[Tool],
    flattened_groups: Sequence[Sequence[Tool]],
    guardrailed_tools: Iterable[object],
) -> tuple[Tool, ...]:
    guardrailed: Final = _validated_tools(guardrailed_tools)
    flattened_keys: Final = _indexed_keys(tuple(chain.from_iterable(flattened_groups)))
    guardrailed_keys: Final = _indexed_keys(guardrailed)
    guardrailed_by_key: Final = MappingProxyType(dict(zip(guardrailed_keys, guardrailed)))
    group_ends: Final = tuple(accumulate(len(group) for group in flattened_groups))
    group_key_slices: Final = tuple(
        flattened_keys[end - len(group) : end] for group, end in zip(flattened_groups, group_ends)
    )
    merged_originals: Final = chain.from_iterable(
        _merged_original(original, group, group_keys, guardrailed_by_key)
        for original, group, group_keys in zip(original_tools, flattened_groups, group_key_slices)
    )
    owned_keys: Final = frozenset(flattened_keys)
    appended: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(
        tuple(tool for key, tool in zip(guardrailed_keys, guardrailed) if key not in owned_keys)
    )
    return tuple(chain(merged_originals, appended))
