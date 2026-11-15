from collections.abc import Iterable, Mapping, Sequence
from itertools import accumulate, chain, groupby
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.responses.litellm_completion_transformation.custom_tools import custom_tool_grammar_suffix
from litellm.responses.litellm_completion_transformation.transformation import (
    NAMESPACE_DESCRIPTION_SEPARATOR,
    NAMESPACE_MEMBER_TYPES_WITH_CHAT_TOOLS,
    LiteLLMCompletionResponsesConfig,
)

Tool: TypeAlias = Mapping[str, object]
IndexedKey: TypeAlias = tuple[str, int]

_TOOL_ADAPTER: Final = TypeAdapter(dict[str, object])
_DEFAULT_FUNCTION_FIELDS: Final = (("strict", False), ("description", ""))
_CHAT_TOOL_TOP_LEVEL_KEYS: Final = frozenset({"type", "function"})


def _as_tool(value: object) -> Tool | None:
    candidate: Final = value.model_dump(exclude_unset=True) if isinstance(value, BaseModel) else value
    try:
        return _TOOL_ADAPTER.validate_python(candidate)
    except ValidationError:
        return None


def _validated_tools(values: Iterable[object]) -> tuple[Tool, ...]:
    validated: Final = tuple(map(_as_tool, values))
    dropped: Final = sum(tool is None for tool in validated)
    if dropped:
        verbose_logger.warning("Dropping %d guardrail-returned tools that are not objects", dropped)
    return tuple(tool for tool in validated if tool is not None)


def _has_chat_tool(member: Tool) -> bool:
    return member.get("type") in NAMESPACE_MEMBER_TYPES_WITH_CHAT_TOOLS


def _chat_tool_key(tool: Tool) -> str:
    tool_type: Final = str(tool.get("type") or "")
    function: Final = _as_tool(tool.get("function"))
    if function is not None:
        return f"{tool_type}:{function.get('name') or ''}"
    return f"{tool_type}:{tool.get('server_label') or tool.get('name') or ''}"


def _key_positions(keys: Sequence[str]) -> Mapping[str, tuple[int, ...]]:
    ordered: Final = sorted(range(len(keys)), key=keys.__getitem__)
    return MappingProxyType({key: tuple(positions) for key, positions in groupby(ordered, key=keys.__getitem__)})


def _positional_ordinals(positions: Mapping[str, tuple[int, ...]]) -> Mapping[int, int]:
    return {position: ordinal for ordinal, position in chain.from_iterable(map(enumerate, positions.values()))}


def _indexed_keys(keys: Sequence[str], positions: Mapping[str, tuple[int, ...]]) -> tuple[IndexedKey, ...]:
    ordinal_by_position: Final = _positional_ordinals(positions)
    return tuple((key, ordinal_by_position[position]) for position, key in enumerate(keys))


def _comparable(tool: Tool, *, with_description: bool) -> Tool:
    fields: Final = {key: value for key, value in tool.items() if value is not None and key != "function"}
    if "function" not in tool:
        return fields
    function: Final = {
        key: value
        for key, value in _function_fields(tool).items()
        if value is not None
        and (key, value) not in _DEFAULT_FUNCTION_FIELDS
        and (with_description or key != "description")
    }
    return {**fields, "function": function}


def _leftmost_alignment(originals: Sequence[Tool], kept: Sequence[Tool]) -> tuple[int, ...] | None:
    remaining: Final = iter(enumerate(originals))
    aligned: Final = tuple(next((ordinal for ordinal, original in remaining if original == tool), -1) for tool in kept)
    return None if -1 in aligned else aligned


def _alignment(
    originals: Sequence[Tool], kept: Sequence[Tool], *, with_description: bool, interchangeable: bool
) -> tuple[int, ...] | None:
    original_views: Final = tuple(_comparable(tool, with_description=with_description) for tool in originals)
    kept_views: Final = tuple(_comparable(tool, with_description=with_description) for tool in kept)
    leftmost: Final = _leftmost_alignment(original_views, kept_views)
    if leftmost is None or interchangeable:
        return leftmost
    rightmost: Final = _leftmost_alignment(original_views[::-1], kept_views[::-1])
    if rightmost is None:
        return None
    return leftmost if leftmost == tuple(len(originals) - 1 - ordinal for ordinal in reversed(rightmost)) else None


def _kept_ordinals(
    flattened: Sequence[Tool], guardrailed: Sequence[Tool], *, interchangeable: bool
) -> tuple[int, ...] | None:
    exact: Final = _alignment(flattened, guardrailed, with_description=True, interchangeable=interchangeable)
    if exact is not None:
        return exact
    return _alignment(flattened, guardrailed, with_description=False, interchangeable=interchangeable)


def _positioned_ordinals(
    flattened: Sequence[Tool],
    guardrailed: Sequence[Tool],
    flattened_positions: Sequence[int],
    guardrailed_positions: Sequence[int],
    top_level_functions: frozenset[int],
) -> tuple[tuple[int, int], ...] | None:
    ordinals: Final = _kept_ordinals(
        tuple(map(flattened.__getitem__, flattened_positions)),
        tuple(map(guardrailed.__getitem__, guardrailed_positions)),
        interchangeable=top_level_functions.issuperset(flattened_positions),
    )
    return None if ordinals is None else tuple(zip(guardrailed_positions, ordinals))


def _guardrailed_keys(
    flattened: Sequence[Tool],
    flattened_positions: Mapping[str, tuple[int, ...]],
    guardrailed: Sequence[Tool],
    top_level_functions: frozenset[int],
) -> tuple[IndexedKey, ...] | None:
    keys: Final = tuple(map(_chat_tool_key, guardrailed))
    positions: Final = _key_positions(keys)
    realigned: Final = tuple(
        _positioned_ordinals(flattened, guardrailed, flattened_positions[key], kept, top_level_functions)
        for key, kept in positions.items()
        if len(kept) < len(flattened_positions.get(key, ()))
    )
    if any(group is None for group in realigned):
        return None
    ordinal_by_position: Final = {
        **_positional_ordinals(positions),
        **dict(chain.from_iterable(group for group in realigned if group)),
    }
    return tuple((key, ordinal_by_position[position]) for position, key in enumerate(keys))


def _namespace_members(namespace: Tool) -> tuple[Tool, ...]:
    members: Final = namespace.get("tools")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        return ()
    return tuple(member for member in map(_as_tool, members) if member is not None)


def _function_fields(tool: Tool) -> Tool:
    function: Final = _as_tool(tool.get("function"))
    return function if function is not None else MappingProxyType({})


def _member_description(key: str, value: object, prefix: str, suffix: str) -> object:
    if key != "description" or not isinstance(value, str):
        return value
    return value.replace(prefix, "", 1).replace(suffix, "", 1)


def _rebuilt_member(member: Tool, flattened: Tool, guardrailed: Tool, namespace_description: str) -> Tool:
    flattened_function: Final = _function_fields(flattened)
    prefix: Final = f"{namespace_description}{NAMESPACE_DESCRIPTION_SEPARATOR}" if namespace_description else ""
    suffix: Final = custom_tool_grammar_suffix(member.get("format")) if member.get("type") == "custom" else ""
    changed_function: Final = MappingProxyType(
        {
            key: _member_description(key, value, prefix, suffix)
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
    return {**member, **changed_extras, **changed_function}


def _rebuilt_flattened_members(
    flattened_members: Sequence[Tool],
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
        for member, flattened, key in zip(flattened_members, flattened_group, group_keys)
    )


def _rebuilt_namespace(
    original: Tool,
    members: Sequence[Tool],
    flattened_group: Sequence[Tool],
    group_keys: Sequence[IndexedKey],
    guardrailed_by_key: Mapping[IndexedKey, Tool],
) -> tuple[Tool, ...]:
    namespace_description: Final = str(original.get("description") or "")
    rebuilt_flattened: Final = iter(
        _rebuilt_flattened_members(
            tuple(member for member in members if _has_chat_tool(member)),
            flattened_group,
            group_keys,
            guardrailed_by_key,
            namespace_description,
        )
    )
    rebuilt_members: Final = tuple(
        rebuilt
        for rebuilt in (next(rebuilt_flattened) if _has_chat_tool(member) else member for member in members)
        if rebuilt is not None
    )
    if not rebuilt_members:
        return ()
    return ({**original, "tools": list(rebuilt_members)},)


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
    if members and sum(map(_has_chat_tool, members)) == len(flattened_group):
        return _rebuilt_namespace(original, members, flattened_group, group_keys, guardrailed_by_key)
    if not guardrailed_group:
        return ()
    return tuple(
        LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(guardrailed_group)
    )


def _top_level_function_positions(
    original_tools: Sequence[Tool], flattened_groups: Sequence[Sequence[Tool]]
) -> frozenset[int]:
    is_function: Final = chain.from_iterable(
        (original.get("type") == "function",) * len(group) for original, group in zip(original_tools, flattened_groups)
    )
    return frozenset(position for position, function in enumerate(is_function) if function)


def merge_guardrailed_tools(
    original_tools: Sequence[Tool],
    flattened_groups: Sequence[Sequence[Tool]],
    guardrailed_tools: Iterable[object],
) -> tuple[Tool, ...] | None:
    guardrailed: Final = _validated_tools(guardrailed_tools)
    flattened: Final = tuple(chain.from_iterable(flattened_groups))
    flattened_names: Final = tuple(map(_chat_tool_key, flattened))
    flattened_positions: Final = _key_positions(flattened_names)
    flattened_keys: Final = _indexed_keys(flattened_names, flattened_positions)
    guardrailed_keys: Final = _guardrailed_keys(
        flattened, flattened_positions, guardrailed, _top_level_function_positions(original_tools, flattened_groups)
    )
    if guardrailed_keys is None:
        return None
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
