import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import accumulate, chain, groupby, repeat
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.base_llm.guardrail_translation.utils import merge_returned_tools_into_request_tools, openai_tool_name
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


def _view(tool: Tool, *, with_description: bool) -> str:
    return json.dumps(_comparable(tool, with_description=with_description), sort_keys=True, default=str)


def _field_leaves(key: str, value: object) -> tuple[tuple[str, object], ...]:
    children: Final = as_str_mapping(value)
    if not children:
        return ((key, value),)
    return tuple((f"{key}.{child_key}", child) for child_key, child in children.items())


def _leaves(tool: Tool) -> Mapping[str, object]:
    return dict(chain.from_iterable(_field_leaves(key, value) for key, value in tool.items()))


def _difference(left: Tool, right: Tool) -> int:
    left_leaves: Final = _leaves(_comparable(left, with_description=True))
    right_leaves: Final = _leaves(_comparable(right, with_description=True))
    return sum(left_leaves.get(key) != right_leaves.get(key) for key in left_leaves.keys() | right_leaves.keys())


Anchor: TypeAlias = tuple[int, int]
OrderBreaks: TypeAlias = Callable[[int, int], int]
Distance: TypeAlias = Callable[[int, int], tuple[int, int]]


def _order_breaks(
    anchors: Sequence[Anchor],
    guardrailed_positions: Sequence[int],
    flattened_positions: Sequence[int],
    kept: int,
    candidate: int,
) -> int:
    guardrailed_position: Final = guardrailed_positions[kept]
    flattened_position: Final = flattened_positions[candidate]
    return sum(
        (anchor_guardrailed < guardrailed_position) != (anchor_flattened < flattened_position)
        for anchor_guardrailed, anchor_flattened in anchors
    )


def _kept_distance(
    guardrailed: Sequence[Tool], originals: Sequence[Tool], order_breaks: OrderBreaks, kept: int, candidate: int
) -> tuple[int, int]:
    return (_difference(guardrailed[kept], originals[candidate]), order_breaks(kept, candidate))


@dataclass(frozen=True, slots=True)
class _Match:
    ordinal: int
    rebuilt: bool


@dataclass(frozen=True, slots=True)
class _Resolution:
    pairs: tuple[tuple[int, _Match], ...] = ()
    passed: tuple[int, ...] = ()
    remaining: tuple[int, ...] = ()


def _paired(kept: Sequence[int], candidates: Sequence[int], *, rebuilt: bool) -> tuple[tuple[int, _Match], ...]:
    return tuple((index, _Match(ordinal, rebuilt)) for index, ordinal in zip(kept, candidates))


Pairs: TypeAlias = tuple[tuple[int, int], ...]


def _with_nearest(candidates: Sequence[int], distance: Distance, pairs: Pairs, index: int) -> Pairs:
    claimed: Final = frozenset(candidate for _, candidate in pairs)
    nearest: Final = min(
        (candidate for candidate in candidates if candidate not in claimed),
        key=lambda candidate: (distance(index, candidate), candidate),
    )
    return (*pairs, (index, nearest))


def _nearest_pairs(kept: Sequence[int], candidates: Sequence[int], distance: Distance) -> Pairs:
    no_pairs: Final[Pairs] = ()
    return reduce(partial(_with_nearest, candidates, distance), kept, no_pairs)


def _class_resolution(
    kept: tuple[int, ...], candidates: tuple[int, ...], rebuildable: Sequence[Tool | None], distance: Distance
) -> _Resolution | None:
    if not kept or len(kept) >= len(candidates):
        return _Resolution(
            pairs=_paired(kept, candidates, rebuilt=False),
            passed=kept[len(candidates) :],
            remaining=candidates[len(kept) :],
        )
    if any(rebuildable[candidate] is None for candidate in candidates):
        return None
    rebuilt: Final = any(rebuildable[candidate] != rebuildable[candidates[0]] for candidate in candidates)
    claimed: Final = frozenset(candidate for _, candidate in _nearest_pairs(kept, candidates, distance))
    return _Resolution(
        pairs=_paired(kept, sorted(claimed), rebuilt=rebuilt),
        remaining=tuple(candidate for candidate in candidates if candidate not in claimed),
    )


def _classes(tools: Sequence[Tool], indexes: Sequence[int], *, with_description: bool) -> Mapping[str, tuple[int, ...]]:
    views: Final = tuple(_view(tools[index], with_description=with_description) for index in indexes)
    return MappingProxyType(
        {view: tuple(map(indexes.__getitem__, positions)) for view, positions in _key_positions(views).items()}
    )


def _round(
    originals: Sequence[Tool],
    guardrailed: Sequence[Tool],
    rebuildable: Sequence[Tool | None],
    previous: _Resolution,
    distance: Distance,
    *,
    with_description: bool,
) -> _Resolution | None:
    candidate_classes: Final = _classes(originals, previous.remaining, with_description=with_description)
    kept_classes: Final = _classes(guardrailed, previous.passed, with_description=with_description)
    resolved: Final = tuple(
        _class_resolution(kept, candidate_classes.get(view, ()), rebuildable, distance)
        for view, kept in kept_classes.items()
    )
    if any(resolution is None for resolution in resolved):
        return None
    results: Final = tuple(resolution for resolution in resolved if resolution is not None)
    unclaimed: Final = (candidates for view, candidates in candidate_classes.items() if view not in kept_classes)
    return _Resolution(
        pairs=(*previous.pairs, *chain.from_iterable(result.pairs for result in results)),
        passed=tuple(sorted(chain.from_iterable(result.passed for result in results))),
        remaining=tuple(sorted(chain(chain.from_iterable(result.remaining for result in results), *unclaimed))),
    )


def _content_matches(
    originals: Sequence[Tool],
    guardrailed: Sequence[Tool],
    rebuildable: Sequence[Tool | None],
    order_breaks: OrderBreaks,
) -> tuple[_Match | None, ...] | None:
    start: Final = _Resolution(passed=tuple(range(len(guardrailed))), remaining=tuple(range(len(originals))))
    distance: Final = partial(_kept_distance, guardrailed, originals, order_breaks)
    exact: Final = _round(originals, guardrailed, rebuildable, start, distance, with_description=True)
    if exact is None:
        return None
    loose: Final = _round(originals, guardrailed, rebuildable, exact, distance, with_description=False)
    if loose is None:
        return None
    leftover: Final = _class_resolution(loose.passed, loose.remaining, rebuildable, distance)
    if leftover is None:
        return None
    match_by_index: Final = MappingProxyType(dict(chain(loose.pairs, leftover.pairs)))
    return tuple(map(match_by_index.get, range(len(guardrailed))))


def _positioned_matches(
    flattened: Sequence[Tool],
    guardrailed: Sequence[Tool],
    flattened_positions: Sequence[int],
    guardrailed_positions: Sequence[int],
    rebuildable_originals: Mapping[int, Tool],
    anchors: Sequence[Anchor],
) -> tuple[tuple[int, _Match], ...] | None:
    matches: Final = _content_matches(
        tuple(map(flattened.__getitem__, flattened_positions)),
        tuple(map(guardrailed.__getitem__, guardrailed_positions)),
        tuple(map(rebuildable_originals.get, flattened_positions)),
        partial(_order_breaks, anchors, guardrailed_positions, flattened_positions),
    )
    if matches is None:
        return None
    unmatched: Final = tuple(position for position, match in zip(guardrailed_positions, matches) if match is None)
    added: Final = MappingProxyType(
        {
            position: _Match(len(flattened_positions) + offset, rebuilt=False)
            for offset, position in enumerate(unmatched)
        }
    )
    return tuple(
        (position, added[position] if match is None else match)
        for position, match in zip(guardrailed_positions, matches)
    )


@dataclass(frozen=True, slots=True)
class _GuardrailedKeys:
    keys: tuple[IndexedKey, ...]
    rebuilt: frozenset[IndexedKey]


def _guardrailed_keys(
    flattened: Sequence[Tool],
    flattened_positions: Mapping[str, tuple[int, ...]],
    guardrailed: Sequence[Tool],
    rebuildable_originals: Mapping[int, Tool],
) -> _GuardrailedKeys | None:
    keys: Final = tuple(map(_chat_tool_key, guardrailed))
    positions: Final = _key_positions(keys)
    anchors: Final = tuple(
        (kept[0], flattened_positions[key][0])
        for key, kept in positions.items()
        if len(kept) == 1 and len(flattened_positions.get(key, ())) == 1
    )
    realigned: Final = tuple(
        _positioned_matches(flattened, guardrailed, flattened_positions[key], kept, rebuildable_originals, anchors)
        for key, kept in positions.items()
        if len(flattened_positions.get(key, ())) > 1
    )
    if any(group is None for group in realigned):
        return None
    matches: Final = tuple(chain.from_iterable(group for group in realigned if group))
    ordinal_by_position: Final = {
        **_positional_ordinals(positions),
        **{position: match.ordinal for position, match in matches},
    }
    indexed: Final = tuple((key, ordinal_by_position[position]) for position, key in enumerate(keys))
    return _GuardrailedKeys(
        keys=indexed, rebuilt=frozenset(indexed[position] for position, match in matches if match.rebuilt)
    )


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
    rebuilt_keys: frozenset[IndexedKey],
) -> tuple[Tool, ...]:
    if not group_keys:
        return (original,)
    guardrailed_group: Final = tuple(guardrailed_by_key[key] for key in group_keys if key in guardrailed_by_key)
    if guardrailed_group == tuple(flattened_group) and rebuilt_keys.isdisjoint(group_keys):
        return (original,)
    members: Final = _namespace_members(original) if original.get("type") == "namespace" else ()
    if members and sum(map(_has_chat_tool, members)) == len(flattened_group):
        return _rebuilt_namespace(original, members, flattened_group, group_keys, guardrailed_by_key)
    if not guardrailed_group:
        return ()
    return tuple(
        LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(guardrailed_group)
    )


def _rebuilds_from_chat_tools(original: Tool, group: Sequence[Tool]) -> bool:
    return original.get("type") == "function" or tuple(group) == (original,)


def _rebuildable_originals(
    original_tools: Sequence[Tool], flattened_groups: Sequence[Sequence[Tool]]
) -> Mapping[int, Tool]:
    owners: Final = chain.from_iterable(
        repeat(original if _rebuilds_from_chat_tools(original, group) else None, len(group))
        for original, group in zip(original_tools, flattened_groups)
    )
    return MappingProxyType({position: original for position, original in enumerate(owners) if original is not None})


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
    realigned: Final = _guardrailed_keys(
        flattened, flattened_positions, guardrailed, _rebuildable_originals(original_tools, flattened_groups)
    )
    if realigned is None:
        return None
    guardrailed_keys: Final = realigned.keys
    guardrailed_by_key: Final = MappingProxyType(dict(zip(guardrailed_keys, guardrailed)))
    group_ends: Final = tuple(accumulate(len(group) for group in flattened_groups))
    group_key_slices: Final = tuple(
        flattened_keys[end - len(group) : end] for group, end in zip(flattened_groups, group_ends)
    )
    merged_originals: Final = chain.from_iterable(
        _merged_original(original, group, group_keys, guardrailed_by_key, realigned.rebuilt)
        for original, group, group_keys in zip(original_tools, flattened_groups, group_key_slices)
    )
    owned_keys: Final = frozenset(flattened_keys)
    appended: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(
        tuple(tool for key, tool in zip(guardrailed_keys, guardrailed) if key not in owned_keys)
    )
    return tuple(chain(merged_originals, appended))


def append_returned_tools(
    original_tools: Sequence[Tool],
    flattened_groups: Sequence[Sequence[Tool]],
    returned_tools: Iterable[object],
) -> tuple[Tool, ...]:
    request_chat_tools: Final = tuple(chain.from_iterable(flattened_groups))
    additions: Final = merge_returned_tools_into_request_tools(
        request_tools=request_chat_tools, returned_tools=_validated_tools(returned_tools), tool_name=openai_tool_name
    )[len(request_chat_tools) :]
    return (
        *original_tools,
        *LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(additions),
    )
