import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

_WRAPPER_FIELDS: Final = ("function", "custom")
_ARGUMENT_FIELDS: Final = ("arguments", "input", "args")
_JSON_VALUE: Final = TypeAdapter(object)

ValuePath: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _JsonNumber:
    text: str


@dataclass(frozen=True, slots=True)
class _JsonObject:
    pairs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _JsonTextArguments:
    tree: object


@dataclass(frozen=True, slots=True)
class _StructuredArguments:
    tree: object


@dataclass(frozen=True, slots=True)
class _RawTextArguments:
    text: str


@dataclass(frozen=True, slots=True)
class _ArgumentsTooDeep:
    pass


_Arguments: TypeAlias = _JsonTextArguments | _StructuredArguments | _RawTextArguments
_Copyable: TypeAlias = Mapping[str, object] | BaseModel


@dataclass(frozen=True, slots=True)
class _WritableArguments:
    tool_call: _Copyable
    wrapper_field: str | None
    source: _Copyable
    field: str
    arguments: _Arguments


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    value_path: ValuePath | None
    value_prefix: str


@dataclass(frozen=True, slots=True)
class ToolCallText:
    tool_call: object
    blocks: tuple[TextBlock, ...]
    writable: _WritableArguments | None


def _field(value: object, name: str) -> object:
    mapping: Final = as_str_mapping(value)
    if mapping is not None:
        return mapping.get(name)
    return getattr(value, name, None)


def _copyable(value: object) -> _Copyable | None:
    return value if isinstance(value, BaseModel) else as_str_mapping(value)


def _with_field(value: _Copyable, name: str, new_value: object) -> _Copyable:
    if isinstance(value, BaseModel):
        return value.model_copy(update={name: new_value})
    return {**value, name: new_value}


def _json_object(pairs: list[tuple[str, object]]) -> _JsonObject:  # mutable-ok: the json decoder hook's type
    return _JsonObject(tuple(pairs))


def _decoded_json(text: str) -> object:
    return _JSON_VALUE.validate_python(
        json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_int=_JsonNumber,
            parse_float=_JsonNumber,
            parse_constant=_JsonNumber,
        )
    )


def _text_arguments(text: str, tree: object) -> _Arguments:
    if isinstance(tree, (_JsonObject, str)) or as_object_list(tree) is not None:
        return _JsonTextArguments(tree)
    return _RawTextArguments(text)


def _parsed_arguments(arguments: object) -> _Arguments | _ArgumentsTooDeep:
    try:
        if isinstance(arguments, str):
            return _text_arguments(arguments, _decoded_json(arguments))
        return _StructuredArguments(_decoded_json(json.dumps(arguments, ensure_ascii=False, default=str)))
    except RecursionError:
        return _ArgumentsTooDeep()
    except ValueError:
        return _RawTextArguments(str(arguments))


def _flattened(groups: Iterable[tuple[TextBlock, ...] | None]) -> tuple[TextBlock, ...] | None:
    materialized: Final = tuple(groups)
    if any(group is None for group in materialized):
        return None
    return tuple(chain.from_iterable(group for group in materialized if group is not None))


def _leaf_text(node: object) -> str | None:
    match node:
        case _JsonNumber(text=text):
            return text
        case str() if node:
            return node
        case _:
            return None


def _child_blocks(node: object, path: ValuePath, max_depth: int) -> tuple[TextBlock, ...] | None:
    match node:
        case _JsonObject(pairs=pairs):
            return _flattened(
                _node_blocks(value, (*path, index), max_depth, key) for index, (key, value) in enumerate(pairs)
            )
        case _:
            items: Final = as_object_list(node) or ()
            return _flattened(_node_blocks(item, (*path, index), max_depth, "") for index, item in enumerate(items))


def _node_blocks(node: object, path: ValuePath, max_depth: int, key: str) -> tuple[TextBlock, ...] | None:
    if len(path) > max_depth:
        return None
    leaf: Final = _leaf_text(node)
    if leaf is not None:
        prefix: Final = f"{key}: " if key else ""
        return (TextBlock(prefix + leaf, path, prefix),)
    children: Final = _child_blocks(node, path, max_depth)
    if children is None:
        return None
    return ((TextBlock(key, None, ""),) if key else ()) + children


def _argument_blocks(arguments: _Arguments | None, max_depth: int) -> tuple[TextBlock, ...] | None:
    match arguments:
        case None:
            return ()
        case _RawTextArguments(text=text):
            return (TextBlock(text, (), ""),) if text else ()
        case _JsonTextArguments(tree=tree) | _StructuredArguments(tree=tree):
            return _node_blocks(tree, (), max_depth, "")


def _writable_arguments(
    tool_call: object,
    wrapper_field: str | None,
    source: object,
    arguments_field: str | None,
    arguments: _Arguments | None,
) -> _WritableArguments | None:
    if arguments is None or arguments_field is None:
        return None
    copyable_tool_call: Final = _copyable(tool_call)
    copyable_source: Final = _copyable(source)
    if copyable_tool_call is None or copyable_source is None:
        return None
    return _WritableArguments(
        tool_call=copyable_tool_call,
        wrapper_field=wrapper_field,
        source=copyable_source,
        field=arguments_field,
        arguments=arguments,
    )


def tool_call_text(tool_call: object, max_depth: int) -> ToolCallText | None:
    wrapper_field: Final = next((field for field in _WRAPPER_FIELDS if _field(tool_call, field)), None)
    source: Final = tool_call if wrapper_field is None else _field(tool_call, wrapper_field)
    arguments_field: Final = next((field for field in _ARGUMENT_FIELDS if _field(source, field) is not None), None)
    raw_arguments: Final = None if arguments_field is None else _field(source, arguments_field)
    arguments: Final = None if raw_arguments is None or raw_arguments == "" else _parsed_arguments(raw_arguments)
    if isinstance(arguments, _ArgumentsTooDeep):
        return None
    argument_blocks: Final = _argument_blocks(arguments, max_depth)
    if argument_blocks is None:
        return None
    name: Final = _field(source, "name")
    return ToolCallText(
        tool_call=tool_call,
        blocks=((TextBlock(name, None, ""),) if isinstance(name, str) and name else ()) + argument_blocks,
        writable=_writable_arguments(tool_call, wrapper_field, source, arguments_field, arguments),
    )


def tool_call_scan_key(tool_call: ToolCallText) -> str:
    return json.dumps([block.text for block in tool_call.blocks], ensure_ascii=False)


def _block_offsets(tool_calls: Sequence[ToolCallText]) -> tuple[int, ...]:
    return tuple(accumulate((len(tool_call.blocks) for tool_call in tool_calls), initial=0))


def _value_prefixes_of(tool_call: ToolCallText, offset: int) -> tuple[tuple[int, str], ...]:
    if tool_call.writable is None:
        return ()
    return tuple(
        (offset + index, block.value_prefix)
        for index, block in enumerate(tool_call.blocks)
        if block.value_path is not None
    )


def maskable_value_prefixes(tool_calls: Sequence[ToolCallText]) -> Mapping[int, str]:
    return MappingProxyType(
        dict(
            chain.from_iterable(
                _value_prefixes_of(tool_call, offset)
                for tool_call, offset in zip(tool_calls, _block_offsets(tool_calls), strict=False)
            )
        )
    )


def _json_text_with_masks(node: object, masks: Mapping[ValuePath, str], path: ValuePath) -> str:
    masked: Final = masks.get(path)
    if masked is not None:
        return json.dumps(masked, ensure_ascii=False)
    match node:
        case _JsonObject(pairs=pairs):
            members: Final = (
                f"{json.dumps(key, ensure_ascii=False)}: {_json_text_with_masks(value, masks, (*path, index))}"
                for index, (key, value) in enumerate(pairs)
            )
            return "{" + ", ".join(members) + "}"
        case _JsonNumber(text=text):
            return text
        case _:
            items: Final = as_object_list(node)
            if items is None:
                return json.dumps(node, ensure_ascii=False)
            return "[" + ", ".join(_json_text_with_masks(item, masks, (*path, i)) for i, item in enumerate(items)) + "]"


def _value_with_masks(node: object, masks: Mapping[ValuePath, str], path: ValuePath) -> object:
    masked: Final = masks.get(path)
    if masked is not None:
        return masked
    match node:
        case _JsonObject(pairs=pairs):
            return {key: _value_with_masks(value, masks, (*path, index)) for index, (key, value) in enumerate(pairs)}
        case _JsonNumber(text=text):
            return _JSON_VALUE.validate_python(json.loads(text))
        case _:
            items: Final = as_object_list(node)
            if items is None:
                return node
            return [_value_with_masks(item, masks, (*path, index)) for index, item in enumerate(items)]


def _arguments_with_masks(arguments: _Arguments, masks: Mapping[ValuePath, str]) -> object:
    match arguments:
        case _JsonTextArguments(tree=tree):
            return _json_text_with_masks(tree, masks, ())
        case _StructuredArguments(tree=tree):
            return _value_with_masks(tree, masks, ())
        case _RawTextArguments(text=text):
            return masks.get((), text)


def _masked_tool_call(tool_call: ToolCallText, masked_texts: Sequence[str]) -> object:
    writable: Final = tool_call.writable
    masks: Final = MappingProxyType(
        {
            block.value_path: masked_text.removeprefix(block.value_prefix)
            for block, masked_text in zip(tool_call.blocks, masked_texts, strict=True)
            if block.value_path is not None and masked_text != block.text
        }
    )
    if writable is None or not masks:
        return tool_call.tool_call
    masked_source: Final = _with_field(
        writable.source, writable.field, _arguments_with_masks(writable.arguments, masks)
    )
    if writable.wrapper_field is None:
        return masked_source
    return _with_field(writable.tool_call, writable.wrapper_field, masked_source)


def masked_tool_calls(tool_calls: Sequence[ToolCallText], masked_texts: Sequence[str]) -> tuple[object, ...]:
    return tuple(
        _masked_tool_call(tool_call, masked_texts[offset : offset + len(tool_call.blocks)])
        for tool_call, offset in zip(tool_calls, _block_offsets(tool_calls), strict=False)
    )


def block_texts(tool_calls: Sequence[ToolCallText]) -> tuple[str, ...]:
    return tuple(chain.from_iterable((block.text for block in tool_call.blocks) for tool_call in tool_calls))
