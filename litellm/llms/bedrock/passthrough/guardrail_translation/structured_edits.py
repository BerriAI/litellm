from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.llms.bedrock.chat.converse_to_openai import (
    ConverseRow,
    JsonPath,
    JsonSlot,
    Slot,
    TextSlot,
    as_json_array,
    as_json_object,
    breadth_first,
)
from litellm.types.llms.openai import ChatCompletionTextObject

Edit: TypeAlias = tuple[JsonPath, str]
_JsonPair: TypeAlias = tuple[JsonPath, object, object]
_TEXT_PARTS: Final = TypeAdapter(tuple[ChatCompletionTextObject, ...])
_PARSED_JSON: Final = TypeAdapter(object)


def _without(mapping: Mapping[str, object], key: str) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(((name, value) for name, value in mapping.items() if name != key), key=lambda item: item[0]))


def _without_content(message: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return _without(message, "content")


def _text_parts(message: Mapping[str, object]) -> tuple[str, ...] | None:
    content: Final = message.get("content")
    if isinstance(content, str):
        return (content,)
    try:
        parts: Final = _TEXT_PARTS.validate_python(content)
    except ValidationError:
        return None
    return tuple(part["text"] for part in parts)


def _merged(children: Sequence[tuple[Edit, ...] | None]) -> tuple[Edit, ...] | None:
    if any(child is None for child in children):
        return None
    return tuple(chain.from_iterable(child for child in children if child is not None))


def _paired_children(path: JsonPath, original: object, returned: object) -> tuple[_JsonPair, ...] | None:
    original_object: Final = as_json_object(original)
    returned_object: Final = as_json_object(returned)
    if original_object is not None and returned_object is not None:
        if original_object.keys() != returned_object.keys():
            return None
        return tuple(((*path, key), value, returned_object[key]) for key, value in original_object.items())
    original_items: Final = as_json_array(original)
    returned_items: Final = as_json_array(returned)
    if original_items is None or returned_items is None or len(original_items) != len(returned_items):
        return None
    return tuple(((*path, index), value, returned_items[index]) for index, value in enumerate(original_items))


def _pair_children(pair: _JsonPair | None) -> tuple[_JsonPair | None, ...]:
    if pair is None:
        return ()
    path, original, returned = pair
    if original == returned or (isinstance(original, str) and isinstance(returned, str)):
        return ()
    children: Final = _paired_children(path, original, returned)
    return (None,) if children is None else children


def _leaf_edit(pair: _JsonPair | None) -> Edit | None:
    match pair:
        case (path, str() as original, str() as returned) if original != returned:
            return path, returned
        case _:
            return None


def _json_edits(path: JsonPath, original: object, returned: object) -> tuple[Edit, ...] | None:
    pairs: Final = tuple(breadth_first((path, original, returned), _pair_children))
    if any(pair is None for pair in pairs):
        return None
    return tuple(edit for pair in pairs if (edit := _leaf_edit(pair)) is not None)


def _slot_edits(slot: Slot, text: str) -> tuple[Edit, ...] | None:
    match slot:
        case TextSlot():
            return () if text == slot.text else ((slot.path, text),)
        case JsonSlot():
            try:
                parsed: Final = _PARSED_JSON.validate_json(text)
            except ValidationError:
                return None
            return _json_edits(slot.path, slot.value, parsed)


def _call_function(call: object) -> Mapping[str, object] | None:
    fields: Final = as_json_object(call)
    return as_json_object(fields.get("function")) if fields is not None else None


def _call_edits(slot: JsonSlot, original: object, returned: object) -> tuple[Edit, ...] | None:
    if original == returned:
        return ()
    original_call: Final = as_json_object(original)
    returned_call: Final = as_json_object(returned)
    original_function: Final = _call_function(original)
    returned_function: Final = _call_function(returned)
    if original_call is None or returned_call is None or original_function is None or returned_function is None:
        return None
    arguments: Final = returned_function.get("arguments")
    if (
        not isinstance(arguments, str)
        or _without(returned_call, "function") != _without(original_call, "function")
        or _without(returned_function, "arguments") != _without(original_function, "arguments")
    ):
        return None
    return _slot_edits(slot, arguments)


def _calls_edits(slots: Sequence[JsonSlot], original: object, returned: object) -> tuple[Edit, ...] | None:
    original_calls: Final = as_json_array(original)
    returned_calls: Final = as_json_array(returned)
    if original_calls is None or returned_calls is None or len(original_calls) != len(returned_calls):
        return None
    return _merged(tuple(_call_edits(*call) for call in zip(slots, original_calls, returned_calls)))


def _carries_no_text(content: object) -> bool:
    match content:
        case None | "" | []:
            return True
        case _:
            return False


def _content_edits(row: ConverseRow, original: object, returned: Mapping[str, object]) -> tuple[Edit, ...] | None:
    if returned.get("content") == original:
        return ()
    if not row.content and _carries_no_text(returned.get("content")):
        return ()
    texts: Final = _text_parts(returned)
    if texts is None or len(texts) != len(row.content):
        return None
    return _merged(tuple(_slot_edits(slot, text) for slot, text in zip(row.content, texts)))


def _field(message: Mapping[str, object], key: str) -> object:
    return message.get(key)


def _row_edits(row: ConverseRow, returned: Mapping[str, object]) -> tuple[Edit, ...] | None:
    if returned == row.message:
        return ()
    calls: Final = _field(row.message, "tool_calls")
    call_edits: Final = _calls_edits(row.calls, calls, returned.get("tool_calls")) if row.calls else ()
    comparable: Final = {**returned, "tool_calls": calls} if row.calls else returned
    if call_edits is None or _without_content(comparable) != _without_content(row.message):
        return None
    return _merged((call_edits, _content_edits(row, _field(row.message, "content"), comparable)))


def _same_content(expected: Mapping[str, object], returned: Mapping[str, object]) -> bool:
    if expected.get("content") == returned.get("content"):
        return True
    expected_parts: Final = _text_parts(expected)
    returned_parts: Final = _text_parts(returned)
    return (
        expected_parts is not None
        and returned_parts is not None
        and "\n".join(expected_parts) == "\n".join(returned_parts)
    )


def _reproduces(expected: ConverseRow, returned: Mapping[str, object] | None) -> bool:
    return (
        returned is not None
        and _without_content(returned) == _without_content(expected.message)
        and _same_content(expected.message, returned)
    )


def structured_edits(
    rows: Sequence[ConverseRow],
    scoped_indices: Sequence[int],
    returned: Sequence[object],
    texts_applied_rows: Callable[[], tuple[ConverseRow, ...]],
) -> tuple[Edit, ...] | None:
    if len(returned) != len(scoped_indices):
        return None
    returned_rows: Final = tuple(as_json_object(message) for message in returned)
    proposed: Final = tuple(
        _row_edits(rows[index], row) if row is not None else None for index, row in zip(scoped_indices, returned_rows)
    )
    unmapped: Final = tuple(
        (index, row) for index, row, edits in zip(scoped_indices, returned_rows, proposed) if edits is None
    )
    if unmapped:
        view: Final = texts_applied_rows()
        if len(view) != len(rows) or not all(_reproduces(view[index], row) for index, row in unmapped):
            return None
    return tuple(chain.from_iterable(edits for edits in proposed if edits is not None))
