"""
Google GenAI generateContent handler for Unified Guardrails.

Scans what the provider or the client reads: system instructions, text and code
parts, tool results, and functionCall parts, which go to the guardrail as tool
calls the way chat sends its tool calls, whether parts arrive as a list or as one
object. Structured values are scanned as one JSON text with numbers quoted so a
mask can replace them, and a changed value is decoded back in place with the
numbers the guardrail left alone keeping their type. A request rewrite that
cannot be written back faithfully is rejected (a 500, as on chat) instead of
being sent unrewritten. Requests and responses may be dicts or google-genai SDK
objects, and raw SSE stream frames are scanned for detection without rewriting.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby, pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    stream_item_field,
    tool_calls_in_guardrail_scope,
    unappliable_request_rewrite,
)
from litellm.llms.custom_httpx.sse_lines import split_sse_lines, split_sse_lines_keepends
from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_EMPTY_REQUEST_DATA: Final[Mapping[str, object]] = MappingProxyType({})
_DECODED_JSON: Final = TypeAdapter(object)
_OBJECT_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_SYSTEM_INSTRUCTION_KEYS: Final = ("systemInstruction", "system_instruction")
_JSON_SCALAR: Final = re.compile(r'"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
_JSON_STRING_ESCAPE: Final = re.compile(r'\\(["\\]|[bfnrt]|u[0-9a-fA-F]{4})')
_CONTROL_CHARACTER_ESCAPES: Final = MappingProxyType({"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"})
_PARTIAL_ARG_VALUE_FIELDS: Final = (
    ("stringValue", "string_value"),
    ("numberValue", "number_value"),
    ("boolValue", "bool_value"),
    ("nullValue", "null_value"),
)
_PARTIAL_ARG_VALUE_NAMES: Final = tuple(chain.from_iterable(_PARTIAL_ARG_VALUE_FIELDS))


@dataclass(frozen=True, slots=True)
class _PartField:
    keys: tuple[str, str]
    name: str
    is_json: bool


_CODE_PART_FIELD: Final = _PartField(keys=("executableCode", "executable_code"), name="code", is_json=False)
_TOOL_CALL_PART_FIELD: Final = _PartField(keys=("toolCall", "tool_call"), name="args", is_json=True)
_TOOL_RESULT_PART_FIELDS: Final = (
    _PartField(keys=("codeExecutionResult", "code_execution_result"), name="output", is_json=False),
    _PartField(keys=("functionResponse", "function_response"), name="response", is_json=True),
    _PartField(keys=("toolResponse", "tool_response"), name="response", is_json=True),
)


@dataclass(frozen=True, slots=True)
class _PartScope:
    message_parts: bool
    tool_calls: bool
    tool_results: bool
    system_instruction: bool


_FULL_PART_SCOPE: Final = _PartScope(message_parts=True, tool_calls=True, tool_results=True, system_instruction=True)


@dataclass(frozen=True, slots=True)
class _TextSlot:
    container: object
    key: str | int
    text: str
    json_value: object = None


def _json_object(value: object) -> Mapping[str, object] | None:
    fields: Final = as_str_mapping(value)
    return fields if isinstance(fields, dict) else None


def _as_list_or_tuple(value: object) -> Sequence[object] | None:
    return value if isinstance(value, (list, tuple)) else None


def _field(container: object, name: str) -> object | None:
    fields: Final = _json_object(container)
    return fields.get(name) if fields is not None else getattr(container, name, None)


def _present_fields(container: object, names: Sequence[str]) -> tuple[object, ...]:
    return tuple(value for name in names if (value := _field(container, name)) is not None)


def _write_field(container: object, key: str | int, value: object) -> None:
    match container:
        case dict():
            container[key] = value  # rebind-ok: guardrail write-back edits the caller's request in place
        case list():
            if isinstance(key, int):
                container[key] = value  # rebind-ok: guardrail write-back edits the caller's request in place
        case _:
            if isinstance(key, str):
                setattr(container, key, value)


def _is_writable(container: object, key: str | int) -> bool:
    match container:
        case dict():
            return True
        case list():
            return isinstance(key, int)
        case _:
            return isinstance(key, str)


def _repeated_field(value: object) -> Sequence[object]:
    fields: Final = _json_object(value)
    if fields is not None:
        return (fields,)
    return _as_list_or_tuple(value) or ()


def _string_slot(container: object, key: str | int, value: object) -> tuple[_TextSlot, ...]:
    return (_TextSlot(container=container, key=key, text=value),) if isinstance(value, str) and value else ()


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _raw_control_character(escape: re.Match[str]) -> str:
    code: Final = escape.group(1)
    if code in ('"', "\\"):
        return escape.group(0)
    return _CONTROL_CHARACTER_ESCAPES.get(code) or chr(int(code[1:], 16))


def _scan_scalar(scalar: re.Match[str]) -> str:
    text: Final = scalar.group(0)
    if not text.startswith('"'):
        return f'"{text}"'
    return _JSON_STRING_ESCAPE.sub(_raw_control_character, text)


def _json_scan_text(value: object) -> str:
    return _JSON_SCALAR.sub(_scan_scalar, _json_text(value))


def _with_original_numbers(original: object, guardrailed: object, depth: int) -> object:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return original if _json_scan_text(original) == _json_scan_text(guardrailed) else guardrailed
    if isinstance(original, (int, float)) and not isinstance(original, bool):
        return original if guardrailed == _json_text(original) else guardrailed
    original_fields: Final = _json_object(original)
    guardrailed_fields: Final = _json_object(guardrailed)
    if original_fields is not None and guardrailed_fields is not None:
        return {
            key: _with_original_numbers(original_fields.get(key), value, depth + 1)
            for key, value in guardrailed_fields.items()
        }
    original_items: Final = _as_list_or_tuple(original)
    guardrailed_items: Final = _as_list_or_tuple(guardrailed)
    if original_items is None or guardrailed_items is None:
        return guardrailed
    if len(original_items) != len(guardrailed_items):
        originals_by_text: Final = {_json_scan_text(item): item for item in original_items}
        return [originals_by_text.get(_json_scan_text(value), value) for value in guardrailed_items]
    return [
        _with_original_numbers(item, value, depth + 1)
        for item, value in zip(original_items, guardrailed_items, strict=True)
    ]


def _json_slot(container: object, key: str | int, value: object) -> tuple[_TextSlot, ...]:
    if not value:
        return ()
    return (_TextSlot(container=container, key=key, text=_json_scan_text(value), json_value=value),)


def _part_field_slots(part: object, part_field: _PartField) -> tuple[_TextSlot, ...]:
    slot: Final = _json_slot if part_field.is_json else _string_slot
    return tuple(
        chain.from_iterable(
            slot(container, part_field.name, _field(container, part_field.name))
            for container in _present_fields(part, part_field.keys)
        )
    )


def _part_text_slots(parts: Sequence[object], index: int) -> tuple[_TextSlot, ...]:
    part: Final = parts[index]
    return (
        _string_slot(parts, index, part) if isinstance(part, str) else _string_slot(part, "text", _field(part, "text"))
    )


def _scoped_part_fields(scope: _PartScope) -> tuple[_PartField, ...]:
    return (
        *((_CODE_PART_FIELD,) if scope.message_parts else ()),
        *((_TOOL_CALL_PART_FIELD,) if scope.tool_calls else ()),
        *(_TOOL_RESULT_PART_FIELDS if scope.tool_results else ()),
    )


def _part_slots(parts: Sequence[object], index: int, scope: _PartScope) -> tuple[_TextSlot, ...]:
    text_slots: Final = _part_text_slots(parts, index) if scope.message_parts else ()
    field_slots: Final = chain.from_iterable(
        _part_field_slots(parts[index], part_field) for part_field in _scoped_part_fields(scope)
    )
    return (*text_slots, *field_slots)


def _content_part_slots(content: object, scope: _PartScope) -> tuple[_TextSlot, ...]:
    parts: Final = _repeated_field(_field(content, "parts"))
    return tuple(chain.from_iterable(_part_slots(parts, index, scope) for index in range(len(parts))))


def _request_contents(data: Mapping[str, object], scope: _PartScope) -> tuple[object, ...]:
    system_instructions: Final = (
        chain.from_iterable(
            _present_fields(container, _SYSTEM_INSTRUCTION_KEYS) for container in (data, data.get("config"))
        )
        if scope.system_instruction
        else ()
    )
    return (*system_instructions, *_repeated_field(data.get("contents")))


def _request_part_scope(guardrail: "CustomGuardrail") -> _PartScope:
    return _PartScope(
        message_parts=not effective_scan_only_tool_results_for_guardrail(guardrail),
        tool_calls=tool_calls_in_guardrail_scope(guardrail),
        tool_results=not effective_skip_tool_message_for_guardrail(guardrail),
        system_instruction=not effective_skip_system_message_for_guardrail(guardrail),
    )


def _request_text_slots(data: Mapping[str, object], scope: _PartScope) -> tuple[_TextSlot, ...]:
    return tuple(chain.from_iterable(_content_part_slots(content, scope) for content in _request_contents(data, scope)))


def _candidate_contents(response: object) -> tuple[object, ...]:
    candidates: Final = _as_list_or_tuple(_field(response, "candidates"))
    if candidates is None:
        return ()
    return tuple(_field(candidate, "content") for candidate in candidates)


def _response_part_slots(response: object) -> tuple[_TextSlot, ...]:
    return tuple(
        chain.from_iterable(_content_part_slots(content, _FULL_PART_SCOPE) for content in _candidate_contents(response))
    )


def _slot_texts(text_slots: Sequence[_TextSlot]) -> tuple[str, ...]:
    return tuple(slot.text for slot in text_slots)


def _camel_or_snake(container: object, camel_case: str, snake_case: str) -> object | None:
    camel_value: Final = _field(container, camel_case)
    return camel_value if camel_value is not None else _field(container, snake_case)


def _function_call_in_part(part: object) -> object | None:
    return _camel_or_snake(part, "functionCall", "function_call")


def _content_function_calls(content: object) -> tuple[object, ...]:
    calls: Final = (_function_call_in_part(part) for part in _repeated_field(_field(content, "parts")))
    return tuple(call for call in calls if call is not None)


def _request_function_calls(data: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        chain.from_iterable(_content_function_calls(content) for content in _repeated_field(data.get("contents")))
    )


def _function_calls_in_response(response: object) -> tuple[object, ...]:
    return tuple(chain.from_iterable(_content_function_calls(content) for content in _candidate_contents(response)))


def _continues(function_call: object) -> bool:
    return _camel_or_snake(function_call, "willContinue", "will_continue") is True


def _starts_function_call(function_call: object) -> bool:
    return bool(_field(function_call, "name")) or bool(as_str_mapping(_field(function_call, "args")))


def _function_call_groups(function_calls: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    starts: Final = tuple(
        index
        for index in range(len(function_calls))
        if index == 0 or not _continues(function_calls[index - 1]) or _starts_function_call(function_calls[index])
    )
    return tuple(tuple(function_calls[start:end]) for start, end in pairwise((*starts, len(function_calls))))


def _partial_args(function_call: object) -> Sequence[object]:
    return as_object_list(_camel_or_snake(function_call, "partialArgs", "partial_args")) or ()


def _partial_arg_path(partial_arg: object) -> str:
    path: Final = _camel_or_snake(partial_arg, "jsonPath", "json_path")
    return path if isinstance(path, str) else ""


def _partial_arg_text(partial_arg: object) -> str:
    values: Final = (_camel_or_snake(partial_arg, camel, snake) for camel, snake in _PARTIAL_ARG_VALUE_FIELDS)
    value: Final = next((value for value in values if value is not None), None)
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _partial_args_by_path(group: Sequence[object]) -> Mapping[str, tuple[object, ...]]:
    partial_args: Final = tuple(chain.from_iterable(_partial_args(function_call) for function_call in group))
    paths: Final = tuple(_partial_arg_path(partial_arg) for partial_arg in partial_args)
    rank: Final = {path: position for position, path in enumerate(dict.fromkeys(paths))}
    order: Final = tuple(sorted(range(len(paths)), key=lambda index: rank[paths[index]]))
    return {
        path: tuple(partial_args[index] for index in indices)
        for path, indices in groupby(order, key=lambda index: paths[index])
    }


def _partial_text(partial_args: Iterable[object]) -> str:
    return "".join(_partial_arg_text(partial_arg) for partial_arg in partial_args)


def _partial_values_by_path(group: Sequence[object]) -> Mapping[str, str]:
    return {path: _partial_text(partial_args) for path, partial_args in _partial_args_by_path(group).items()}


def _group_field(group: Sequence[object], name: str) -> str | None:
    values: Final = (_field(function_call, name) for function_call in group)
    return next((value for value in values if isinstance(value, str) and value), None)


def _group_arguments(group: Sequence[object]) -> Mapping[str, object]:
    args: Final = (as_str_mapping(_field(function_call, "args")) or {} for function_call in group)
    return {**dict(chain.from_iterable(mapping.items() for mapping in args)), **_partial_values_by_path(group)}


def _group_arguments_text(group: Sequence[object]) -> str:
    return json.dumps(_group_arguments(group), ensure_ascii=False, default=str)


def _tool_call(group: Sequence[object], index: int) -> ChatCompletionToolCallChunk:
    return ChatCompletionToolCallChunk(
        id=_group_field(group, "id"),
        type="function",
        function={"name": _group_field(group, "name"), "arguments": _group_arguments_text(group)},
        index=index,
    )


def _tool_calls_payload(groups: Sequence[Sequence[object]]) -> tuple[ChatCompletionToolCallChunk, ...]:
    return tuple(_tool_call(group, index) for index, group in enumerate(groups))


def _set_field(container: object, name: str, value: object) -> None:
    if isinstance(container, dict):
        container[name] = value  # rebind-ok: guardrail write-back rewrites the caller's part in place
        return
    setattr(container, name, value)


def _clear_field(container: object, name: str) -> None:
    if isinstance(container, dict) and name in container:
        del container[name]  # rebind-ok: guardrail write-back rewrites the caller's part in place
    elif getattr(container, name, None) is not None:
        setattr(container, name, None)


def _string_value_field(partial_arg: object) -> str:
    snake_case: Final = _field(partial_arg, "json_path") is not None or _field(partial_arg, "string_value") is not None
    return "string_value" if snake_case else "stringValue"


def _write_partial_text(partial_arg: object, text: str) -> None:
    string_field: Final = _string_value_field(partial_arg)
    for name in _PARTIAL_ARG_VALUE_NAMES:
        _clear_field(partial_arg, name)
    _set_field(partial_arg, string_field, text)


def _write_partial_path(
    partial_args: Sequence[object], returned: object, sent_text: str, guardrail_name: str | None
) -> None:
    if not isinstance(returned, str):
        raise unappliable_request_rewrite(guardrail_name)
    if returned == sent_text:
        return
    for index, partial_arg in enumerate(partial_args):
        _write_partial_text(partial_arg, returned if index == 0 else "")


def _write_group_args(group: Sequence[object], args: Mapping[str, object], guardrail_name: str | None) -> None:
    partial_args_by_path: Final = _partial_args_by_path(group)
    head_args: Final = {key: value for key, value in args.items() if key not in partial_args_by_path}
    if head_args != (as_str_mapping(_field(group[0], "args")) or {}):
        _set_field(group[0], "args", head_args)
    for path, partial_args in partial_args_by_path.items():
        _write_partial_path(partial_args, args.get(path), _partial_text(partial_args), guardrail_name)


def _returned_function_field(returned_tool_call: object, name: str) -> object:
    return stream_item_field(stream_item_field(returned_tool_call, "function"), name)


def _write_group_name(group: Sequence[object], returned_tool_call: object) -> None:
    name: Final = _returned_function_field(returned_tool_call, "name")
    if not isinstance(name, str) or not name or name == _group_field(group, "name"):
        return
    holder: Final = next((function_call for function_call in group if _field(function_call, "name")), group[0])
    _set_field(holder, "name", name)


def _guardrailed_args(returned_tool_call: object, guardrail_name: str | None) -> Mapping[str, object]:
    arguments: Final = _returned_function_field(returned_tool_call, "arguments")
    parsed: Final = _parse_json_dict_or_none(arguments) if isinstance(arguments, str) else None
    if parsed is None:
        raise unappliable_request_rewrite(guardrail_name)
    return parsed


def _returned_tool_calls(
    groups: Sequence[Sequence[object]], guardrailed_inputs: GenericGuardrailAPIInputs
) -> Sequence[object] | None:
    return as_object_list(stream_item_field(guardrailed_inputs, "tool_calls")) if groups else None


def _text_count_differs(text_slots: Sequence[_TextSlot], returned_texts: Sequence[str] | None) -> bool:
    return bool(text_slots and returned_texts and len(returned_texts) != len(text_slots))


def _write_back_function_calls(
    groups: Sequence[Sequence[object]], returned_tool_calls: Sequence[object] | None, guardrail_name: str | None
) -> None:
    if returned_tool_calls is None:
        return
    for group, returned in zip(groups, returned_tool_calls, strict=False):
        if _returned_function_field(returned, "arguments") != _group_arguments_text(group):
            _write_group_args(group, _guardrailed_args(returned, guardrail_name), guardrail_name)
        _write_group_name(group, returned)


def _texts_payload(
    texts: Sequence[str],
) -> list[str]:  # mutable-ok: GenericGuardrailAPIInputs.texts is declared list[str]
    return list(texts)


def _written_value(slot: _TextSlot, text: str, guardrail_name: str | None) -> object:
    if slot.json_value is None:
        return text
    try:
        decoded: Final = _DECODED_JSON.validate_python(json.loads(text, strict=False))
    except json.JSONDecodeError:
        raise unappliable_request_rewrite(guardrail_name) from None
    value: Final = _with_original_numbers(slot.json_value, decoded, depth=0)
    if isinstance(slot.json_value, dict) and _json_object(value) is None:
        raise unappliable_request_rewrite(guardrail_name)
    return value


def _write_back_texts(
    text_slots: Sequence[_TextSlot], guardrailed_texts: Sequence[str] | None, guardrail_name: str | None
) -> None:
    if not guardrailed_texts:
        return
    writes: Final = tuple(
        (slot, _written_value(slot, text, guardrail_name))
        for slot, text in zip(text_slots, guardrailed_texts)
        if text != slot.text
    )
    if not all(_is_writable(slot.container, slot.key) for slot, _ in writes):
        raise unappliable_request_rewrite(guardrail_name)
    for slot, value in writes:
        _write_field(slot.container, slot.key, value)


def _parse_json_dict_or_none(payload: str) -> Mapping[str, object] | None:
    try:
        parsed: Final = _DECODED_JSON.validate_python(json.loads(payload))
    except json.JSONDecodeError:
        return None
    return as_str_mapping(parsed)


def _sse_payload_slots(sse_text: str) -> tuple[_TextSlot, ...]:
    payloads: Final = (line[len("data:") :].strip() for line in split_sse_lines(sse_text) if line.startswith("data:"))
    parsed: Final = (_parse_json_dict_or_none(payload) for payload in payloads if payload and payload != "[DONE]")
    return tuple(chain.from_iterable(_response_part_slots(chunk) for chunk in parsed if chunk is not None))


def _chunk_sse_text(chunk: object) -> str | None:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, str):
        return chunk
    return None


def _is_text_part_slot(slot: _TextSlot) -> bool:
    return slot.key == "text" or isinstance(slot.key, int)


def _stream_separator(previous: _TextSlot, current: _TextSlot) -> str:
    return "" if _is_text_part_slot(previous) and _is_text_part_slot(current) else "\n"


def _joined_stream_text(text_slots: Sequence[_TextSlot]) -> str:
    if not text_slots:
        return ""
    return text_slots[0].text + "".join(
        _stream_separator(previous, current) + current.text for previous, current in pairwise(text_slots)
    )


def _accumulated_stream_text(responses_so_far: Sequence[object]) -> str:
    object_slots: Final = tuple(
        chain.from_iterable(_response_part_slots(chunk) for chunk in responses_so_far if _chunk_sse_text(chunk) is None)
    )
    return _joined_stream_text((*object_slots, *_sse_payload_slots(_joined_sse_text(responses_so_far))))


def _joined_sse_text(responses_so_far: Sequence[object]) -> str:
    sse_texts: Final = (_chunk_sse_text(chunk) for chunk in responses_so_far)
    return "".join(sse for sse in sse_texts if sse is not None)


def _sse_payload_function_calls(sse_text: str) -> tuple[object, ...]:
    payloads: Final = (line[len("data:") :].strip() for line in split_sse_lines(sse_text) if line.startswith("data:"))
    parsed: Final = (_parse_json_dict_or_none(payload) for payload in payloads if payload and payload != "[DONE]")
    return tuple(chain.from_iterable(_function_calls_in_response(chunk) for chunk in parsed if chunk is not None))


def _accumulated_stream_function_calls(responses_so_far: Sequence[object]) -> tuple[object, ...]:
    object_calls: Final = tuple(
        chain.from_iterable(
            _function_calls_in_response(chunk) for chunk in responses_so_far if _chunk_sse_text(chunk) is None
        )
    )
    return object_calls + _sse_payload_function_calls(_joined_sse_text(responses_so_far))


def _guardrail_inputs(
    texts: Sequence[str], tool_calls: Sequence[ChatCompletionToolCallChunk], model: object
) -> GenericGuardrailAPIInputs:
    model_fields: Final[GenericGuardrailAPIInputs] = {"model": model} if isinstance(model, str) else {}
    tool_call_fields: Final[GenericGuardrailAPIInputs] = {"tool_calls": list(tool_calls)} if tool_calls else {}
    return {"texts": _texts_payload(texts), **model_fields, **tool_call_fields}


_SSE_FIELD_PREFIXES: Final = (":", "event:", "id:", "retry:")


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _object_sequence(value: object) -> Sequence[object] | None:
    return _OBJECT_SEQUENCE.validate_python(value) if _is_sequence(value) else None


def _is_response_shaped(item: object) -> bool:
    """What ``_field`` can read: a wire-format dict or an SDK response object."""
    return isinstance(item, dict) or hasattr(item, "candidates")


def _candidate_carries_function_call(candidate: object) -> bool:
    parts: Final = _field(_field(candidate, "content"), "parts")
    if parts is None:
        return False
    part_list: Final = _object_sequence(parts)
    if part_list is None:
        return True
    return any(
        _field(part, "functionCall") is not None or _field(part, "function_call") is not None for part in part_list
    )


def _response_carries_function_call(response: object) -> bool:
    """Whether a generateContent response chunk carries a ``functionCall`` part, the tool call its client runs.
    A chunk shaped another way than candidates of content parts counts as carrying one."""
    candidates: Final = _field(response, "candidates")
    if candidates is None:
        return False
    candidate_list: Final = _object_sequence(candidates)
    if candidate_list is None:
        return True
    return any(_candidate_carries_function_call(candidate) for candidate in candidate_list)


def _sse_carries_function_call(sse_text: str) -> bool:
    """The same for raw SSE frames. A line that is not an SSE field, or a data payload that is not a JSON
    object, cannot be read for a function call, so it counts as carrying one."""
    return any(_sse_line_carries_function_call(line) for line in split_sse_lines(sse_text) if line.strip())


def _sse_line_carries_function_call(line: str) -> bool:
    if not line.startswith("data:"):
        return not line.startswith(_SSE_FIELD_PREFIXES)
    payload: Final = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return False
    parsed: Final = _parse_json_dict_or_none(payload)
    return parsed is None or _response_carries_function_call(parsed)


@dataclass(frozen=True, slots=True)
class PartLocation:
    line: int
    candidate: int
    part: int


@dataclass(frozen=True, slots=True)
class StreamedFunctionCall:
    location: PartLocation
    candidate_index: int
    name: str
    args: Mapping[str, object]
    part: Mapping[str, object]


_FUNCTION_CALL_FIELDS: Final = ("functionCall", "function_call")
_FUNCTION_CALL_KEYS: Final = frozenset({"id", "name", "args"})
_THOUGHT_SIGNATURE_FIELDS: Final = ("thoughtSignature", "thought_signature")
_NO_ARGS: Final[Mapping[str, object]] = MappingProxyType({})
_StreamedCalls: TypeAlias = tuple[StreamedFunctionCall, ...]


def _all_calls(results: Iterable[_StreamedCalls | None]) -> _StreamedCalls | None:
    collected: Final = tuple(results)
    if any(result is None for result in collected):
        return None
    return tuple(chain.from_iterable(result for result in collected if result is not None))


def _function_call(
    location: PartLocation, candidate_index: int, part: Mapping[str, object], value: object
) -> StreamedFunctionCall | None:
    fields: Final = as_str_mapping(value)
    if fields is None or not fields.keys() <= _FUNCTION_CALL_KEYS:
        return None
    name: Final = fields.get("name")
    raw_args: Final = fields.get("args")
    args: Final = _NO_ARGS if raw_args is None else as_str_mapping(raw_args)
    if not isinstance(name, str) or args is None:
        return None
    return StreamedFunctionCall(location=location, candidate_index=candidate_index, name=name, args=args, part=part)


def _part_function_calls(location: PartLocation, candidate_index: int, part: object) -> _StreamedCalls | None:
    fields: Final = as_str_mapping(part)
    if fields is None:
        return None
    present: Final = tuple(fields[key] for key in _FUNCTION_CALL_FIELDS if fields.get(key) is not None)
    match present:
        case ():
            return ()
        case (function_call,):
            call: Final = _function_call(location, candidate_index, fields, function_call)
            return None if call is None else (call,)
        case _:
            return None


def _candidate_parts(candidate: Mapping[str, object]) -> Sequence[object] | None:
    content: Final = candidate.get("content")
    if content is None:
        return ()
    content_fields: Final = as_str_mapping(content)
    if content_fields is None:
        return None
    parts: Final = content_fields.get("parts")
    return () if parts is None else _object_sequence(parts)


def _candidate_index(candidate: Mapping[str, object], position: int) -> int | None:
    index: Final = candidate.get("index")
    if index is None:
        return position
    return index if isinstance(index, int) and not isinstance(index, bool) else None


def _candidate_function_calls(line: int, position: int, candidate: object) -> _StreamedCalls | None:
    fields: Final = as_str_mapping(candidate)
    parts: Final = None if fields is None else _candidate_parts(fields)
    candidate_index: Final = None if fields is None else _candidate_index(fields, position)
    if parts is None or candidate_index is None:
        return None
    return _all_calls(
        _part_function_calls(PartLocation(line, position, index), candidate_index, part)
        for index, part in enumerate(parts)
    )


def _response_function_calls(line: int, response: Mapping[str, object]) -> _StreamedCalls | None:
    candidates: Final = response.get("candidates")
    if candidates is None:
        return ()
    candidate_list: Final = _object_sequence(candidates)
    if candidate_list is None:
        return None
    return _all_calls(
        _candidate_function_calls(line, position, candidate) for position, candidate in enumerate(candidate_list)
    )


def _without_line_break(line: str) -> str:
    return line.removesuffix("\n").removesuffix("\r")


def _data_payload(line: str) -> str | None:
    payload: Final = line[len("data:") :].strip()
    return None if not payload or payload == "[DONE]" else payload


def _starts_a_line_for_wider_readers(line: str) -> bool:
    """Whether ``str.splitlines``, which also breaks on U+2028 and the like, reads a ``data`` line inside it."""
    return any(piece.lstrip().startswith("data") for piece in line.splitlines()[1:])


def _sse_line_function_calls(index: int, line: str) -> _StreamedCalls | None:
    if _starts_a_line_for_wider_readers(line):
        return None
    if not line.strip() or line.startswith(_SSE_FIELD_PREFIXES):
        return ()
    if not line.startswith("data:"):
        return None
    payload: Final = _data_payload(line)
    if payload is None:
        return ()
    parsed: Final = _parse_json_dict_or_none(payload)
    return None if parsed is None else _response_function_calls(index, parsed)


def generate_content_sse_function_calls(sse_text: str) -> _StreamedCalls | None:
    """Every ``functionCall`` part of a raw ``:streamGenerateContent`` SSE stream, in stream order.

    None when a line cannot be read the way any client could read it, so a caller can refuse a stream
    that may carry a call it did not see.
    """
    lines: Final = tuple(map(_without_line_break, split_sse_lines_keepends(sse_text)))
    return _all_calls(_sse_line_function_calls(index, line) for index, line in enumerate(lines))


def _thought_signature(part: Mapping[str, object]) -> Mapping[str, object]:
    return {key: part[key] for key in _THOUGHT_SIGNATURE_FIELDS if key in part}


def _calls_by_candidate(calls: _StreamedCalls) -> tuple[_StreamedCalls, ...]:
    indexes: Final = frozenset(call.candidate_index for call in calls)
    return tuple(tuple(call for call in calls if call.candidate_index == index) for index in indexes)


def _signature_handoff(
    candidate_calls: _StreamedCalls, texts: Mapping[PartLocation, str]
) -> tuple[StreamedFunctionCall, StreamedFunctionCall] | None:
    first: Final = candidate_calls[0]
    kept: Final = next((call for call in candidate_calls if call.location not in texts), None)
    if kept is None or _thought_signature(kept.part):
        return None
    if first.location not in texts or not _thought_signature(first.part):
        return None
    return first, kept


def _replacement_parts(
    calls: _StreamedCalls, texts: Mapping[PartLocation, str]
) -> Mapping[PartLocation, Mapping[str, object]]:
    handoffs: Final = tuple(filter(None, (_signature_handoff(group, texts) for group in _calls_by_candidate(calls))))
    moved: Final = frozenset(source.location for source, _ in handoffs)
    as_text: Final = {
        call.location: (
            {"text": texts[call.location]}
            if call.location in moved
            else {"text": texts[call.location], **_thought_signature(call.part)}
        )
        for call in calls
        if call.location in texts
    }
    signed: Final = {target.location: {**target.part, **_thought_signature(source.part)} for source, target in handoffs}
    return MappingProxyType({**as_text, **signed})


def _candidate_with_parts(
    line: int, position: int, candidate: object, replacements: Mapping[PartLocation, Mapping[str, object]]
) -> object:
    fields: Final = as_str_mapping(candidate)
    parts: Final = None if fields is None else _candidate_parts(fields)
    content: Final = None if fields is None else as_str_mapping(fields.get("content"))
    if fields is None or content is None or not parts:
        return candidate
    return {
        **fields,
        "content": {
            **content,
            "parts": [replacements.get(PartLocation(line, position, index), part) for index, part in enumerate(parts)],
        },
    }


def _line_with_parts(index: int, line: str, replacements: Mapping[PartLocation, Mapping[str, object]]) -> str:
    if not any(location.line == index for location in replacements):
        return line
    body: Final = _without_line_break(line)
    payload: Final = _data_payload(body) if body.startswith("data:") else None
    response: Final = None if payload is None else _parse_json_dict_or_none(payload)
    candidates: Final = None if response is None else _object_sequence(response.get("candidates"))
    if response is None or candidates is None:
        return line
    rewritten: Final = {
        **response,
        "candidates": [
            _candidate_with_parts(index, position, candidate, replacements)
            for position, candidate in enumerate(candidates)
        ],
    }
    return f"data: {json.dumps(rewritten)}{line.removeprefix(body)}"


def generate_content_sse_with_parts_replaced(
    sse_text: str, calls: _StreamedCalls, texts: Mapping[PartLocation, str]
) -> str:
    """The stream with each of ``calls`` at a location in ``texts`` replaced by that text part, every other byte kept.

    A replaced call's thought signature stays on its text part, except that a turn's replaced first call hands it
    to the first call kept after it when that call is unsigned, since Gemini 3 signs only a turn's first call and
    answers 400 to a next turn whose first call is unsigned.
    """
    replacements: Final = _replacement_parts(calls, texts)
    return "".join(
        _line_with_parts(index, line, replacements) for index, line in enumerate(split_sse_lines_keepends(sse_text))
    )


class GoogleGenAIGenerateContentHandler(BaseTranslation):
    """
    Guardrail translation for the google genai generateContent surface
    (/models/{model}:generateContent, :streamGenerateContent, and the
    litellm SDK generate_content call types).
    """

    async def process_input_messages(
        self,
        data: dict[str, object],  # mutable-ok: base handler contract passes the proxy's request dict to apply_guardrail
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> object:
        request: Final[Mapping[str, object]] = as_str_mapping(data) or _EMPTY_REQUEST_DATA
        scope: Final = _request_part_scope(guardrail_to_apply)
        text_slots: Final = _request_text_slots(request, scope)
        call_groups: Final = _function_call_groups(_request_function_calls(request)) if scope.tool_calls else ()
        if not text_slots and not call_groups:
            verbose_proxy_logger.debug("Google GenAI guardrail: no request text found, skipping")
            return data
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(_slot_texts(text_slots), _tool_calls_payload(call_groups), request.get("model")),
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        returned_texts: Final = guardrailed_inputs.get("texts")
        returned_tool_calls: Final = _returned_tool_calls(call_groups, guardrailed_inputs)
        if _text_count_differs(text_slots, returned_texts):
            raise unappliable_request_rewrite(guardrail_to_apply.guardrail_name)
        _write_back_texts(text_slots, returned_texts, guardrail_to_apply.guardrail_name)
        _write_back_function_calls(call_groups, returned_tool_calls, guardrail_to_apply.guardrail_name)
        return data

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: Mapping[str, object] | None = None,
    ) -> object:
        text_slots: Final = _response_part_slots(response)
        call_groups: Final = _function_call_groups(_function_calls_in_response(response))
        if not text_slots and not call_groups:
            verbose_proxy_logger.debug("Google GenAI guardrail: no response text found, skipping")
            return response
        guardrail_request_data: Final = self._merged_request_data(
            request_data=request_data,
            user_api_key_dict=user_api_key_dict,
            context_key="response",
            context_value=response,
        )
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(
                _slot_texts(text_slots), _tool_calls_payload(call_groups), guardrail_request_data.get("model")
            ),
            request_data=guardrail_request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_slots, guardrailed_inputs.get("texts"), guardrail_to_apply.guardrail_name)
        _write_back_function_calls(
            call_groups, _returned_tool_calls(call_groups, guardrailed_inputs), guardrail_to_apply.guardrail_name
        )
        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: Sequence[object],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: Mapping[str, object] | None = None,
        stream_transform_sink: StreamTransformSink | None = None,
    ) -> object:
        accumulated_text: Final = _accumulated_stream_text(responses_so_far)
        call_groups: Final = _function_call_groups(_accumulated_stream_function_calls(responses_so_far))
        if not accumulated_text and not call_groups:
            return responses_so_far
        guardrail_request_data: Final = self._merged_request_data(
            request_data=request_data,
            user_api_key_dict=user_api_key_dict,
            context_key="responses_so_far",
            context_value=responses_so_far,
        )
        _guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(
                (accumulated_text,) if accumulated_text else (), _tool_calls_payload(call_groups), None
            ),
            request_data=guardrail_request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        return responses_so_far

    def starts_client_tool_call(self, item: object) -> bool:
        """A ``functionCall`` part is the tool call a generateContent client runs, and every stream item that
        carries one, or a piece of one, has that part: the proxy hands this translation one SSE event per item.
        An item this cannot read for a function call counts as carrying one, so it waits for the verdict."""
        sse_text: Final = _chunk_sse_text(item)
        if sse_text is not None:
            return _sse_carries_function_call(sse_text)
        return not _is_response_shaped(item) or _response_carries_function_call(item)

    def _merged_request_data(
        self,
        request_data: Mapping[str, object] | None,
        user_api_key_dict: Optional["UserAPIKeyAuth"],
        context_key: str,
        context_value: object,
    ) -> dict[str, object]:  # mutable-ok: CustomGuardrail.apply_guardrail requires a plain dict request payload
        base: Final = request_data if request_data is not None else _EMPTY_REQUEST_DATA
        user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
        context_pairs: Final = ((context_key, context_value),) if context_key not in base else ()
        metadata_pairs: Final = (
            (("litellm_metadata", user_metadata),) if user_metadata and "litellm_metadata" not in base else ()
        )
        return dict((*base.items(), *context_pairs, *metadata_pairs))
