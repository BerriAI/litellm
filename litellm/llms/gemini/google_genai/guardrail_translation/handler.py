"""
Google GenAI generateContent handler for Unified Guardrails.

Extracts text from generateContent requests (systemInstruction.parts[].text
and contents[].parts[].text) and responses (candidates[].content.parts[].text),
applies the guardrail, and
writes the guardrailed text back in place. A guardrail that opts into
scan_tool_calls_as_text also gets every functionCall part as a tool call, with
the values a streamed call sends in partialArgs joined per JSON path, and
masked request arguments are written back into functionCall.args. Requests and
responses may be dicts (wire format) or google-genai SDK objects; streaming
chunks may additionally be raw SSE frames, which are scanned for detection (a
blocking guardrail raises) without rewriting the frames.
"""

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_tool_calls_as_text_for_guardrail,
    stream_item_field,
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
_PARTIAL_ARG_VALUE_FIELDS: Final = (
    ("stringValue", "string_value"),
    ("numberValue", "number_value"),
    ("boolValue", "bool_value"),
    ("nullValue", "null_value"),
)
_PARTIAL_ARG_VALUE_NAMES: Final = tuple(chain.from_iterable(_PARTIAL_ARG_VALUE_FIELDS))


def _as_dict(value: object) -> Mapping[object, object] | None:
    return value if isinstance(value, dict) else None


def _as_list_or_tuple(value: object) -> Sequence[object] | None:
    return value if isinstance(value, (list, tuple)) else None


def _field(container: object, name: str) -> object | None:
    mapping: Final = _as_dict(container)
    if mapping is not None:
        return mapping.get(name)
    return getattr(container, name, None)


def _part_text(part: object) -> str | None:
    text: Final = _field(part, "text")
    if isinstance(text, str) and text:
        return text
    return None


def _write_part_text(part: object, text: str) -> None:
    if isinstance(part, dict):
        part["text"] = text  # rebind-ok: guardrail write-back rewrites the caller's part in place by handler contract
        return
    setattr(part, "text", text)  # noqa: B010  # SDK parts are typed as object here; direct assignment cannot type-check


def _content_text_parts(content: object) -> tuple[object, ...]:
    parts: Final = _as_list_or_tuple(_field(content, "parts"))
    if parts is None:
        return ()
    return tuple(part for part in parts if _part_text(part) is not None)


def _system_instruction(data: Mapping[str, object]) -> object | None:
    return next(
        (
            value
            for container in (data, data.get("config"))
            if container is not None
            for key in ("systemInstruction", "system_instruction")
            for value in (_field(container, key),)
            if value is not None
        ),
        None,
    )


def _request_text_parts(data: Mapping[str, object]) -> tuple[object, ...]:
    contents: Final = data.get("contents")
    listed: Final = as_object_list(contents)
    content_list: Final = (contents,) if _as_dict(contents) is not None else tuple(listed) if listed is not None else ()
    return (
        *_content_text_parts(_system_instruction(data)),
        *(part for content in content_list for part in _content_text_parts(content)),
    )


def _response_text_parts(response: object) -> tuple[object, ...]:
    candidates: Final = _as_list_or_tuple(_field(response, "candidates"))
    if candidates is None:
        return ()
    return tuple(part for candidate in candidates for part in _content_text_parts(_field(candidate, "content")))


def _camel_or_snake(container: object, camel_case: str, snake_case: str) -> object | None:
    camel_value: Final = _field(container, camel_case)
    return camel_value if camel_value is not None else _field(container, snake_case)


def _function_call_in_part(part: object) -> object | None:
    return _camel_or_snake(part, "functionCall", "function_call")


def _content_function_calls(content: object) -> tuple[object, ...]:
    calls: Final = (_function_call_in_part(part) for part in as_object_list(_field(content, "parts")) or ())
    return tuple(call for call in calls if call is not None)


def _request_function_calls(data: Mapping[str, object]) -> tuple[object, ...]:
    contents: Final = data.get("contents")
    single_content: Final = as_str_mapping(contents)
    content_list: Final = (single_content,) if single_content is not None else as_object_list(contents) or ()
    return tuple(chain.from_iterable(_content_function_calls(content) for content in content_list))


def _function_calls_in_response(response: object) -> tuple[object, ...]:
    candidates: Final = as_object_list(_field(response, "candidates")) or ()
    return tuple(chain.from_iterable(_content_function_calls(_field(candidate, "content")) for candidate in candidates))


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


def _partial_texts_at(partial_args: Sequence[object], path: str) -> Iterator[str]:
    return (_partial_arg_text(partial_arg) for partial_arg in partial_args if _partial_arg_path(partial_arg) == path)


def _partial_values_by_path(group: Sequence[object]) -> Mapping[str, str]:
    partial_args: Final = tuple(chain.from_iterable(_partial_args(function_call) for function_call in group))
    paths: Final = tuple(dict.fromkeys(_partial_arg_path(partial_arg) for partial_arg in partial_args))
    return {path: "".join(_partial_texts_at(partial_args, path)) for path in paths}


def _group_field(group: Sequence[object], name: str) -> str | None:
    values: Final = (_field(function_call, name) for function_call in group)
    return next((value for value in values if isinstance(value, str) and value), None)


def _group_arguments(group: Sequence[object]) -> Mapping[str, object]:
    args: Final = (as_str_mapping(_field(function_call, "args")) or {} for function_call in group)
    return {**dict(chain.from_iterable(mapping.items() for mapping in args)), **_partial_values_by_path(group)}


def _tool_call(group: Sequence[object], index: int) -> ChatCompletionToolCallChunk:
    return ChatCompletionToolCallChunk(
        id=_group_field(group, "id"),
        type="function",
        function={
            "name": _group_field(group, "name"),
            "arguments": json.dumps(_group_arguments(group), ensure_ascii=False, default=str),
        },
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
    partial_args: Final = tuple(chain.from_iterable(_partial_args(function_call) for function_call in group))
    sent_partial_values: Final = _partial_values_by_path(group)
    head_args: Final = {key: value for key, value in args.items() if key not in sent_partial_values}
    if head_args != (as_str_mapping(_field(group[0], "args")) or {}):
        _set_field(group[0], "args", head_args)
    for path, sent_text in sent_partial_values.items():
        _write_partial_path(
            tuple(partial_arg for partial_arg in partial_args if _partial_arg_path(partial_arg) == path),
            args.get(path),
            sent_text,
            guardrail_name,
        )


def _guardrailed_args(returned_tool_call: object, guardrail_name: str | None) -> Mapping[str, object]:
    arguments: Final = stream_item_field(stream_item_field(returned_tool_call, "function"), "arguments")
    parsed: Final = _parse_json_dict_or_none(arguments) if isinstance(arguments, str) else None
    if parsed is None:
        raise unappliable_request_rewrite(guardrail_name)
    return parsed


def _write_back_function_calls(
    groups: Sequence[Sequence[object]],
    sent_tool_calls: Sequence[ChatCompletionToolCallChunk],
    returned_tool_calls: Sequence[object] | None,
    guardrail_name: str | None,
) -> None:
    if returned_tool_calls is None:
        return
    if len(returned_tool_calls) != len(sent_tool_calls):
        raise unappliable_request_rewrite(guardrail_name)
    changed: Final = tuple(
        (group, returned)
        for group, sent, returned in zip(groups, sent_tool_calls, returned_tool_calls, strict=True)
        if stream_item_field(stream_item_field(returned, "function"), "arguments") != sent["function"].get("arguments")
    )
    for group, returned in changed:
        _write_group_args(group, _guardrailed_args(returned, guardrail_name), guardrail_name)


def _part_texts(text_parts: Sequence[object]) -> tuple[str, ...]:
    return tuple(text for part in text_parts for text in (_part_text(part),) if text is not None)


def _texts_payload(
    texts: Sequence[str],
) -> list[str]:  # mutable-ok: GenericGuardrailAPIInputs.texts is declared list[str]
    return list(texts)


def _write_back_texts(text_parts: Sequence[object], guardrailed_texts: Sequence[str] | None) -> None:
    if not guardrailed_texts or len(guardrailed_texts) != len(text_parts):
        return
    for part, text in zip(text_parts, guardrailed_texts):
        _write_part_text(part, text)


def _parse_json_dict_or_none(payload: str) -> Mapping[str, object] | None:
    try:
        parsed: Final = _DECODED_JSON.validate_python(json.loads(payload))
    except json.JSONDecodeError:
        return None
    return as_str_mapping(parsed)


def _sse_payload_texts(sse_text: str) -> tuple[str, ...]:
    return tuple(
        text
        for line in split_sse_lines(sse_text)
        if line.startswith("data:")
        for payload in (line[len("data:") :].strip(),)
        if payload and payload != "[DONE]"
        for parsed in (_parse_json_dict_or_none(payload),)
        if parsed is not None
        for text in _part_texts(_response_text_parts(parsed))
    )


def _chunk_sse_text(chunk: object) -> str | None:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, str):
        return chunk
    return None


def _accumulated_stream_text(responses_so_far: Sequence[object]) -> str:
    object_texts: Final = tuple(
        text
        for chunk in responses_so_far
        if _chunk_sse_text(chunk) is None
        for text in _part_texts(_response_text_parts(chunk))
    )
    sse_text: Final = "".join(sse for chunk in responses_so_far for sse in (_chunk_sse_text(chunk),) if sse is not None)
    return "".join(object_texts) + "".join(_sse_payload_texts(sse_text))


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
        text_parts: Final = _request_text_parts(data)
        call_groups: Final = (
            _function_call_groups(_request_function_calls(data))
            if effective_scan_tool_calls_as_text_for_guardrail(guardrail_to_apply)
            else ()
        )
        if not text_parts and not call_groups:
            verbose_proxy_logger.debug("Google GenAI guardrail: no request text found, skipping")
            return data
        tool_calls: Final = _tool_calls_payload(call_groups)
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(_part_texts(text_parts), tool_calls, data.get("model")),
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_parts, guardrailed_inputs.get("texts"))
        _write_back_function_calls(
            call_groups,
            tool_calls,
            as_object_list(stream_item_field(guardrailed_inputs, "tool_calls")) if tool_calls else None,
            guardrail_to_apply.guardrail_name,
        )
        return data

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: Mapping[str, object] | None = None,
    ) -> object:
        text_parts: Final = _response_text_parts(response)
        call_groups: Final = (
            _function_call_groups(_function_calls_in_response(response))
            if effective_scan_tool_calls_as_text_for_guardrail(guardrail_to_apply)
            else ()
        )
        if not text_parts and not call_groups:
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
                _part_texts(text_parts), _tool_calls_payload(call_groups), guardrail_request_data.get("model")
            ),
            request_data=guardrail_request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_parts, guardrailed_inputs.get("texts"))
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
        call_groups: Final = (
            _function_call_groups(_accumulated_stream_function_calls(responses_so_far))
            if effective_scan_tool_calls_as_text_for_guardrail(guardrail_to_apply)
            else ()
        )
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
