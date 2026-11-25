"""
Google GenAI generateContent handler for Unified Guardrails.

Request text comes from every system instruction the request carries
(systemInstruction and system_instruction, top level and inside config) and from
contents[].parts[]: text parts, plain-string parts, executableCode code, functionCall
and toolCall args, and the tool results (codeExecutionResult output, functionResponse
and toolResponse response), which are left out when the guardrail skips tool messages.
Structured values are scanned as one JSON text, the way chat scans a tool message,
with control characters left raw so patterns still see line starts and numbers
quoted so a mask can replace them. A changed text is decoded back into the part,
numbers the guardrail left alone keep their type, and a text that no longer
decodes, or turns an object into something else, is blocked with a 400.
Responses are scanned the same way across candidates[].content.parts[], tool
results included, and streaming chunks only for their text. Guardrailed text is
written back in place, so the provider and the client get their wire shape. Requests and
responses may be dicts (wire format) or google-genai SDK objects; streaming
chunks may additionally be raw SSE frames, which are scanned for detection (a
blocking guardrail raises) without rewriting the frames.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_tool_message_for_guardrail,
    unappliable_request_rewrite,
)
from litellm.llms.custom_httpx.sse_lines import split_sse_lines, split_sse_lines_keepends
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


@dataclass(frozen=True, slots=True)
class _PartField:
    keys: tuple[str, str]
    name: str
    is_json: bool


_MODEL_PART_FIELDS: Final = (
    _PartField(keys=("executableCode", "executable_code"), name="code", is_json=False),
    _PartField(keys=("functionCall", "function_call"), name="args", is_json=True),
    _PartField(keys=("toolCall", "tool_call"), name="args", is_json=True),
)
_TOOL_RESULT_PART_FIELDS: Final = (
    _PartField(keys=("codeExecutionResult", "code_execution_result"), name="output", is_json=False),
    _PartField(keys=("functionResponse", "function_response"), name="response", is_json=True),
    _PartField(keys=("toolResponse", "tool_response"), name="response", is_json=True),
)


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


def _original_number_or(scalar: str, original: str) -> str:
    return original if not original.startswith('"') and scalar == f'"{original}"' else scalar


def _with_original_numbers(original_value: object, guardrailed_text: str) -> str:
    original_scalars: Final = tuple(scalar.group(0) for scalar in _JSON_SCALAR.finditer(_json_text(original_value)))
    if len(original_scalars) != sum(1 for _ in _JSON_SCALAR.finditer(guardrailed_text)):
        return guardrailed_text
    originals: Final = iter(original_scalars)
    return _JSON_SCALAR.sub(lambda scalar: _original_number_or(scalar.group(0), next(originals)), guardrailed_text)


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


def _request_part_slots(parts: Sequence[object], index: int, include_tool_results: bool) -> tuple[_TextSlot, ...]:
    part_fields: Final = (
        (*_MODEL_PART_FIELDS, *_TOOL_RESULT_PART_FIELDS) if include_tool_results else _MODEL_PART_FIELDS
    )
    field_slots: Final = chain.from_iterable(_part_field_slots(parts[index], part_field) for part_field in part_fields)
    return (*_part_text_slots(parts, index), *field_slots)


def _content_part_slots(content: object, include_tool_results: bool) -> tuple[_TextSlot, ...]:
    parts: Final = _repeated_field(_field(content, "parts"))
    return tuple(
        chain.from_iterable(_request_part_slots(parts, index, include_tool_results) for index in range(len(parts)))
    )


def _request_contents(data: Mapping[str, object]) -> tuple[object, ...]:
    system_instructions: Final = chain.from_iterable(
        _present_fields(container, _SYSTEM_INSTRUCTION_KEYS) for container in (data, data.get("config"))
    )
    return (*system_instructions, *_repeated_field(data.get("contents")))


def _request_text_slots(data: Mapping[str, object], include_tool_results: bool) -> tuple[_TextSlot, ...]:
    return tuple(
        chain.from_iterable(_content_part_slots(content, include_tool_results) for content in _request_contents(data))
    )


def _candidate_contents(response: object) -> tuple[object, ...]:
    candidates: Final = _as_list_or_tuple(_field(response, "candidates"))
    if candidates is None:
        return ()
    return tuple(_field(candidate, "content") for candidate in candidates)


def _response_part_slots(response: object) -> tuple[_TextSlot, ...]:
    return tuple(
        chain.from_iterable(
            _content_part_slots(content, include_tool_results=True) for content in _candidate_contents(response)
        )
    )


def _slot_texts(text_slots: Sequence[_TextSlot]) -> tuple[str, ...]:
    return tuple(slot.text for slot in text_slots)


def _texts_payload(
    texts: Sequence[str],
) -> list[str]:  # mutable-ok: GenericGuardrailAPIInputs.texts is declared list[str]
    return list(texts)


def _guardrail_inputs(texts: Sequence[str], model: object) -> GenericGuardrailAPIInputs:
    with_texts: Final[GenericGuardrailAPIInputs] = {"texts": _texts_payload(texts)}
    return {**with_texts, "model": model} if isinstance(model, str) else with_texts


def _written_value(slot: _TextSlot, text: str, guardrail_name: str | None) -> object:
    if slot.json_value is None:
        return text
    try:
        value: Final = _DECODED_JSON.validate_python(
            json.loads(_with_original_numbers(slot.json_value, text), strict=False)
        )
    except json.JSONDecodeError:
        raise unappliable_request_rewrite(guardrail_name) from None
    if isinstance(slot.json_value, dict) and _json_object(value) is None:
        raise unappliable_request_rewrite(guardrail_name)
    return value


def _write_back_texts(
    text_slots: Sequence[_TextSlot], guardrailed_texts: Sequence[str] | None, guardrail_name: str | None
) -> None:
    if not guardrailed_texts or len(guardrailed_texts) != len(text_slots):
        return
    writes: Final = tuple(
        (slot, _written_value(slot, text, guardrail_name))
        for slot, text in zip(text_slots, guardrailed_texts)
        if text != slot.text
    )
    for slot, value in writes:
        _write_field(slot.container, slot.key, value)


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
        for text in _slot_texts(_response_part_slots(parsed))
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
        for text in _slot_texts(_response_part_slots(chunk))
    )
    sse_text: Final = "".join(sse for chunk in responses_so_far for sse in (_chunk_sse_text(chunk),) if sse is not None)
    return "".join(object_texts) + "".join(_sse_payload_texts(sse_text))


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
        data: dict,  # mutable-ok: base handler contract passes the proxy's request dict through to apply_guardrail
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> object:
        request: Final[Mapping[str, object]] = as_str_mapping(data) or _EMPTY_REQUEST_DATA
        text_slots: Final = _request_text_slots(
            request, include_tool_results=not effective_skip_tool_message_for_guardrail(guardrail_to_apply)
        )
        if not text_slots:
            verbose_proxy_logger.debug("Google GenAI guardrail: no request text found, skipping")
            return data
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(_slot_texts(text_slots), request.get("model")),
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_slots, guardrailed_inputs.get("texts"), guardrail_to_apply.guardrail_name)
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
        if not text_slots:
            verbose_proxy_logger.debug("Google GenAI guardrail: no response text found, skipping")
            return response
        guardrail_request_data: Final = self._merged_request_data(
            request_data=request_data,
            user_api_key_dict=user_api_key_dict,
            context_key="response",
            context_value=response,
        )
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(_slot_texts(text_slots), guardrail_request_data.get("model")),
            request_data=guardrail_request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_slots, guardrailed_inputs.get("texts"), guardrail_to_apply.guardrail_name)
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
        if not accumulated_text:
            return responses_so_far
        guardrail_request_data: Final = self._merged_request_data(
            request_data=request_data,
            user_api_key_dict=user_api_key_dict,
            context_key="responses_so_far",
            context_value=responses_so_far,
        )
        _guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=_texts_payload((accumulated_text,))),
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
