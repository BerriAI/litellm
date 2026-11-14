"""
Google GenAI generateContent handler for Unified Guardrails.

Request text comes from every system instruction the request carries
(systemInstruction and system_instruction, top level and inside config) and from
contents[].parts[]: text parts, plain-string parts, executableCode code, toolCall
args, and the tool results (codeExecutionResult output, functionResponse and
toolResponse response), which are left out when the guardrail skips tool messages.
Structured values are scanned as one JSON text, the way chat scans a tool message,
and a changed text is decoded back into the part, or blocked with a 400 when it no
longer decodes. Request functionCall parts reach the guardrail as chat-shaped
tool_calls under the call's own id (or a per-position one), and arguments the
guardrail changed are written back into args when it returns every call.
Response text comes from candidates[].content.parts[].text. Guardrailed text is
written back in place, so the provider gets the caller's wire shape. Requests and
responses may be dicts (wire format) or google-genai SDK objects; streaming
chunks may additionally be raw SSE frames, which are scanned for detection (a
blocking guardrail raises) without rewriting the frames.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import effective_skip_tool_message_for_guardrail
from litellm.llms.custom_httpx.sse_lines import split_sse_lines
from litellm.types.llms.openai import ChatCompletionToolCallChunk, ChatCompletionToolCallFunctionChunk
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from fastapi import HTTPException

    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_EMPTY_REQUEST_DATA: Final[Mapping[str, object]] = MappingProxyType({})
_DECODED_JSON: Final = TypeAdapter(object)
_SYSTEM_INSTRUCTION_KEYS: Final = ("systemInstruction", "system_instruction")
_FUNCTION_CALL_KEYS: Final = ("functionCall", "function_call")


@dataclass(frozen=True, slots=True)
class _PartField:
    keys: tuple[str, str]
    name: str
    is_json: bool


_MODEL_PART_FIELDS: Final = (
    _PartField(keys=("executableCode", "executable_code"), name="code", is_json=False),
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
    is_json: bool = False


@dataclass(frozen=True, slots=True)
class _FunctionCallSlot:
    call: object
    call_id: str | None
    name: str
    arguments: str


def _json_object(value: object) -> Mapping[str, object] | None:
    fields: Final = as_str_mapping(value)
    return fields if isinstance(fields, dict) else None


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
    match value:
        case dict():
            return (value,)
        case list() | tuple():
            return value
        case _:
            return ()


def _string_slot(container: object, key: str | int, value: object) -> tuple[_TextSlot, ...]:
    return (_TextSlot(container=container, key=key, text=value),) if isinstance(value, str) and value else ()


def _json_slot(container: object, key: str | int, value: object) -> tuple[_TextSlot, ...]:
    if not value:
        return ()
    text: Final = json.dumps(value, ensure_ascii=False, default=str)
    return (_TextSlot(container=container, key=key, text=text, is_json=True),)


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


def _content_request_slots(content: object, include_tool_results: bool) -> tuple[_TextSlot, ...]:
    parts: Final = _repeated_field(_field(content, "parts"))
    return tuple(
        chain.from_iterable(_request_part_slots(parts, index, include_tool_results) for index in range(len(parts)))
    )


def _content_text_slots(content: object) -> tuple[_TextSlot, ...]:
    parts: Final = _repeated_field(_field(content, "parts"))
    return tuple(chain.from_iterable(_part_text_slots(parts, index) for index in range(len(parts))))


def _request_contents(data: Mapping[str, object]) -> tuple[object, ...]:
    system_instructions: Final = chain.from_iterable(
        _present_fields(container, _SYSTEM_INSTRUCTION_KEYS) for container in (data, data.get("config"))
    )
    return (*system_instructions, *_repeated_field(data.get("contents")))


def _request_text_slots(data: Mapping[str, object], include_tool_results: bool) -> tuple[_TextSlot, ...]:
    return tuple(
        chain.from_iterable(
            _content_request_slots(content, include_tool_results) for content in _request_contents(data)
        )
    )


def _function_call_slot(call: object) -> _FunctionCallSlot:
    call_id: Final = _field(call, "id")
    name: Final = _field(call, "name")
    args: Final = _field(call, "args")
    return _FunctionCallSlot(
        call=call,
        call_id=call_id if isinstance(call_id, str) and call_id else None,
        name=name if isinstance(name, str) else "",
        arguments=json.dumps({} if args is None else args),
    )


def _content_function_calls(content: object) -> Iterable[object]:
    return chain.from_iterable(
        _present_fields(part, _FUNCTION_CALL_KEYS) for part in _repeated_field(_field(content, "parts"))
    )


def _request_function_calls(data: Mapping[str, object]) -> tuple[_FunctionCallSlot, ...]:
    calls: Final = chain.from_iterable(_content_function_calls(content) for content in _request_contents(data))
    return tuple(_function_call_slot(call) for call in calls)


def _response_text_slots(response: object) -> tuple[_TextSlot, ...]:
    candidates: Final = _field(response, "candidates")
    if not isinstance(candidates, (list, tuple)):
        return ()
    return tuple(chain.from_iterable(_content_text_slots(_field(candidate, "content")) for candidate in candidates))


def _slot_texts(text_slots: Sequence[_TextSlot]) -> tuple[str, ...]:
    return tuple(slot.text for slot in text_slots)


def _texts_payload(
    texts: Sequence[str],
) -> list[str]:  # mutable-ok: GenericGuardrailAPIInputs.texts is declared list[str]
    return list(texts)


def _tool_call_chunk(index: int, slot: _FunctionCallSlot) -> ChatCompletionToolCallChunk:
    return ChatCompletionToolCallChunk(
        id=slot.call_id or f"call_{index}_{slot.name or 'unknown'}",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(name=slot.name, arguments=slot.arguments),
        index=index,
    )


def _guardrail_inputs(
    texts: Sequence[str],
    model: object,
    function_calls: Sequence[_FunctionCallSlot] = (),
) -> GenericGuardrailAPIInputs:
    with_texts: Final[GenericGuardrailAPIInputs] = {"texts": _texts_payload(texts)}
    with_model: Final[GenericGuardrailAPIInputs] = (
        {**with_texts, "model": model} if isinstance(model, str) else with_texts
    )
    if not function_calls:
        return with_model
    return {**with_model, "tool_calls": [_tool_call_chunk(index, slot) for index, slot in enumerate(function_calls)]}


def _unwritable_guardrail_change(key: str | int) -> "HTTPException":
    from fastapi import HTTPException

    return HTTPException(
        status_code=400,
        detail={"error": f"Content blocked: the guardrail rewrote a part's {key} into text that is not valid JSON"},
    )


def _written_value(slot: _TextSlot, text: str) -> object:
    if not slot.is_json:
        return text
    try:
        return _DECODED_JSON.validate_python(json.loads(text))
    except json.JSONDecodeError:
        raise _unwritable_guardrail_change(slot.key) from None


def _write_back_texts(text_slots: Sequence[_TextSlot], guardrailed_texts: Sequence[str] | None) -> None:
    if not guardrailed_texts or len(guardrailed_texts) != len(text_slots):
        return
    writes: Final = tuple(
        (slot, _written_value(slot, text)) for slot, text in zip(text_slots, guardrailed_texts) if text != slot.text
    )
    for slot, value in writes:
        _write_field(slot.container, slot.key, value)


def _guardrailed_args(slot: _FunctionCallSlot, tool_call: object) -> Mapping[str, object] | None:
    arguments: Final = _field(_field(tool_call, "function"), "arguments")
    if not isinstance(arguments, str) or arguments == slot.arguments:
        return None
    return _parse_json_dict_or_none(arguments)


def _write_back_function_calls(
    function_calls: Sequence[_FunctionCallSlot], guardrailed_tool_calls: Sequence[object] | None
) -> None:
    if guardrailed_tool_calls is None or len(guardrailed_tool_calls) != len(function_calls):
        return
    rewrites: Final = tuple(
        (slot.call, args)
        for slot, tool_call in zip(function_calls, guardrailed_tool_calls)
        if (args := _guardrailed_args(slot, tool_call)) is not None
    )
    for call, args in rewrites:
        _write_field(call, "args", args)


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
        for text in _slot_texts(_response_text_slots(parsed))
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
        for text in _slot_texts(_response_text_slots(chunk))
    )
    sse_text: Final = "".join(sse for chunk in responses_so_far for sse in (_chunk_sse_text(chunk),) if sse is not None)
    return "".join(object_texts) + "".join(_sse_payload_texts(sse_text))


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
        function_calls: Final = _request_function_calls(request)
        if not text_slots and not function_calls:
            verbose_proxy_logger.debug("Google GenAI guardrail: no request text or function call found, skipping")
            return data
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(_slot_texts(text_slots), request.get("model"), function_calls),
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_slots, guardrailed_inputs.get("texts"))
        _write_back_function_calls(function_calls, guardrailed_inputs.get("tool_calls"))
        return data

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: Mapping[str, object] | None = None,
    ) -> object:
        text_slots: Final = _response_text_slots(response)
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
        _write_back_texts(text_slots, guardrailed_inputs.get("texts"))
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

    def _merged_request_data(
        self,
        request_data: Mapping[str, object] | None,
        user_api_key_dict: Optional["UserAPIKeyAuth"],
        context_key: str,
        context_value: object,
    ) -> dict:  # mutable-ok: CustomGuardrail.apply_guardrail requires a plain dict request payload
        base: Final = request_data if request_data is not None else _EMPTY_REQUEST_DATA
        user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
        context_pairs: Final = ((context_key, context_value),) if context_key not in base else ()
        metadata_pairs: Final = (
            (("litellm_metadata", user_metadata),) if user_metadata and "litellm_metadata" not in base else ()
        )
        return dict((*base.items(), *context_pairs, *metadata_pairs))
