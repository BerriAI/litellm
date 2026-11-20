"""
Google GenAI generateContent handler for Unified Guardrails.

Extracts text from generateContent requests (systemInstruction.parts[].text
and contents[].parts[].text) and responses (candidates[].content.parts[].text),
applies the guardrail, and
writes the guardrailed text back in place. A guardrail that opts into
scan_tool_calls_as_text also gets every functionCall part as a tool call, and
masked request arguments are written back into functionCall.args. Requests and
responses may be dicts (wire format) or google-genai SDK objects; streaming
chunks may additionally be raw SSE frames, which are scanned for detection (a
blocking guardrail raises) without rewriting the frames.
"""

import json
from collections.abc import Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional

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
from litellm.llms.custom_httpx.sse_lines import split_sse_lines
from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_EMPTY_REQUEST_DATA: Final[Mapping[str, object]] = MappingProxyType({})
_DECODED_JSON: Final = TypeAdapter(object)


def _field(container: object, name: str) -> object | None:
    if isinstance(container, dict):
        return container.get(name)
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
    parts: Final = _field(content, "parts")
    if not isinstance(parts, (list, tuple)):
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
    content_list: Final = (
        (contents,) if isinstance(contents, dict) else tuple(contents) if isinstance(contents, list) else ()
    )
    return (
        *_content_text_parts(_system_instruction(data)),
        *(part for content in content_list for part in _content_text_parts(content)),
    )


def _response_text_parts(response: object) -> tuple[object, ...]:
    candidates: Final = _field(response, "candidates")
    if not isinstance(candidates, (list, tuple)):
        return ()
    return tuple(part for candidate in candidates for part in _content_text_parts(_field(candidate, "content")))


def _function_call(part: object) -> object | None:
    camel_case: Final = _field(part, "functionCall")
    return camel_case if camel_case is not None else _field(part, "function_call")


def _content_function_calls(content: object) -> tuple[object, ...]:
    calls: Final = (_function_call(part) for part in as_object_list(_field(content, "parts")) or ())
    return tuple(call for call in calls if call is not None)


def _request_function_calls(data: Mapping[str, object]) -> tuple[object, ...]:
    contents: Final = data.get("contents")
    single_content: Final = as_str_mapping(contents)
    content_list: Final = (single_content,) if single_content is not None else as_object_list(contents) or ()
    return tuple(chain.from_iterable(_content_function_calls(content) for content in content_list))


def _response_function_calls(response: object) -> tuple[object, ...]:
    candidates: Final = as_object_list(_field(response, "candidates")) or ()
    return tuple(chain.from_iterable(_content_function_calls(_field(candidate, "content")) for candidate in candidates))


def _tool_call(function_call: object, index: int) -> ChatCompletionToolCallChunk:
    name: Final = _field(function_call, "name")
    call_id: Final = _field(function_call, "id")
    return ChatCompletionToolCallChunk(
        id=call_id if isinstance(call_id, str) else None,
        type="function",
        function={
            "name": name if isinstance(name, str) else None,
            "arguments": json.dumps(_field(function_call, "args") or {}, ensure_ascii=False, default=str),
        },
        index=index,
    )


def _tool_calls_payload(function_calls: Sequence[object]) -> tuple[ChatCompletionToolCallChunk, ...]:
    return tuple(_tool_call(function_call, index) for index, function_call in enumerate(function_calls))


def _write_function_call_args(function_call: object, args: Mapping[str, object]) -> None:
    if isinstance(function_call, dict):
        function_call["args"] = dict(args)  # rebind-ok: guardrail write-back rewrites the caller's part in place
        return
    setattr(function_call, "args", dict(args))  # noqa: B010  # SDK parts are typed as object here


def _guardrailed_args(returned_tool_call: object, guardrail_name: str | None) -> Mapping[str, object]:
    arguments: Final = stream_item_field(stream_item_field(returned_tool_call, "function"), "arguments")
    parsed: Final = _parse_json_dict_or_none(arguments) if isinstance(arguments, str) else None
    if parsed is None:
        raise unappliable_request_rewrite(guardrail_name)
    return parsed


def _write_back_function_calls(
    function_calls: Sequence[object],
    sent_tool_calls: Sequence[ChatCompletionToolCallChunk],
    returned_tool_calls: Sequence[object] | None,
    guardrail_name: str | None,
) -> None:
    if returned_tool_calls is None:
        return
    if len(returned_tool_calls) != len(sent_tool_calls):
        raise unappliable_request_rewrite(guardrail_name)
    changed: Final = tuple(
        (function_call, returned)
        for function_call, sent, returned in zip(function_calls, sent_tool_calls, returned_tool_calls, strict=True)
        if stream_item_field(stream_item_field(returned, "function"), "arguments") != sent["function"].get("arguments")
    )
    for function_call, returned in changed:
        _write_function_call_args(function_call, _guardrailed_args(returned, guardrail_name))


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
    return tuple(chain.from_iterable(_response_function_calls(chunk) for chunk in parsed if chunk is not None))


def _accumulated_stream_function_calls(responses_so_far: Sequence[object]) -> tuple[object, ...]:
    object_calls: Final = tuple(
        chain.from_iterable(
            _response_function_calls(chunk) for chunk in responses_so_far if _chunk_sse_text(chunk) is None
        )
    )
    return object_calls + _sse_payload_function_calls(_joined_sse_text(responses_so_far))


def _guardrail_inputs(
    texts: Sequence[str], tool_calls: Sequence[ChatCompletionToolCallChunk], model: object
) -> GenericGuardrailAPIInputs:
    model_fields: Final[GenericGuardrailAPIInputs] = {"model": model} if isinstance(model, str) else {}
    tool_call_fields: Final[GenericGuardrailAPIInputs] = {"tool_calls": list(tool_calls)} if tool_calls else {}
    return {"texts": _texts_payload(texts), **model_fields, **tool_call_fields}


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
        function_calls: Final = (
            _request_function_calls(data) if effective_scan_tool_calls_as_text_for_guardrail(guardrail_to_apply) else ()
        )
        if not text_parts and not function_calls:
            verbose_proxy_logger.debug("Google GenAI guardrail: no request text found, skipping")
            return data
        tool_calls: Final = _tool_calls_payload(function_calls)
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(_part_texts(text_parts), tool_calls, data.get("model")),
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_parts, guardrailed_inputs.get("texts"))
        _write_back_function_calls(
            function_calls,
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
        function_calls: Final = (
            _response_function_calls(response)
            if effective_scan_tool_calls_as_text_for_guardrail(guardrail_to_apply)
            else ()
        )
        if not text_parts and not function_calls:
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
                _part_texts(text_parts), _tool_calls_payload(function_calls), guardrail_request_data.get("model")
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
        function_calls: Final = (
            _accumulated_stream_function_calls(responses_so_far)
            if effective_scan_tool_calls_as_text_for_guardrail(guardrail_to_apply)
            else ()
        )
        if not accumulated_text and not function_calls:
            return responses_so_far
        guardrail_request_data: Final = self._merged_request_data(
            request_data=request_data,
            user_api_key_dict=user_api_key_dict,
            context_key="responses_so_far",
            context_value=responses_so_far,
        )
        _guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=_guardrail_inputs(
                (accumulated_text,) if accumulated_text else (), _tool_calls_payload(function_calls), None
            ),
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
