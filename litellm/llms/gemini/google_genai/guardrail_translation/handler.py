"""
Google GenAI generateContent handler for Unified Guardrails.

Extracts text from generateContent requests (systemInstruction.parts[].text
and contents[].parts[].text) and responses (candidates[].content.parts[].text),
applies the guardrail, and
writes the guardrailed text back in place. Requests and responses may be
dicts (wire format) or google-genai SDK objects; streaming chunks may
additionally be raw SSE frames, which are scanned for detection (a blocking
guardrail raises) without rewriting the frames.
"""

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamTransformSink,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth

_EMPTY_REQUEST_DATA: Final[Mapping[str, object]] = MappingProxyType({})


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


def _part_texts(text_parts: Sequence[object]) -> tuple[str, ...]:
    return tuple(text for part in text_parts for text in (_part_text(part),) if text is not None)


def _texts_payload(
    texts: Sequence[str],
) -> list[str]:  # mutable-ok: GenericGuardrailAPIInputs.texts is declared list[str]
    return list(texts)  # mutable-ok: GenericGuardrailAPIInputs.texts is declared list[str]


def _write_back_texts(text_parts: Sequence[object], guardrailed_texts: Sequence[str] | None) -> None:
    if not guardrailed_texts or len(guardrailed_texts) != len(text_parts):
        return
    for part, text in zip(text_parts, guardrailed_texts):
        _write_part_text(part, text)


def _parse_json_dict_or_none(payload: str) -> Mapping[str, object] | None:
    try:
        parsed: Final = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _sse_payload_texts(sse_text: str) -> tuple[str, ...]:
    return tuple(
        text
        for line in sse_text.splitlines()
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
        text_parts: Final = _request_text_parts(data)
        if not text_parts:
            verbose_proxy_logger.debug("Google GenAI guardrail: no request text found, skipping")
            return data
        model: Final = data.get("model")
        inputs: Final = (
            GenericGuardrailAPIInputs(texts=_texts_payload(_part_texts(text_parts)), model=model)
            if isinstance(model, str)
            else GenericGuardrailAPIInputs(texts=_texts_payload(_part_texts(text_parts)))
        )
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        _write_back_texts(text_parts, guardrailed_inputs.get("texts"))
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
        if not text_parts:
            verbose_proxy_logger.debug("Google GenAI guardrail: no response text found, skipping")
            return response
        guardrail_request_data: Final = self._merged_request_data(
            request_data=request_data,
            user_api_key_dict=user_api_key_dict,
            context_key="response",
            context_value=response,
        )
        model: Final = guardrail_request_data.get("model")
        inputs: Final = (
            GenericGuardrailAPIInputs(texts=_texts_payload(_part_texts(text_parts)), model=model)
            if isinstance(model, str)
            else GenericGuardrailAPIInputs(texts=_texts_payload(_part_texts(text_parts)))
        )
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
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
        return dict((*base.items(), *context_pairs, *metadata_pairs))  # mutable-ok: apply_guardrail takes a plain dict
