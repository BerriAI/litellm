"""Assistant text the caller received, per choice, so the logging payload stores the response a
post-call guardrail rewrote rather than the provider response the proxy assembled before it ran."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final, Literal
from weakref import WeakKeyDictionary, WeakValueDictionary

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import ModelResponse, ModelResponseStream, TextCompletionResponse

SERVED_OUTPUT_TEXTS_KEY: Final = "served_output_texts"
_SERVED_OUTPUT_SANITIZER_KEY: Final = "_served_output_sanitizer_id"

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_LIST: Final = TypeAdapter(list[object])
_TEXTS: Final = TypeAdapter(tuple[str | None, ...])


class _OutputLogSanitizer:
    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
        mapping: Final[dict[str, str]] = {}  # mutable-ok: build deduplicated cleartext-to-surrogate map locally
        for cleartext, token in pairs:
            if not cleartext or cleartext == token:
                continue
            for variant in (
                cleartext,
                json.dumps(cleartext)[1:-1],
                json.dumps(cleartext, ensure_ascii=False)[1:-1],
            ):
                if variant and variant != token:
                    mapping[variant] = token
        self._mapping: dict[str, str] = mapping  # mutable-ok: cleared on logging completion zeroization
        self._pattern: re.Pattern[str] | None = None

    def sanitize_text(self, text: str) -> str:
        if not text or not self._mapping:
            return text
        pattern = self._pattern  # rebind-ok: lazily compiled single-pass alternation
        if pattern is None:
            ordered_keys: Final = sorted(self._mapping.keys(), key=len, reverse=True)
            pattern = re.compile("|".join(re.escape(key) for key in ordered_keys))  # rebind-ok: cache compiled pattern
            self._pattern = pattern
        mapping: Final = self._mapping
        return pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)

    def clear(self) -> None:
        self._mapping.clear()
        self._pattern = None


_LOGGING_SANITIZERS: Final = WeakKeyDictionary[object, _OutputLogSanitizer]()
_DETAILS_SANITIZERS: Final = WeakValueDictionary[str, _OutputLogSanitizer]()

ServedTexts = tuple[str | None, ...]


class _TextBlock(LiteLLMBaseModel):
    type: str
    text: str | None = None


class _AnthropicMessage(LiteLLMBaseModel):
    type: Literal["message"]
    content: list[_TextBlock]


class _ResponsesOutputItem(LiteLLMBaseModel):
    type: str
    content: list[_TextBlock] = Field(default=[])


class _ResponsesResponse(LiteLLMBaseModel):
    object: Literal["response"]
    output: list[_ResponsesOutputItem]


class _ChatChoices(LiteLLMBaseModel):
    choices: list[object]


class _TextCompletionChoice(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    index: int | None = None
    text: str | None = None


class _TextCompletionChunk(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    choices: tuple[_TextCompletionChoice, ...]


def _as_json_object(response: object) -> dict[str, object] | None:
    candidate: Final = response.model_dump() if isinstance(response, BaseModel) else response
    try:
        return _JSON_OBJECT.validate_python(candidate)
    except ValidationError:
        return None


def _joined_block_texts(blocks: Sequence[_TextBlock], *, text_type: str) -> str | None:
    texts: Final = tuple(block.text for block in blocks if block.type == text_type and block.text is not None)
    return "".join(texts) if texts else None


def _chat_texts(response: ModelResponse) -> ServedTexts | None:
    texts: Final = tuple(
        choice.message.content if isinstance(choice.message.content, str) else None for choice in response.choices
    )
    return texts if any(text is not None for text in texts) else None


def _anthropic_message_text(response: dict[str, object]) -> str | None:
    try:
        message: Final = _AnthropicMessage.model_validate(response)
    except ValidationError:
        return None
    return _joined_block_texts(message.content, text_type="text")


def _responses_api_text(response: dict[str, object]) -> str | None:
    try:
        parsed: Final = _ResponsesResponse.model_validate(response)
    except ValidationError:
        return None
    texts: Final = tuple(
        text
        for item in parsed.output
        if item.type == "message" and (text := _joined_block_texts(item.content, text_type="output_text")) is not None
    )
    return "".join(texts) if texts else None


def _chat_dict_texts(response: dict[str, object]) -> ServedTexts | None:
    try:
        _ChatChoices.model_validate(response)
        return _chat_texts(ModelResponse(**response))
    except (ValidationError, TypeError, ValueError):
        return None


def served_output_texts(response: object) -> ServedTexts | None:
    if isinstance(response, ModelResponse):
        return _chat_texts(response)
    mapping: Final = _as_json_object(response)
    if mapping is None:
        return None
    chat_texts: Final = _chat_dict_texts(mapping)
    if chat_texts is not None:
        return chat_texts
    anthropic_text: Final = _anthropic_message_text(mapping)
    text: Final = anthropic_text if anthropic_text is not None else _responses_api_text(mapping)
    return (text,) if text is not None else None


def served_stream_output_texts(chunks: Sequence[object]) -> ServedTexts | None:
    chat_chunks: Final = tuple(chunk for chunk in chunks if isinstance(chunk, ModelResponseStream))
    if chunks and len(chat_chunks) == len(chunks):
        return _indexed_texts_by_choice(tuple(chain.from_iterable(_chat_chunk_texts(chunk) for chunk in chat_chunks)))
    if chunks and all(isinstance(chunk, TextCompletionResponse) for chunk in chunks):
        return _indexed_texts_by_choice(
            tuple(chain.from_iterable(_text_completion_chunk_texts(chunk) for chunk in chunks))
        )
    from litellm.llms.anthropic.messages.stream_assembly import (
        assemble_anthropic_sse_stream,
        is_anthropic_sse_stream,
    )

    if not is_anthropic_sse_stream(chunks):
        return None
    assembled: Final = assemble_anthropic_sse_stream(chunks)
    return _chat_texts(assembled) if assembled is not None else None


def _chat_chunk_texts(chunk: ModelResponseStream) -> tuple[tuple[int, str | None], ...]:
    return tuple(
        (choice.index or 0, content if isinstance(content := choice.delta.content, str) else None)
        for choice in chunk.choices
    )


def _text_completion_chunk_texts(chunk: object) -> tuple[tuple[int, str | None], ...]:
    parsed: Final = _TextCompletionChunk.model_validate(_as_json_object(chunk))
    return tuple((choice.index or 0, choice.text) for choice in parsed.choices)


def _joined_choice_text(indexed_texts: Sequence[tuple[int, str | None]], index: int) -> str | None:
    texts: Final = tuple(text for text_index, text in indexed_texts if text_index == index and text is not None)
    return "".join(texts) if texts else None


def _indexed_texts_by_choice(indexed_texts: Sequence[tuple[int, str | None]]) -> ServedTexts | None:
    choice_count: Final = max((index + 1 for index, _ in indexed_texts), default=0)
    texts: Final = tuple(_joined_choice_text(indexed_texts, index) for index in range(choice_count))
    return texts if any(text is not None for text in texts) else None


def register_served_output_sanitizer(
    logging_obj: object,
    model_call_details: dict[str, object] | None,  # mutable-ok: stamps sanitizer id on Logging.model_call_details
    pairs: Sequence[tuple[str, str]],
    sanitizer_id: str,
) -> None:
    if logging_obj is None or not pairs:
        return
    sanitizer: Final = _OutputLogSanitizer(pairs)
    try:
        _LOGGING_SANITIZERS[logging_obj] = sanitizer
    except TypeError:
        return
    _DETAILS_SANITIZERS[sanitizer_id] = sanitizer
    if model_call_details is not None:
        model_call_details[_SERVED_OUTPUT_SANITIZER_KEY] = sanitizer_id  # rebind-ok: bind sanitizer id to call details


def clear_served_output_sanitizer(logging_obj: object) -> None:
    if logging_obj is None:
        return
    try:
        sanitizer: Final = _LOGGING_SANITIZERS.pop(logging_obj, None)
    except TypeError:
        return
    if sanitizer is not None:
        sanitizer.clear()


def sanitize_logged_output_str(request_data: Mapping[str, object], rendered: str | None) -> str | None:
    if rendered is None:
        return None
    logging_obj: Final = request_data.get("litellm_logging_obj")
    if logging_obj is None:
        return rendered
    try:
        sanitizer: Final = _LOGGING_SANITIZERS.get(logging_obj)
    except TypeError:
        return rendered
    return sanitizer.sanitize_text(rendered) if sanitizer is not None else rendered


def record_served_output_texts(model_call_details: dict[str, object], texts: ServedTexts | None) -> None:
    if texts is None:
        return
    sanitizer_id: Final = model_call_details.get(_SERVED_OUTPUT_SANITIZER_KEY)
    sanitizer: Final = _DETAILS_SANITIZERS.get(sanitizer_id) if isinstance(sanitizer_id, str) else None
    recorded: Final = (
        tuple(sanitizer.sanitize_text(t) if t is not None else None for t in texts) if sanitizer is not None else texts
    )
    model_call_details[SERVED_OUTPUT_TEXTS_KEY] = recorded  # rebind-ok: model_call_details is the shared kwargs bag


def overlay_served_output_texts(
    response_obj: dict[str, object] | str | list[object] | None, served_texts: object
) -> dict[str, object] | str | list[object] | None:
    if not isinstance(response_obj, dict):
        return response_obj
    logged: Final = _as_json_object(response_obj)
    if logged is None:
        return response_obj
    try:
        texts: Final = _TEXTS.validate_python(served_texts)
        choices: Final = _JSON_LIST.validate_python(logged.get("choices"))
    except ValidationError:
        return response_obj
    return {
        **logged,
        "choices": [
            _choice_with_text(choice, texts[index]) if index < len(texts) else choice
            for index, choice in enumerate(choices)
        ],
    }


def _choice_with_text(choice: object, text: str | None) -> object:
    choice_obj: Final = _as_json_object(choice)
    if choice_obj is None or text is None:
        return choice
    message: Final = _as_json_object(choice_obj.get("message"))
    if message is not None:
        return choice if message.get("content") == text else {**choice_obj, "message": {**message, "content": text}}
    if isinstance(choice_obj.get("text"), str):
        return {**choice_obj, "text": text}
    return choice
