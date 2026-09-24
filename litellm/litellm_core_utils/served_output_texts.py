"""Assistant text the caller received, per choice, so the logging payload stores the response a
post-call guardrail rewrote rather than the provider response the proxy assembled before it ran."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.types.utils import ModelResponse, ModelResponseStream

SERVED_OUTPUT_TEXTS_KEY: Final = "served_output_texts"

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_LIST: Final = TypeAdapter(list[object])
_TEXTS: Final = TypeAdapter(tuple[str | None, ...])

ServedTexts = tuple[str | None, ...]


class _TextBlock(BaseModel):
    type: str
    text: str | None = None


class _AnthropicMessage(BaseModel):
    type: Literal["message"]
    content: list[_TextBlock]


class _ResponsesOutputItem(BaseModel):
    type: str
    content: list[_TextBlock] = []


class _ResponsesResponse(BaseModel):
    object: Literal["response"]
    output: list[_ResponsesOutputItem]


class _ChatChoices(BaseModel):
    choices: list[object]


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
    if chunks and all(isinstance(chunk, ModelResponseStream) for chunk in chunks):
        return _chat_stream_texts(tuple(chunk for chunk in chunks if isinstance(chunk, ModelResponseStream)))
    from litellm.proxy.guardrails.anthropic_sse import assemble_anthropic_sse_stream, is_anthropic_sse_stream

    if not is_anthropic_sse_stream(chunks):
        return None
    assembled: Final = assemble_anthropic_sse_stream(chunks)
    return _chat_texts(assembled) if assembled is not None else None


def _chat_stream_choice_text(chunks: Sequence[ModelResponseStream], index: int) -> str | None:
    contents: Final = tuple(
        content
        for chunk in chunks
        for choice in chunk.choices
        if choice.index == index and isinstance(content := choice.delta.content, str)
    )
    return "".join(contents) if contents else None


def _chat_stream_texts(chunks: Sequence[ModelResponseStream]) -> ServedTexts | None:
    choice_count: Final = max((choice.index + 1 for chunk in chunks for choice in chunk.choices), default=0)
    texts: Final = tuple(_chat_stream_choice_text(chunks, index) for index in range(choice_count))
    return texts if any(text is not None for text in texts) else None


def record_served_output_texts(model_call_details: dict[str, object], texts: ServedTexts | None) -> None:
    if texts is None:
        return
    model_call_details[SERVED_OUTPUT_TEXTS_KEY] = texts  # rebind-ok: model_call_details is the shared kwargs bag


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
    if message is None or message.get("content") == text:
        return choice
    return {**choice_obj, "message": {**message, "content": text}}
