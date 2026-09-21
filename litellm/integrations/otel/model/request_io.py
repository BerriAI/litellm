from collections.abc import Mapping, Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.integrations.otel.mappers.utils import json_or_none
from litellm.proxy.guardrails.anthropic_sse import assemble_anthropic_sse_stream, is_raw_sse_stream
from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse
from litellm.types.utils import ModelResponse, ModelResponseStream

_SYSTEM_KEYS: Final = ("system", "instructions")
_TURNS: Final = TypeAdapter(tuple[object, ...])
_MESSAGES: Final = TypeAdapter(list[object] | None)


class _Turn(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[object]


class _AnthropicMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["message"] = Field(exclude=True)
    role: str = "assistant"
    content: object = None


def request_input(data: Mapping[str, object]) -> str | None:
    turns: Final = data.get("messages", data.get("input"))
    if turns is None:
        return None
    return json_or_none((*_system_turns(data), *_user_turns(turns)))


def _system_turns(data: Mapping[str, object]) -> tuple[_Turn, ...]:
    return tuple(_Turn(role="system", content=data[key]) for key in _SYSTEM_KEYS if data.get(key) is not None)


def _user_turns(turns: object) -> tuple[object, ...]:
    if isinstance(turns, str):
        return (_Turn(role="user", content=turns),)
    try:
        return _TURNS.validate_python(turns)
    except ValidationError:
        return (_Turn(role="user", content=turns),)


def response_output(response: object) -> str | None:
    match response:
        case ModelResponse():
            return json_or_none(tuple(choice.message.model_dump(exclude_none=True) for choice in response.choices))
        case ResponsesAPIResponse():
            return json_or_none(response.model_dump(exclude_none=True).get("output"))
        case _:
            return _anthropic_message_output(response)


def _anthropic_message_output(message: object) -> str | None:
    try:
        parsed: Final = _AnthropicMessage.model_validate(message)
    except ValidationError:
        return None
    return json_or_none((parsed.model_dump(),))


def stream_output(chunks: Sequence[object], data: Mapping[str, object]) -> str | None:
    if not chunks:
        return None
    if is_raw_sse_stream(chunks):
        return response_output(assemble_anthropic_sse_stream(chunks))
    if all(isinstance(chunk, ModelResponseStream) for chunk in chunks):
        return response_output(_assembled_chat_stream(chunks, data))
    return response_output(_completed_response(chunks))


def _assembled_chat_stream(chunks: Sequence[object], data: Mapping[str, object]) -> object:
    try:
        return litellm.stream_chunk_builder(  # pyright: ignore[reportUnknownMemberType]  # upstream types chunks as a bare list
            chunks=list(chunks),  # mutable-ok: stream_chunk_builder takes a list
            messages=_MESSAGES.validate_python(data.get("messages")),
        )
    except (litellm.APIError, ValidationError):
        return None


def _completed_response(chunks: Sequence[object]) -> ResponsesAPIResponse | None:
    return next((chunk.response for chunk in reversed(chunks) if isinstance(chunk, ResponseCompletedEvent)), None)
