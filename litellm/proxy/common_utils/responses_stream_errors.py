import json
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from itertools import accumulate, chain, groupby, islice, repeat, takewhile
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator
from typing_extensions import TypeIs  # noqa: TID251  # the guards narrow hook values to object items, isinstance cannot

from litellm._logging import redact_internal_details_from_client_message
from litellm._uuid import uuid
from litellm.exceptions import MidStreamFallbackError
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import ResponseFailedEvent, ResponsesAPIResponse, ResponsesAPIStreamEvents


class _ResponseIdentity(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: str | None = None
    model: str | None = None
    created_at: int | None = None


class _StreamEvent(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    type: str | None = None
    sequence_number: int | None = None
    response: _ResponseIdentity | None = None


class _FramedEvent(_StreamEvent):
    error: object = None


_LAX_INT: Final = TypeAdapter(int)


class _FailureDetails(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    message: str | None = None
    code: str | int | None = None
    type: str | None = None
    status_code: int | None = None

    @field_validator("status_code", mode="before")
    @classmethod
    def normalize_status_code(cls, value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            return _LAX_INT.validate_python(value)
        except ValidationError:
            return None

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: object) -> str | None:
        return value if isinstance(value, str) else None

    @field_validator("code", mode="before")
    @classmethod
    def normalize_code(cls, value: object) -> str | int | None:
        return value if isinstance(value, (str, int)) and not isinstance(value, bool) else None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> str | None:
        return value if isinstance(value, str) else None

    @classmethod
    def of_exception(cls, exception: Exception) -> "_FailureDetails":
        return cls.model_validate({field: getattr(exception, field, None) for field in cls.model_fields})


class _ErrorPayload(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    error: Mapping[str, object] | str

    def error_object(self) -> Mapping[str, object]:
        return {"message": self.error} if isinstance(self.error, str) else self.error


_JSON_LEAF_TYPES: Final = frozenset((str, int, float, bool, type(None)))
_MAX_PLAIN_JSON_DEPTH: Final = 100


def _is_dict(value: object) -> TypeIs[Mapping[object, object]]:  # guard-ok: dict items as object
    return isinstance(value, dict)


def _is_sequence(value: object) -> TypeIs[Sequence[object]]:  # guard-ok: items as object
    return isinstance(value, (list, tuple))


def _is_plain_node(value: object) -> bool:
    kind: Final = type(value)
    if kind in _JSON_LEAF_TYPES or kind in (list, tuple):
        return True
    if isinstance(value, Enum):
        return isinstance(value, (str, int)) and kind.__repr__ is Enum.__repr__
    return kind is dict and _is_dict(value) and all(type(key) is str for key in value)


def _children(value: object) -> tuple[object, ...]:
    if _is_dict(value):
        return tuple(value.values())
    return tuple(value) if _is_sequence(value) else ()


@dataclass(frozen=True, slots=True)
class _PlainJsonLevel:
    values: tuple[object, ...]
    seen: frozenset[int]


def _next_level(level: _PlainJsonLevel, _: None) -> _PlainJsonLevel:
    unseen: Final = {id(value): value for value in level.values if id(value) not in level.seen}
    return _PlainJsonLevel(
        values=tuple(chain.from_iterable(map(_children, unseen.values()))),
        seen=level.seen.union(unseen),
    )


def _settles(level: _PlainJsonLevel) -> bool:
    return not level.values or not all(map(_is_plain_node, level.values))


def _is_plain_json(chunk: object) -> bool:
    first: Final = _PlainJsonLevel(values=(chunk,), seen=frozenset())
    levels: Final = islice(accumulate(repeat(None), _next_level, initial=first), _MAX_PLAIN_JSON_DEPTH)
    return next((not level.values for level in levels if _settles(level)), False)


def _carries_a_type(chunk: object) -> bool:
    return _is_dict(chunk) and type(chunk) is dict and "type" in chunk


def responses_chunk_data(chunk: object) -> str | None:
    try:
        return json.dumps(chunk, allow_nan=False) if _carries_a_type(chunk) and _is_plain_json(chunk) else None
    except (ValueError, RecursionError):
        return None


def sse_error_object(data: str | bytes) -> Mapping[str, object] | None:
    try:
        return _ErrorPayload.model_validate_json(data).error_object()
    except ValidationError:
        return None


def event_error_object(event: Mapping[str, object]) -> Mapping[str, object] | None:
    try:
        return _ErrorPayload.model_validate(event).error_object() or None
    except ValidationError:
        return None


def _original_failure(exception: Exception) -> Exception:
    current = exception  # rebind-ok: the recursion gate requires iterative wrapper traversal
    while isinstance(current, MidStreamFallbackError) and current.original_exception is not None:
        current = current.original_exception
    return current


def _failure_details(original: Exception) -> _FailureDetails:
    mapped: Final = _FailureDetails.of_exception(original)
    body: Final = getattr(original, "body", None)
    if not isinstance(body, Mapping):
        return mapped
    upstream: Final = _FailureDetails.model_validate(body)
    return _FailureDetails(
        message=upstream.message or mapped.message,
        code=upstream.code if upstream.code is not None else mapped.code,
        type=upstream.type or mapped.type,
        status_code=mapped.status_code,
    )


_CLIENT_ERROR_CODES: Final = MappingProxyType(
    {
        int(HTTPStatus.UNAUTHORIZED): "authentication_error",
        int(HTTPStatus.FORBIDDEN): "permission_error",
        int(HTTPStatus.NOT_FOUND): "not_found_error",
        int(HTTPStatus.REQUEST_TIMEOUT): "request_timeout",
        int(HTTPStatus.TOO_MANY_REQUESTS): "rate_limit_exceeded",
    }
)


def _status_error_code(status_code: int | None) -> str:
    if status_code is None or not HTTPStatus.BAD_REQUEST <= status_code < HTTPStatus.INTERNAL_SERVER_ERROR:
        return "server_error"
    return _CLIENT_ERROR_CODES.get(status_code, "invalid_request_error")


def _status_of(code: str | int | None) -> int | None:
    if isinstance(code, int):
        return code
    if code is None or not code.isdecimal():
        return None
    with suppress(ValueError):
        return int(code)
    return None


def _response_error_code(details: _FailureDetails) -> str:
    for value in (details.code, details.type):
        if value == "insufficient_quota":
            return "insufficient_quota"
        if value in (429, "429") or isinstance(value, str) and value.startswith("rate_limit"):
            return "rate_limit_exceeded"
    if isinstance(details.code, str) and details.code and not details.code.isdecimal():
        return details.code
    return _status_error_code(details.status_code)


_STATUS_PHRASES: Final = MappingProxyType({int(status): status.phrase for status in HTTPStatus})


def _status_phrase(status_code: int | None) -> str:
    return _STATUS_PHRASES.get(status_code, "Unknown error") if status_code is not None else "Unknown error"


def responses_error_fields(error: Mapping[str, object]) -> Mapping[str, object]:
    details: Final = _FailureDetails.model_validate(error)
    status_code: Final = _status_of(details.code)
    return {
        "code": _response_error_code(details.model_copy(update={"status_code": status_code})),
        "message": details.message or _status_phrase(status_code),
    }


def _error_event(
    fields: Mapping[str, object], error: Mapping[str, object], sequence_number: int
) -> Mapping[str, object]:
    return {
        "type": ResponsesAPIStreamEvents.ERROR.value,
        **fields,
        "param": error.get("param"),
        "sequence_number": sequence_number,
        "error": error,
    }


SSE_LINE_BREAK: Final = re.compile(r"\r\n|\r|\n")
_TERMINAL_EVENT_TYPES: Final = frozenset(("response.completed", "response.failed", "response.incomplete", "error"))
_MAX_SEQUENCE_NUMBER: Final = 2**63 - 1


def _field_values(lines: Sequence[str], name: str) -> tuple[str, ...]:
    prefix: Final = f"{name}:"
    return tuple(line.removeprefix(prefix).removeprefix(" ") for line in lines if line.startswith(prefix))


def _parsed_event(data: str) -> _FramedEvent:
    try:
        return _FramedEvent.model_validate_json(data)
    except ValidationError:
        return _FramedEvent()


def _in_range_sequence(event: _FramedEvent) -> _FramedEvent:
    sequence_number: Final = event.sequence_number
    if sequence_number is None or 0 <= sequence_number < _MAX_SEQUENCE_NUMBER:
        return event
    return event.model_copy(update={"sequence_number": None})


def _names_an_error_object(event: _FramedEvent, lines: Sequence[str]) -> bool:
    return _is_dict(event.error) and bool(event.error) and _field_values(lines, "event")[-1:] == ("error",)


def _block_data(lines: Sequence[str]) -> str:
    return "\n".join(_field_values(lines, "data"))


def _block_event(lines: Sequence[str]) -> _StreamEvent:
    parsed: Final = _in_range_sequence(_parsed_event(_block_data(lines)))
    if parsed.type in _TERMINAL_EVENT_TYPES or not _names_an_error_object(parsed, lines):
        return parsed
    return parsed.model_copy(update={"type": "error"})


def _blocks(frames: str) -> Iterator[tuple[str, ...]]:
    return (tuple(lines) for filled, lines in groupby(SSE_LINE_BREAK.split(frames), key=bool) if filled)


def _frame_events(frames: str) -> tuple[_StreamEvent, ...]:
    return tuple(_block_event(lines) for lines in _blocks(frames))


def _before_done(lines: Sequence[str]) -> bool:
    return _block_data(lines).strip() != "[DONE]"


def _terminal_blocks(frame: str) -> tuple[tuple[tuple[str, ...], _StreamEvent], ...]:
    read: Final = ((lines, _block_event(lines)) for lines in takewhile(_before_done, _blocks(frame)))
    return tuple((lines, event) for lines, event in read if event.type in _TERMINAL_EVENT_TYPES)


def frame_payloads(frame: str) -> tuple[tuple[str, ...], bool]:
    payloads: Final = tuple(_block_data(lines).strip() for lines in _blocks(frame) if _field_values(lines, "data"))
    return tuple(takewhile(lambda payload: payload != "[DONE]", payloads)), payloads[:1] == ("[DONE]",)


def frame_error_data(frame: str) -> str | None:
    terminal: Final = _terminal_blocks(frame)
    if not terminal:
        return None
    lines, event = terminal[-1]
    return _block_data(lines) if event.type == "error" else None


def _observed_event(chunk: object) -> _StreamEvent | None:
    if not isinstance(chunk, (BaseModel, Mapping)):
        return None
    try:
        return _StreamEvent.model_validate(chunk)
    except ValidationError:
        return None


def _wire_event(frame: str | bytes) -> _StreamEvent | None:
    text: Final = frame.decode("utf-8", errors="replace") if isinstance(frame, bytes) else frame
    try:
        return _StreamEvent.model_validate_json(text.removeprefix("data: "))
    except ValidationError:
        return None


class ResponsesStreamErrorState:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.model: str | None = None
        self.created_at: int | None = None
        self.sequence_number = -1
        self.terminal_emitted = False
        self._pending_event: _StreamEvent | None = None

    def observe_chunk(self, chunk: object) -> None:
        self._pending_event = _observed_event(chunk)

    def _record(self, event: _StreamEvent) -> None:
        if event.sequence_number is not None:
            self.sequence_number = max(self.sequence_number, event.sequence_number)
        if event.response is not None:
            self.response_id = event.response.id or self.response_id
            self.model = event.response.model or self.model
            if event.response.created_at is not None:
                self.created_at = event.response.created_at
        if event.type in _TERMINAL_EVENT_TYPES:
            self.terminal_emitted = True

    def observe_frames(self, frames: str) -> str:
        for event in _frame_events(frames):
            self._record(event)
        return frames

    def mark_emitted(self, frame: str | bytes) -> str | bytes:
        observed: Final = self._pending_event
        if observed is None:
            return frame
        event: Final = _wire_event(frame)
        self._record(observed if event is None else event)
        if event is None or not event.type or "\n" in event.type or "\r" in event.type:
            return frame
        prefix: Final = f"event: {event.type}\n"
        return prefix.encode("utf-8") + frame if isinstance(frame, bytes) else prefix + frame

    def format_error(self, error: Mapping[str, object]) -> str:
        fields: Final = responses_error_fields(error)
        sequence_number: Final = self.sequence_number + 1
        payload: Final = json.dumps(_error_event(fields, error, sequence_number), default=str)
        return f"event: error\ndata: {payload}\n\n"

    def format_error_message(self, frame: str) -> str:
        error: Final = sse_error_object(frame.removeprefix("data: "))
        return frame if error is None else self.format_error(error)

    def format_failure(self, exception: Exception) -> str | None:
        if self.terminal_emitted:
            return None
        original: Final = _original_failure(exception)
        details: Final = _failure_details(original)
        response: Final = ResponsesAPIResponse.model_validate(
            MappingProxyType(
                {
                    "id": self.response_id or f"resp_{uuid.uuid4().hex}",
                    "object": "response",
                    "created_at": self.created_at if self.created_at is not None else int(time.time()),
                    "model": self.model,
                    "status": "failed",
                    "output": (),
                    "error": MappingProxyType(
                        {
                            "code": _response_error_code(details),
                            "message": redact_internal_details_from_client_message(details.message or str(original)),
                        }
                    ),
                }
            )
        )
        event: Final = ResponseFailedEvent.model_validate(
            MappingProxyType(
                {
                    "type": ResponsesAPIStreamEvents.RESPONSE_FAILED,
                    "response": response,
                    "sequence_number": self.sequence_number + 1,
                }
            )
        )
        payload: Final = event.model_dump_json(exclude_none=True)
        self.terminal_emitted = True
        return f"event: response.failed\ndata: {payload}\n\n"
