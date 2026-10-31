import json
import re
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import Enum
from http import HTTPStatus
from itertools import chain, groupby, zip_longest
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


_DROPPED: Final = object()


class _BaseForm(Enum):
    PRINTED = "printed"
    DUMPED = "dumped"


def _repr_owner(kind: type[object]) -> type[object]:
    return next((owner for owner in kind.__mro__ if "__repr__" in owner.__dict__), object)


def _repr_prints_every_item(kind: type[object]) -> bool:
    return kind in (OrderedDict, Counter) or _repr_owner(kind) in (dict, list, tuple, defaultdict)


def _format_prints_every_item(kind: type[object]) -> bool:
    return kind.__str__ is object.__str__ and kind.__format__ is object.__format__ and _repr_prints_every_item(kind)


def _str_prints_every_item(kind: type[object]) -> bool:
    return kind.__str__ is object.__str__ and _repr_owner(kind) in (
        dict,
        list,
        tuple,
        OrderedDict,
        defaultdict,
        Counter,
    )


def _is_dict(value: object) -> TypeIs[Mapping[object, object]]:  # guard-ok: dict items as object
    return isinstance(value, dict)


def _is_sequence(value: object) -> TypeIs[Sequence[object]]:  # guard-ok: items as object
    return isinstance(value, (list, tuple))


def _is_proxy(value: object) -> TypeIs[Mapping[object, object]]:  # guard-ok: mappingproxy cannot be subclassed
    return isinstance(value, MappingProxyType)


def _printed_proxy_snapshot(proxy: Mapping[object, object], printed: str) -> Mapping[object, object] | None:
    snapshot: Final = dict(proxy)
    shown: Final = repr(snapshot)
    return snapshot if printed in (shown, f"mappingproxy({shown})") else None


def _repr_prints_its_fields(model: BaseModel) -> bool:
    kind: Final = type(model)
    return kind.__repr__ is BaseModel.__repr__ and kind.__repr_str__ is BaseModel.__repr_str__


def _plain_scalar(value: str | float) -> str | float:
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int.__int__(value)
    return float.__float__(value)


def _repr_shows_its_value(value: str | float, printed: str) -> bool:
    plain: Final = _plain_scalar(value)
    if isinstance(value, Enum):
        return printed == f"<{type(value).__name__}.{value.name}: {plain!r}>"
    return printed == repr(plain)


def _printed_text(value: object) -> object:
    printed: Final = repr(value)
    if isinstance(value, (str, int, float)) and _repr_shows_its_value(value, printed):
        return _plain_scalar(value)
    return printed


def _dumped_text(value: object) -> object:
    return _plain_scalar(value) if isinstance(value, (str, int, float)) else str(value)


def _text(value: object, form: _BaseForm) -> object:
    try:
        return _printed_text(value) if form is _BaseForm.PRINTED else _dumped_text(value)
    except Exception:  # noqa: BLE001  # hook values run their own __repr__ and __str__, so a failure drops the value
        return _DROPPED


def _json_key(key: object, form: _BaseForm) -> object:
    return key if key is None or type(key) in (str, int, float, bool) else _text(key, form)


def _stored_sequence(value: Sequence[object]) -> Iterator[object]:
    if isinstance(value, list):
        return list[object].__iter__(value)
    if isinstance(value, tuple):
        return tuple[object, ...].__iter__(value)
    return iter(value)


def _stored_items(value: Mapping[object, object]) -> Iterable[tuple[object, object]]:
    return dict[object, object].items(value) if isinstance(value, dict) else value.items()


def _dumps_shown(kind: type[object], shown: Iterable[object], stored: Iterable[object]) -> bool:
    return _str_prints_every_item(kind) or any(left is not right for left, right in zip_longest(shown, stored))


def _shown_sequence(value: object, form: _BaseForm) -> Iterable[object] | None:
    if not _is_sequence(value):
        return None
    if form is _BaseForm.PRINTED:
        return _stored_sequence(value) if _repr_prints_every_item(type(value)) else None
    dumped: Final = tuple(value)
    return dumped if _dumps_shown(type(value), dumped, _stored_sequence(value)) else None


def _shown_items(value: object, form: _BaseForm) -> Iterable[tuple[object, object]] | None:
    if not _is_dict(value):
        return None
    if form is _BaseForm.DUMPED:
        dumped: Final = tuple(value.items())
        stored: Final = chain.from_iterable(_stored_items(value))
        return dumped if _dumps_shown(type(value), chain.from_iterable(dumped), stored) else None
    if not _repr_prints_every_item(type(value)):
        return None
    if isinstance(value, OrderedDict):
        return OrderedDict[object, object].items(value)
    if isinstance(value, Counter):
        return Counter[object].most_common(value)
    return _stored_items(value)


def _expansion(value: object, form: _BaseForm) -> tuple[Iterable[tuple[object, object]], _BaseForm] | None:
    shown_items: Final = _shown_items(value, form)
    if shown_items is not None:
        return shown_items, form
    if _is_proxy(value):
        snapshot: Final = _printed_proxy_snapshot(value, repr(value) if form is _BaseForm.PRINTED else str(value))
        return None if snapshot is None else (snapshot.items(), _BaseForm.PRINTED)
    if form is _BaseForm.PRINTED and isinstance(value, BaseModel) and _repr_prints_its_fields(value):
        return value.__repr_args__(), _BaseForm.PRINTED
    return None


def _json_safe(value: object, form: _BaseForm, path: frozenset[int]) -> object:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if id(value) in path:
        return _text(value, form)
    inner_path: Final = path | frozenset((id(value),))
    try:
        sequence: Final = _shown_sequence(value, form)
        if sequence is not None:
            shown_items: Final = [_json_safe(item, form, inner_path) for item in sequence]
            return [item for item in shown_items if item is not _DROPPED]
        expansion: Final = _expansion(value, form)
        if expansion is None:
            return _text(value, form)
        items, inner = expansion
        pairs: Final = [(_json_key(key, inner), _json_safe(item, inner, inner_path)) for key, item in items]
        return {key: item for key, item in pairs if key is not _DROPPED and item is not _DROPPED}
    except Exception:  # noqa: BLE001  # hook values run their own printers and can nest past the recursion limit
        return _text(value, form)


def _proxy_chunk_data(proxy: Mapping[object, object]) -> str:
    printed: Final = format(proxy, "")
    try:
        snapshot: Final = _printed_proxy_snapshot(proxy, printed)
        shown: Final = None if snapshot is None else _json_safe(snapshot, _BaseForm.PRINTED, frozenset((id(proxy),)))
        return json.dumps(shown) if isinstance(shown, dict) else printed
    except Exception:  # noqa: BLE001  # hook values run their own printers, so the text base printed stands
        return printed


def responses_chunk_data(chunk: object) -> str | None:
    if _is_proxy(chunk):
        return _proxy_chunk_data(chunk)
    if not _is_dict(chunk) or not _format_prints_every_item(type(chunk)):
        return None
    shown: Final = _json_safe(chunk, _BaseForm.PRINTED, frozenset())
    if isinstance(shown, str):
        return shown
    return None if shown is _DROPPED else json.dumps(shown)


def _dumped_detail(error: Mapping[str, object]) -> Mapping[str, object]:
    path: Final = frozenset((id(error),))
    shown: Final = [(key, _json_safe(value, _BaseForm.DUMPED, path)) for key, value in error.items()]
    return {key: value for key, value in shown if value is not _DROPPED}


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
    return int(code) if code is not None and code.isdecimal() else None


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


def _parsed_event(data: str) -> _StreamEvent:
    try:
        return _StreamEvent.model_validate_json(data)
    except ValidationError:
        return _StreamEvent()


def _block_event(lines: Sequence[str]) -> _StreamEvent:
    parsed: Final = _parsed_event("\n".join(_field_values(lines, "data")))
    if parsed.type in _TERMINAL_EVENT_TYPES or not parsed.error or _field_values(lines, "event")[-1:] != ("error",):
        return parsed
    return parsed.model_copy(update={"type": "error"})


def _frame_events(frames: str) -> tuple[_StreamEvent, ...]:
    blocks: Final = groupby(SSE_LINE_BREAK.split(frames), key=bool)
    return tuple(_block_event(tuple(lines)) for filled, lines in blocks if filled)


class ResponsesStreamErrorState:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.model: str | None = None
        self.created_at: int | None = None
        self.sequence_number = -1
        self.terminal_emitted = False
        self._pending_event: _StreamEvent | None = None

    def observe_chunk(self, chunk: object) -> None:
        self._pending_event = _StreamEvent.model_validate(chunk) if isinstance(chunk, (BaseModel, Mapping)) else None

    def _record(self, event: _StreamEvent) -> None:
        if event.sequence_number is not None and 0 <= event.sequence_number <= _MAX_SEQUENCE_NUMBER:
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
        event: Final = self._pending_event
        if event is None:
            return frame
        self._record(event)
        if not event.type or "\n" in event.type or "\r" in event.type:
            return frame
        prefix: Final = f"event: {event.type}\n"
        return prefix.encode("utf-8") + frame if isinstance(frame, bytes) else prefix + frame

    def format_error(self, error: Mapping[str, object]) -> str:
        fields: Final = responses_error_fields(error)
        sequence_number: Final = self.sequence_number + 1
        payload: Final = json.dumps(_error_event(fields, _dumped_detail(error), sequence_number))
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
