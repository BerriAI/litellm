import inspect
import json
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, TypeGuard  # noqa: TID251  # recursive JSON validation

from typing_extensions import ReadOnly, TypeAliasType, TypedDict

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue = TypeAliasType(  # rebind-ok: named recursive alias required for runtime validation
    "JSONValue", JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
)

SUPPORTED_SCHEMAS: Final = frozenset({1})
EVENTS: Final = frozenset(
    {
        "call.started",
        "request.sending",
        "response.received",
        "call.succeeded",
        "call.failed",
    }
)


class TimingFactsV1(TypedDict):
    start_time: ReadOnly[float]
    end_time: ReadOnly[float]
    duration: ReadOnly[float]


class CallFactsV1(TypedDict):
    start_time: ReadOnly[float]
    asynchronous: ReadOnly[bool]
    metadata: ReadOnly[Mapping[str, JSONValue]]
    metadata_dropped: ReadOnly[Sequence[str]]


class RequestFactsV1(TypedDict):
    model: ReadOnly[str]
    custom_llm_provider: ReadOnly[str]
    optional_params: ReadOnly[JSONValue]
    url: ReadOnly[str]
    headers: ReadOnly[Sequence[tuple[str, str]]]
    body: ReadOnly[JSONValue]


ErrorFactsV1 = TypedDict(  # rebind-ok: runtime TypedDict type for the reserved class field
    "ErrorFactsV1",
    {  # mutable-ok: functional TypedDict syntax requires a dict for the reserved class field
        "class": ReadOnly[str],
        "message": ReadOnly[str],
        "status_code": ReadOnly[int | None],
    },
)


class CallStartedV1(CallFactsV1):
    type: ReadOnly[Literal["call.started"]]


class RequestSendingV1(RequestFactsV1):
    type: ReadOnly[Literal["request.sending"]]


class ResponseReceivedV1(TypedDict):
    type: ReadOnly[Literal["response.received"]]
    body: ReadOnly[str]


class CallSucceededV1(TypedDict):
    type: ReadOnly[Literal["call.succeeded"]]
    timing: ReadOnly[TimingFactsV1]
    streamed: ReadOnly[bool]
    response: ReadOnly[JSONValue]
    response_error: ReadOnly[str | None]


class CallFailedV1(TypedDict):
    type: ReadOnly[Literal["call.failed"]]
    timing: ReadOnly[TimingFactsV1]
    streamed: ReadOnly[bool]
    origin: ReadOnly[Literal["call", "host"]]
    error: ReadOnly[ErrorFactsV1]


EventV1: TypeAlias = CallStartedV1 | RequestSendingV1 | ResponseReceivedV1 | CallSucceededV1 | CallFailedV1


class EnvelopeV1(TypedDict):
    schema: ReadOnly[Literal[1]]
    call_id: ReadOnly[str]
    call_type: ReadOnly[str]
    seq: ReadOnly[int]
    event: ReadOnly[EventV1]


class HeaderPatchV1(TypedDict, total=False):
    remove: ReadOnly[Sequence[str]]
    set: ReadOnly[Sequence[tuple[str, str]]]


class WirePatchV1(TypedDict, total=False):
    headers: ReadOnly[HeaderPatchV1]
    body: ReadOnly[JSONValue]


class Callback(Protocol):
    name: str
    schema: int
    events: frozenset[str]

    def on_event(self, event: EnvelopeV1) -> None: ...

    async def async_on_event(self, event: EnvelopeV1) -> None: ...

    def before_send(self, request: RequestFactsV1) -> WirePatchV1 | None: ...

    async def async_before_send(self, request: RequestFactsV1) -> WirePatchV1 | None: ...


class RegistrationError(ValueError):
    pass


class UnsupportedSchema(RegistrationError):
    pass


class UnknownEvent(RegistrationError):
    pass


class MissingHandler(RegistrationError):
    pass


class EmptyObserverEvents(RegistrationError):
    pass


class HandlerKindMismatch(RegistrationError):
    pass


class DuplicateName(RegistrationError):
    pass


@dataclass(frozen=True, slots=True)
class Subscriber:
    name: str
    schema: int
    events: frozenset[str]
    on_event: object | None
    async_on_event: object | None
    before_send: object | None
    async_before_send: object | None


class _Registry:
    def __init__(self) -> None:
        self.subscribers: tuple[Subscriber, ...] = ()
        self.lock: Final = threading.Lock()


_REGISTRY: Final = _Registry()


def _handler(callback: Callback, name: str) -> object | None:
    return getattr(callback, name, None)


def _validate_kind(name: str, handler: object | None, asynchronous: bool) -> None:
    if handler is None:
        return
    if not callable(handler):
        raise HandlerKindMismatch(f"{name} must be callable")
    is_async: Final = inspect.iscoroutinefunction(handler)
    if is_async != asynchronous:
        expected: Final = "async" if asynchronous else "sync"
        raise HandlerKindMismatch(f"{name} must be {expected}")


def register(callback: Callback) -> None:
    schema: Final = callback.schema
    if schema not in SUPPORTED_SCHEMAS:
        raise UnsupportedSchema(f"unsupported callback schema: {schema}")
    events: Final = frozenset(callback.events)
    unknown: Final = events - EVENTS
    if unknown:
        raise UnknownEvent(f"unknown callback events: {sorted(unknown)}")

    on_event: Final = _handler(callback, "on_event")
    async_on_event: Final = _handler(callback, "async_on_event")
    before_send: Final = _handler(callback, "before_send")
    async_before_send: Final = _handler(callback, "async_before_send")
    handlers: Final = (on_event, async_on_event, before_send, async_before_send)
    if all(handler is None for handler in handlers):
        raise MissingHandler("callback must define at least one handler")
    if (on_event is not None or async_on_event is not None) and not events:
        raise EmptyObserverEvents("observer callback events cannot be empty")

    _validate_kind("on_event", on_event, False)
    _validate_kind("async_on_event", async_on_event, True)
    _validate_kind("before_send", before_send, False)
    _validate_kind("async_before_send", async_before_send, True)

    subscriber: Final = Subscriber(
        name=callback.name,
        schema=schema,
        events=events,
        on_event=on_event,
        async_on_event=async_on_event,
        before_send=before_send,
        async_before_send=async_before_send,
    )
    with _REGISTRY.lock:
        if any(existing.name == subscriber.name for existing in _REGISTRY.subscribers):
            raise DuplicateName(f"duplicate callback name: {subscriber.name}")
        _REGISTRY.subscribers = (*_REGISTRY.subscribers, subscriber)


def unregister(callback: Callback) -> None:
    with _REGISTRY.lock:
        _REGISTRY.subscribers = tuple(
            subscriber for subscriber in _REGISTRY.subscribers if subscriber.name != callback.name
        )


def snapshot() -> tuple[Subscriber, ...]:
    return _REGISTRY.subscribers


def _is_json_value(
    value: object,
) -> TypeGuard[JSONValue]:  # guard-ok: recursively validates the complete JSON value
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(
            _is_json_value(item)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # list items are validated recursively
            for item in value  # pyright: ignore[reportUnknownVariableType]  # list items are validated recursively
        )
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # mapping entries are validated recursively
            for key, item in value.items()  # pyright: ignore[reportUnknownVariableType]  # mapping entries are validated recursively
        )
    return False


def project_response(response: object) -> JSONValue:
    model_dump: Final = getattr(response, "model_dump", None)
    projected: Final = model_dump(mode="json") if callable(model_dump) else response
    json.dumps(projected, allow_nan=False)
    if _is_json_value(projected):
        return projected
    raise TypeError(f"unsupported response type: {type(response).__qualname__}")


def new_call_id() -> str:
    return str(uuid.uuid4())


def report(name: str, event: str, error: BaseException) -> None:
    from litellm._logging import verbose_logger

    exception: Final = (type(error), error, error.__traceback__)
    verbose_logger.exception("callback %s failed while handling %s", name, event, exc_info=exception)
