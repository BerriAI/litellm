"""The v1 callback contract as a callback author sees it.

A registered callback observes a call through envelopes it cannot influence, and may
influence one only by returning a patch that Rust validates. The schema itself is owned
by `litellm-rust/crates/callbacks-v1`; this module is its typed Python face and the
process-wide registry. `litellm.rust_bridge.callbacks_v1` is the marshalling the
native call reaches for, and is not part of this surface.

Two ways to register. A single function takes a decorator, which derives the name and the
handler slot for you and returns the function unchanged:

    @on_event("call.succeeded", "call.failed")
    def record(event: EnvelopeV1) -> None: ...

    @before_send
    async def add_header(request: RequestFactsV1) -> WirePatchV1 | None: ...

An object with several handlers, or with state, declares them by the protocol names and
goes through `register`.

Every rejectable mistake is rejected at registration, never during a call, and each static
rule here mirrors one `RegistrationError`: the handler protocols mirror `MissingHandler`
and `HandlerKindMismatch`, and `EventName` mirrors `UnknownEvent`. Annotate `events` with
`frozenset[EventName]` on the object form to get that check before import time; the
decorator infers it from its arguments.
"""

import inspect
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, TypeVar

from typing_extensions import ReadOnly, TypeAliasType, TypedDict

__all__ = (
    "EVENTS",
    "SCHEMA",
    "SUPPORTED_SCHEMAS",
    "AsyncInterceptor",
    "AsyncObserver",
    "CallFactsV1",
    "CallFailedV1",
    "CallStartedV1",
    "CallSucceededV1",
    "Callback",
    "CallbackBase",
    "DuplicateName",
    "EmptyObserverEvents",
    "EnvelopeV1",
    "ErrorFactsV1",
    "EventName",
    "EventV1",
    "HandlerKindMismatch",
    "HeaderPatchV1",
    "Interceptor",
    "InterceptorFn",
    "JSONValue",
    "MissingHandler",
    "Observer",
    "ObserverFn",
    "RegistrationError",
    "RequestFactsV1",
    "RequestSendingV1",
    "ResponseReceivedV1",
    "SchemaVersion",
    "Subscriber",
    "TimingFactsV1",
    "UnknownEvent",
    "UnsupportedSchema",
    "WirePatchV1",
    "before_send",
    "on_event",
    "register",
    "snapshot",
    "unregister",
)

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue = TypeAliasType(  # rebind-ok: named recursive alias required for runtime validation
    "JSONValue", JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
)

EventName: TypeAlias = Literal[
    "call.started",
    "request.sending",
    "response.received",
    "call.succeeded",
    "call.failed",
]
SchemaVersion: TypeAlias = Literal[1]

SCHEMA: Final[SchemaVersion] = 1
SUPPORTED_SCHEMAS: Final[frozenset[SchemaVersion]] = frozenset({SCHEMA})
EVENTS: Final[frozenset[EventName]] = frozenset(
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


class CallbackBase(Protocol):
    """What every callback declares, whatever it handles.

    `schema` is not here: the module you imported from already fixes it. Declaring one
    anyway is still honoured, so a callback written against another version is rejected
    by `register` rather than silently reinterpreted.
    """

    name: str
    events: frozenset[EventName]


class Observer(CallbackBase, Protocol):
    """Receives envelopes on a sync call; its errors are reported and swallowed.

    Observers are asynchronous to the call: an envelope may arrive after the call has
    moved on or returned, in `seq` order within one call and in no order across calls, at
    most once. This host happens to run observers inline; do not rely on it.
    """

    def on_event(self, event: EnvelopeV1) -> None: ...


class AsyncObserver(CallbackBase, Protocol):
    """Receives envelopes on an async call; its errors are reported and swallowed.
    Asynchronous to the call in the same sense as `Observer`."""

    async def async_on_event(self, event: EnvelopeV1) -> None: ...


class Interceptor(CallbackBase, Protocol):
    """Patches the sync request before it is sent; its failure fails the call."""

    def before_send(self, request: RequestFactsV1) -> WirePatchV1 | None: ...


class AsyncInterceptor(CallbackBase, Protocol):
    """Patches the async request before it is sent; its failure fails the call."""

    async def async_before_send(self, request: RequestFactsV1) -> WirePatchV1 | None: ...


Callback: TypeAlias = Observer | AsyncObserver | Interceptor | AsyncInterceptor

# The decorators return the function unchanged, so each is generic over the exact function
# it decorated; without that, the decorated name would widen to the union below and stop
# being directly callable at its own signature.
_SyncObserverFn: TypeAlias = Callable[[EnvelopeV1], None]  # mutable-ok: Callable's bracketed parameter list is call syntax, not a list literal
_AsyncObserverFn: TypeAlias = Callable[[EnvelopeV1], Awaitable[None]]  # mutable-ok: Callable's bracketed parameter list is call syntax, not a list literal
_SyncInterceptorFn: TypeAlias = Callable[[RequestFactsV1], WirePatchV1 | None]  # mutable-ok: Callable's bracketed parameter list is call syntax, not a list literal
_AsyncInterceptorFn: TypeAlias = Callable[[RequestFactsV1], Awaitable[WirePatchV1 | None]]  # mutable-ok: Callable's bracketed parameter list is call syntax, not a list literal

ObserverFn: TypeAlias = _SyncObserverFn | _AsyncObserverFn
InterceptorFn: TypeAlias = _SyncInterceptorFn | _AsyncInterceptorFn
ObserverFnT = TypeVar("ObserverFnT", bound=ObserverFn)  # rebind-ok: a TypeVar must bind to a bare name to be recognised as one
InterceptorFnT = TypeVar("InterceptorFnT", bound=InterceptorFn)  # rebind-ok: a TypeVar must bind to a bare name to be recognised as one


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
    schema: SchemaVersion
    events: frozenset[EventName]
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


def _validate_events(events: frozenset[EventName], observing: bool) -> None:
    unknown: Final = events - EVENTS
    if unknown:
        raise UnknownEvent(f"unknown callback events: {sorted(unknown)}")
    if observing and not events:
        raise EmptyObserverEvents("observer callback events cannot be empty")


def _insert(subscriber: Subscriber) -> None:
    with _REGISTRY.lock:
        if any(existing.name == subscriber.name for existing in _REGISTRY.subscribers):
            raise DuplicateName(f"duplicate callback name: {subscriber.name}")
        _REGISTRY.subscribers = (*_REGISTRY.subscribers, subscriber)


def register(callback: Callback) -> None:
    """Registers an object whose handler methods are named as the protocols declare them."""
    schema: Final = getattr(callback, "schema", SCHEMA)
    if schema not in SUPPORTED_SCHEMAS:
        raise UnsupportedSchema(f"unsupported callback schema: {schema}")
    events: Final = frozenset(callback.events)

    on_event: Final = _handler(callback, "on_event")
    async_on_event: Final = _handler(callback, "async_on_event")
    before_send: Final = _handler(callback, "before_send")
    async_before_send: Final = _handler(callback, "async_before_send")
    handlers: Final = (on_event, async_on_event, before_send, async_before_send)
    if all(handler is None for handler in handlers):
        raise MissingHandler("callback must define at least one handler")
    _validate_events(events, on_event is not None or async_on_event is not None)

    _validate_kind("on_event", on_event, False)
    _validate_kind("async_on_event", async_on_event, True)
    _validate_kind("before_send", before_send, False)
    _validate_kind("async_before_send", async_before_send, True)

    _insert(
        Subscriber(
            name=callback.name,
            schema=schema,
            events=events,
            on_event=on_event,
            async_on_event=async_on_event,
            before_send=before_send,
            async_before_send=async_before_send,
        )
    )


def _derived_name(function: object) -> str:
    module: Final = getattr(function, "__module__", "") or ""
    qualified: Final = getattr(function, "__qualname__", None) or repr(function)
    return f"{module}.{qualified}" if module else str(qualified)


def on_event(*events: EventName, name: str | None = None) -> Callable[[ObserverFnT], ObserverFnT]:
    """Registers one function as an observer of `events`; `async def` selects the async slot.

    The function is returned unchanged, so it stays directly callable and testable. Its
    registry name defaults to `module.qualname`, which is unique by construction.
    """

    def decorate(function: ObserverFnT) -> ObserverFnT:
        subscribed: Final = frozenset(events)
        _validate_events(subscribed, True)
        asynchronous: Final = inspect.iscoroutinefunction(function)
        _insert(
            Subscriber(
                name=name if name is not None else _derived_name(function),
                schema=SCHEMA,
                events=subscribed,
                on_event=None if asynchronous else function,
                async_on_event=function if asynchronous else None,
                before_send=None,
                async_before_send=None,
            )
        )
        return function

    return decorate


def before_send(function: InterceptorFnT) -> InterceptorFnT:
    """Registers one function as an interceptor; `async def` selects the async slot.

    An interceptor subscribes to no events, and its failure fails the call.
    """
    asynchronous: Final = inspect.iscoroutinefunction(function)
    _insert(
        Subscriber(
            name=_derived_name(function),
            schema=SCHEMA,
            events=frozenset(),
            on_event=None,
            async_on_event=None,
            before_send=None if asynchronous else function,
            async_before_send=function if asynchronous else None,
        )
    )
    return function


def unregister(callback: Callback | Subscriber | str) -> None:
    """Drops the callback of this name, given the object, a `Subscriber`, or the name."""
    name: Final = callback if isinstance(callback, str) else callback.name
    with _REGISTRY.lock:
        _REGISTRY.subscribers = tuple(subscriber for subscriber in _REGISTRY.subscribers if subscriber.name != name)


def snapshot() -> tuple[Subscriber, ...]:
    return _REGISTRY.subscribers
