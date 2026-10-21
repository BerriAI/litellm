from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import Lock
from typing import Final, Protocol
from uuid import uuid4

from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.configuration import Decision


@dataclass(frozen=True, slots=True)
class RouteObservation:
    event: str
    trace_id: str | None
    attempt_id: str | None
    parent_attempt_id: str | None
    depth: int
    route: Route | None
    context: RouteContext | None = None
    decision: Decision | None = None
    outcome: str | None = None
    reason: str | None = None
    engine: str | None = None


class RouteObserver(Protocol):
    def record(self, observation: RouteObservation) -> None: ...


@dataclass(frozen=True, slots=True)
class _RequestScope:
    trace_id: str
    depth: int = 0
    parent_attempt_id: str | None = None


class _EventBuffer:
    def __init__(self) -> None:
        self._events: tuple[RouteObservation, ...] = ()
        self._lock: Final = Lock()

    def append(self, event: RouteObservation) -> None:
        with self._lock:
            self._events = (*self._events, event)

    def snapshot(self) -> tuple[RouteObservation, ...]:
        with self._lock:
            return self._events


_OBSERVER: Final[ContextVar[RouteObserver | None]] = ContextVar("litellm_rust_route_observer", default=None)
_REQUEST: Final[ContextVar[_RequestScope | None]] = ContextVar("litellm_rust_request_scope", default=None)
_CURRENT_ATTEMPT: Final[ContextVar[tuple[str, int, str | None] | None]] = ContextVar(
    "litellm_rust_current_attempt", default=None
)
_EVENTS: Final[ContextVar[_EventBuffer | None]] = ContextVar("litellm_rust_route_events", default=None)


def set_observer(observer: RouteObserver | None) -> Token[RouteObserver | None]:
    return _OBSERVER.set(observer)


def reset_observer(token: Token[RouteObserver | None]) -> None:
    _OBSERVER.reset(token)


@contextmanager
def request_scope(trace_id: str) -> Generator[None, None, None]:
    request_token: Final = _REQUEST.set(_RequestScope(trace_id))
    try:
        yield
    finally:
        _REQUEST.reset(request_token)


@contextmanager
def trace_scope() -> Generator[None, None, None]:
    token: Final = _EVENTS.set(_EventBuffer())
    try:
        yield
    finally:
        _EVENTS.reset(token)


def current_events() -> tuple[RouteObservation, ...]:
    events: Final = _EVENTS.get()
    return () if events is None else events.snapshot()


def record_event(observation: RouteObservation) -> None:
    events: Final = _EVENTS.get()
    if events is not None:
        events.append(observation)


def observe_early(route: Route, reason: str) -> None:
    request: Final = _REQUEST.get()
    current: Final = _CURRENT_ATTEMPT.get()
    observer: Final = _OBSERVER.get()
    if observer is None:
        return
    observer.record(
        RouteObservation(
            event="early",
            trace_id=None if request is None else request.trace_id,
            attempt_id=None,
            parent_attempt_id=(
                current[0] if current is not None else None if request is None else request.parent_attempt_id
            ),
            depth=(current[1] + 1 if current is not None else 0 if request is None else request.depth),
            route=route,
            reason=reason,
        )
    )


@contextmanager
def observed_attempt(context: RouteContext, selected: Decision) -> Generator[str, None, None]:
    request: Final = _REQUEST.get()
    observer: Final = _OBSERVER.get()
    if observer is None:
        yield ""
        return
    attempt_id: Final = uuid4().hex
    trace_id: Final = None if request is None else request.trace_id
    parent: Final = _CURRENT_ATTEMPT.get()
    parent_attempt_id: Final = (
        parent[0] if parent is not None else None if request is None else request.parent_attempt_id
    )
    depth: Final = parent[1] + 1 if parent is not None else 0 if request is None else request.depth
    attempt_token: Final = _CURRENT_ATTEMPT.set((attempt_id, depth, parent_attempt_id))
    observer.record(
        RouteObservation(
            event="selected",
            trace_id=trace_id,
            attempt_id=attempt_id,
            parent_attempt_id=parent_attempt_id,
            depth=depth,
            route=context.route,
            context=context,
            decision=selected,
        )
    )
    try:
        yield attempt_id
    except BaseException:
        observer.record(
            RouteObservation(
                event="terminal",
                trace_id=trace_id,
                attempt_id=attempt_id,
                parent_attempt_id=parent_attempt_id,
                depth=depth,
                route=context.route,
                context=context,
                decision=selected,
                outcome="failure",
                reason="execution_error",
            )
        )
        raise
    finally:
        _CURRENT_ATTEMPT.reset(attempt_token)


def observe_terminal(
    attempt_id: str,
    context: RouteContext,
    selected: Decision,
    *,
    engine: str,
) -> None:
    observer: Final = _OBSERVER.get()
    if observer is None:
        return
    request: Final = _REQUEST.get()
    current: Final = _CURRENT_ATTEMPT.get()
    observer.record(
        RouteObservation(
            event="terminal",
            trace_id=None if request is None else request.trace_id,
            attempt_id=attempt_id,
            parent_attempt_id=(
                current[2] if current is not None else None if request is None else request.parent_attempt_id
            ),
            depth=current[1] if current is not None else 0 if request is None else request.depth,
            route=context.route,
            context=context,
            decision=selected,
            outcome="success",
            engine=engine,
        )
    )


def observe_request_outcome(trace_id: str, outcome: str) -> None:
    observer: Final = _OBSERVER.get()
    if observer is None:
        return
    observer.record(
        RouteObservation(
            event="request",
            trace_id=trace_id,
            attempt_id=None,
            parent_attempt_id=None,
            depth=0,
            route=None,
            outcome=outcome,
        )
    )
