"""
Internal request context for LiteLLM.

Provides a ContextVar-based mechanism for internal signals that must not
be settable from user input. Context variables are scoped to the current
asyncio task and cannot be injected via HTTP request bodies.
"""

import inspect
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager, suppress
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from functools import wraps
from typing import Final, ParamSpec, TypeVar, cast

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

# When True, suppresses async logging and billing for internal sub-calls
# (e.g., emulated file-search steps that make nested LLM calls).
is_internal_call: Final[ContextVar[bool]] = ContextVar("is_internal_call", default=False)

# One request prices its totals, its per-token-type lines and the rates it reports on
# separate code paths. Each reads the clock for off-peak pricing, so without a pinned
# moment they can land on either side of a window boundary and disagree with each other.
_billing_time: Final[ContextVar[datetime | None]] = ContextVar("billing_time", default=None)

_post_response: Final[ContextVar[bool]] = ContextVar("post_response", default=False)

_service_target: Final[ContextVar[str | None]] = ContextVar("service_target", default=None)
# Event-metadata key under which a Redis pipeline reports the sorted, comma-joined families its ops were
# declared under when they span more than one.
REDIS_FAMILIES_METADATA_KEY: Final = "families"

_service_caller: Final[ContextVar[str | None]] = ContextVar("service_caller", default=None)


_emulated_file_search: Final[ContextVar[bool]] = ContextVar("emulated_file_search", default=False)


@contextmanager
def emulated_file_search_phase() -> Generator[None]:
    """Nested calls of emulated file_search, whose answer keeps only its own tool calls."""
    token: Final = _emulated_file_search.set(True)
    try:
        yield
    finally:
        _emulated_file_search.reset(token)


def in_emulated_file_search() -> bool:
    return _emulated_file_search.get()


@contextmanager
def post_response_phase() -> Generator[None]:
    """Work the caller no longer waits for (success callbacks, response-cache writes), including tasks it spawns."""
    token: Final = _post_response.set(True)
    try:
        yield
    finally:
        _post_response.reset(token)


def in_post_response_phase() -> bool:
    return _post_response.get()


def _restore(var: ContextVar[_T], token: Token[_T]) -> None:
    """Reset ``var``; a coroutine the GC closes from another context has no value left to restore."""
    with suppress(ValueError):
        var.reset(token)


@contextmanager
def service_target(target: str | None) -> Generator[None]:
    """Name what the datastore calls inside this block are for; ``None`` clears an inherited target."""
    token: Final = _service_target.set(target)
    try:
        yield
    finally:
        _restore(_service_target, token)


def current_service_target() -> str | None:
    return _service_target.get()


def with_service_target(target: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Run every call of the decorated function, coroutine functions included, under ``service_target(target)``."""

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if inspect.iscoroutinefunction(fn):
            awaitable_fn: Final[Callable[_P, Awaitable[object]]] = cast(  # cast-ok: checked by iscoroutinefunction
                "Callable[_P, Awaitable[object]]", fn
            )

            @wraps(fn)
            async def run_async(*args: _P.args, **kwargs: _P.kwargs) -> object:
                with service_target(target):
                    return await awaitable_fn(*args, **kwargs)

            return cast("Callable[_P, _R]", run_async)  # cast-ok: same coroutine-returning signature as ``fn``

        @wraps(fn)
        def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with service_target(target):
                return fn(*args, **kwargs)

        return run

    return decorate


@contextmanager
def service_caller(caller: str | None) -> Generator[None]:
    """Name the litellm code a datastore call was issued for when its own frames cannot: an operation
    declared in one task and run in another (a batch op retried on the flush) carries the chain captured
    where it was declared."""
    token: Final = _service_caller.set(caller)
    try:
        yield
    finally:
        _restore(_service_caller, token)


def current_service_caller() -> str | None:
    return _service_caller.get()


@contextmanager
def pinned_billing_time(moment: datetime) -> Generator[None]:
    """Price every rate lookup inside this block at ``moment`` rather than at each one's own clock read."""
    token: Final = _billing_time.set(moment)
    try:
        yield
    finally:
        _billing_time.reset(token)


def current_billing_time() -> datetime:
    """The pinned billing moment, or now in UTC outside a pinned block."""
    pinned: Final = _billing_time.get()
    return pinned if pinned is not None else datetime.now(timezone.utc)
