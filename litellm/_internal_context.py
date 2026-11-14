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
from typing import TYPE_CHECKING, Final, ParamSpec, TypeVar, cast

if TYPE_CHECKING:
    from opentelemetry.trace import Span
else:
    Span: Final = object

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

# When True, suppresses async logging and billing for internal sub-calls
# (e.g., emulated file-search steps that make nested LLM calls).
is_internal_call: Final[ContextVar[bool]] = ContextVar("is_internal_call", default=False)
internal_parent_request_id: Final[ContextVar[str | None]] = ContextVar("internal_parent_request_id", default=None)

# One request prices its totals, its per-token-type lines and the rates it reports on
# separate code paths. Each reads the clock for off-peak pricing, so without a pinned
# moment they can land on either side of a window boundary and disagree with each other.
_billing_time: Final[ContextVar[datetime | None]] = ContextVar("billing_time", default=None)

_post_response: Final[ContextVar[bool]] = ContextVar("post_response", default=False)

_service_target: Final[ContextVar[str | None]] = ContextVar("service_target", default=None)

_mirrored_call: Final[ContextVar[bool]] = ContextVar("mirrored_call", default=False)

_resent_on: Final[ContextVar[Callable[[Exception], bool] | None]] = ContextVar("resent_on", default=None)

_request_parent_otel_span: Final[ContextVar[Span | None]] = ContextVar("request_parent_otel_span", default=None)


@contextmanager
def internal_sub_call(parent_request_id: str | None = None) -> Generator[None]:
    """A nested call the parent call logs and bills, so the parent must fold this call's cost into its own."""
    token: Final = is_internal_call.set(True)
    parent_token: Final = internal_parent_request_id.set(parent_request_id or internal_parent_request_id.get())
    try:
        yield
    finally:
        is_internal_call.reset(token)
        internal_parent_request_id.reset(parent_token)


@contextmanager
def billed_separately() -> Generator[None]:
    """A call that writes its own spend row even inside a sub-call whose parent bills the model response."""
    token: Final = is_internal_call.set(False)
    parent_token: Final = internal_parent_request_id.set(None)
    try:
        yield
    finally:
        is_internal_call.reset(token)
        internal_parent_request_id.reset(parent_token)


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


def with_service_target(
    target: str, *, preserve_existing: bool = False
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Run calls under the target, or the inherited target when ``preserve_existing`` is true."""

    def call_target() -> str:
        if preserve_existing:
            return current_service_target() or target
        return target

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if inspect.iscoroutinefunction(fn):
            awaitable_fn: Final[Callable[_P, Awaitable[object]]] = cast(  # cast-ok: checked by iscoroutinefunction
                "Callable[_P, Awaitable[object]]", fn
            )

            @wraps(fn)
            async def run_async(*args: _P.args, **kwargs: _P.kwargs) -> object:
                with service_target(call_target()):
                    return await awaitable_fn(*args, **kwargs)

            return cast("Callable[_P, _R]", run_async)  # cast-ok: same coroutine-returning signature as ``fn``

        @wraps(fn)
        def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with service_target(call_target()):
                return fn(*args, **kwargs)

        return run

    return decorate


@contextmanager
def mirrored_call() -> Generator[None]:
    """A silent_model mirror of a request: every litellm call inside, its retries and fallbacks included."""
    token: Final = _mirrored_call.set(True)
    try:
        yield
    finally:
        _restore(_mirrored_call, token)


def in_mirrored_call() -> bool:
    return _mirrored_call.get()


@contextmanager
def resent_on(rejection: Callable[[Exception], bool]) -> Generator[None]:
    """An attempt its caller sends again when it fails with ``rejection``. That failure is not the request's
    outcome, so it is neither logged as the request's failure nor retried, and its budget reservation and rate-limit
    slot stay with the attempt sent again."""
    token: Final = _resent_on.set(rejection)
    try:
        yield
    finally:
        _restore(_resent_on, token)


def is_resent_failure(error: Exception) -> bool:
    rejection: Final = _resent_on.get()
    return rejection is not None and rejection(error)


def set_request_parent_otel_span(span: Span) -> None:
    """Anchor the proxy request's OTEL span for datastore service spans that no call site passes a parent to.
    Never reset: it dies with the request task and rides the tasks it spawns, like ``request.state``."""
    _request_parent_otel_span.set(span)


def current_request_parent_otel_span() -> Span | None:
    return _request_parent_otel_span.get()


@contextmanager
def request_parent_otel_span(span: Span | None) -> Generator[None]:
    """Anchor the datastore service spans made inside this block, and by the tasks it starts, under the span an SDK
    caller passed. A request that already anchored its own, as the proxy does at auth, keeps it."""
    if span is None or _request_parent_otel_span.get() is not None:
        yield
        return
    token: Final = _request_parent_otel_span.set(span)
    try:
        yield
    finally:
        _restore(_request_parent_otel_span, token)


@contextmanager
def outside_mirrored_call() -> Generator[None]:
    """Logging a mirrored call: the calls its callbacks start are not mirrors themselves."""
    token: Final = _mirrored_call.set(False)
    try:
        yield
    finally:
        _restore(_mirrored_call, token)


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
