"""
One wall-clock budget for a whole non-streaming proxy request (``total_timeout``): the proxy opens it,
and the router reads it to refuse a retry backoff that cannot finish in time. A request that turns out
to stream or run in the background is not bounded: a body ``total_timeout`` refuses it, a header one
(often set globally on a client) lets it go unbounded.
"""

import asyncio
import contextlib
import math
import sys
from collections.abc import AsyncGenerator, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final

from litellm._logging import verbose_router_logger

if sys.version_info >= (3, 11):
    from asyncio import Timeout as _Scope
    from asyncio import timeout_at as _timeout_at

    _ScopeTimeout: Final = TimeoutError

    def _expired(scope: _Scope) -> bool:
        return scope.expired()

    def _disarm(scope: _Scope) -> None:
        scope.reschedule(None)

else:
    from async_timeout import Timeout as _Scope
    from async_timeout import timeout_at as _timeout_at

    _ScopeTimeout: Final = asyncio.TimeoutError

    def _expired(scope: _Scope) -> bool:
        return scope.expired

    def _disarm(scope: _Scope) -> None:
        scope.reject()


TOTAL_TIMEOUT_HEADER: Final = "x-litellm-total-timeout"
TOTAL_TIMEOUT_FIELD: Final = "total_timeout"


@dataclass(frozen=True, slots=True)
class TotalTimeout:
    seconds: float
    from_body: bool


@dataclass(frozen=True, slots=True)
class InvalidTotalTimeout:
    message: str


def _positive_finite_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        seconds: Final = float(value)
    except (ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _total_timeout(value: object, *, from_body: bool) -> TotalTimeout | InvalidTotalTimeout:
    seconds: Final = _positive_finite_seconds(value)
    if seconds is None:
        name: Final = TOTAL_TIMEOUT_FIELD if from_body else TOTAL_TIMEOUT_HEADER
        return InvalidTotalTimeout(f"{name} must be a positive, finite number of seconds, got {value!r}")
    return TotalTimeout(seconds, from_body)


def total_timeout_from_request(
    header: str | None, body: Mapping[str, object]
) -> TotalTimeout | InvalidTotalTimeout | None:
    """The request's ``total_timeout``: the header wins over the body field."""
    if header is not None:
        return _total_timeout(header, from_body=False)
    if TOTAL_TIMEOUT_FIELD not in body:
        return None
    return _total_timeout(body[TOTAL_TIMEOUT_FIELD], from_body=True)


def bounded_by_total_timeout(call: Mapping[str, object]) -> bool:
    """Only a request that neither streams nor runs in the background can be bounded."""
    return call.get("stream") is not True and call.get("background") is not True


class TotalTimeoutNotApplicable(Exception):
    """A request bounded by a body ``total_timeout`` streams or runs in the background."""


class TotalTimeoutExpired(TimeoutError):
    """The request's ``total_timeout`` ran out."""


class TotalDeadline:
    __slots__ = ("_at", "_clock", "_released", "_scope", "from_body")

    def __init__(self, at: float, clock: Callable[[], float], scope: _Scope, *, from_body: bool) -> None:
        self._at: Final = at
        self._clock: Final = clock
        self._scope: Final = scope
        self._released: bool = False
        self.from_body: Final = from_body

    def expired(self) -> bool:
        return _expired(self._scope)

    def allows_wait(self, seconds: float) -> bool:
        return self._released or self._clock() + seconds < self._at

    def release(self) -> None:
        self._released = True
        if not self.expired():
            _disarm(self._scope)


_CURRENT: Final[ContextVar[TotalDeadline | None]] = ContextVar("total_deadline", default=None)


def current_total_deadline() -> TotalDeadline | None:
    return _CURRENT.get()


def enforce_total_timeout_scope(call: Mapping[str, object]) -> None:
    """Raise ``TotalTimeoutNotApplicable`` for a body ``total_timeout`` on a streaming or background call,
    and stop bounding it for a header one."""
    deadline: Final = _CURRENT.get()
    if deadline is None or deadline._released or bounded_by_total_timeout(call):  # pyright: ignore[reportPrivateUsage]  # same module
        return
    if deadline.from_body:
        raise TotalTimeoutNotApplicable(
            "total_timeout applies only to non-streaming, non-background requests; drop it or the stream/background flag"
        )
    verbose_router_logger.debug("x-litellm-total-timeout ignored: the request streams or runs in the background")
    deadline.release()


@contextlib.asynccontextmanager
async def total_deadline(
    seconds: float, *, from_body: bool, clock: Callable[[], float] | None = None
) -> AsyncGenerator[TotalDeadline, None]:
    """Bound the block by ``seconds``: past it the block is cancelled and ``TotalTimeoutExpired`` raised,
    while a ``TimeoutError`` raised inside the block passes through as it was."""
    loop: Final = asyncio.get_running_loop()
    now: Final = clock if clock is not None else loop.time
    scope: Final = _timeout_at(loop.time() + seconds)
    deadline: Final = TotalDeadline(now() + seconds, now, scope, from_body=from_body)
    try:
        async with scope:
            token: Final = _CURRENT.set(deadline)
            try:
                yield deadline
            finally:
                _CURRENT.reset(token)
    except _ScopeTimeout as error:
        if not deadline.expired():
            raise
        raise TotalTimeoutExpired(f"the request did not finish within its total_timeout of {seconds}s") from error
