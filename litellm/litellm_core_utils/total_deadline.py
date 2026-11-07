"""
One wall-clock budget for a whole non-streaming proxy request (``total_timeout``): the proxy opens it,
and the router reads it to refuse a retry backoff that cannot finish in time. A request that turns out
to stream or run in the background is not bounded: a body ``total_timeout`` refuses it, a header one
(often set globally on a client) lets it go unbounded.
"""

import asyncio
import contextlib
import sys
from collections.abc import AsyncGenerator, Callable, Mapping
from contextvars import ContextVar
from typing import Final

from litellm._logging import verbose_router_logger

if sys.version_info >= (3, 11):
    from asyncio import Timeout as _Scope
    from asyncio import timeout_at as _timeout_at

    _ScopeTimeout = TimeoutError

    def _expired(scope: _Scope) -> bool:
        return scope.expired()

    def _disarm(scope: _Scope) -> None:
        scope.reschedule(None)

else:
    from async_timeout import Timeout as _Scope
    from async_timeout import timeout_at as _timeout_at

    _ScopeTimeout = asyncio.TimeoutError

    def _expired(scope: _Scope) -> bool:
        return scope.expired

    def _disarm(scope: _Scope) -> None:
        scope.reject()


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


def _excluded(call: Mapping[str, object]) -> bool:
    return call.get("stream") is True or call.get("background") is True


def enforce_total_timeout_scope(call: Mapping[str, object]) -> None:
    """Raise ``TotalTimeoutNotApplicable`` for a body ``total_timeout`` on a streaming or background call,
    and stop bounding it for a header one."""
    deadline: Final = _CURRENT.get()
    if deadline is None or deadline._released or not _excluded(call):  # pyright: ignore[reportPrivateUsage]  # same module
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
