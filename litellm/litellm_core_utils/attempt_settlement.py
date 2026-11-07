"""
What a routed attempt holds (io-token reservations, least-busy in-flight counts) is released exactly
once, by whichever of the attempt itself or its logging callbacks gets there first, and the release
runs to completion even when the request is cancelled mid-release.
"""

import asyncio
import contextlib
import sys
import uuid
from collections.abc import Coroutine, Generator, Sequence
from contextvars import ContextVar
from typing import Final

from litellm._logging import verbose_router_logger

ATTEMPT_SETTLEMENT_GRACE_SECONDS: Final = 5.0


class RoutedAttempt:
    __slots__ = ("answered", "attempt_id")

    def __init__(self) -> None:
        self.attempt_id: Final = uuid.uuid4().hex
        self.answered: bool = False


_ATTEMPT: Final[ContextVar[RoutedAttempt | None]] = ContextVar("routed_attempt", default=None)
_RELEASING: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: a release leaves it when it finishes


def current_attempt_id() -> str | None:
    attempt: Final = _ATTEMPT.get()
    return None if attempt is None else attempt.attempt_id


def mark_attempt_answered() -> None:
    """The provider answered the current routed attempt: what it reserved was spent, not just held."""
    attempt: Final = _ATTEMPT.get()
    if attempt is not None:
        attempt.answered = True


@contextlib.contextmanager
def routed_attempt() -> Generator[RoutedAttempt, None, None]:
    attempt: Final = RoutedAttempt()
    token: Final = _ATTEMPT.set(attempt)
    try:
        yield attempt
    finally:
        _ATTEMPT.reset(token)


def _finished(task: "asyncio.Task[None]") -> None:
    _RELEASING.discard(task)
    if task.cancelled():
        verbose_router_logger.warning("an attempt's release was cancelled before it finished")
        return
    error: Final = task.exception()
    if error is not None:
        verbose_router_logger.warning("an attempt's release failed: %s", error)


def _start(release: Coroutine[object, object, None]) -> "asyncio.Task[None]":
    """Run the release up to its first real wait right away, so one that never waits (an in-memory
    counter) is done before its caller goes on, as it was when the caller awaited it inline."""
    loop: Final = asyncio.get_running_loop()
    if sys.version_info >= (3, 12):
        return asyncio.Task(release, loop=loop, eager_start=True)
    return loop.create_task(release)


def release_in_background(release: Coroutine[object, object, None]) -> "asyncio.Task[None]":
    task: Final = _start(release)
    _RELEASING.add(task)
    task.add_done_callback(_finished)
    return task


async def released(releases: Sequence["asyncio.Task[None]"], grace: float | None = None) -> None:
    """Wait for ``releases`` without ever cancelling them: a cancelled waiter leaves them running."""
    if all(release.done() for release in releases):
        return
    _, pending = await asyncio.wait(releases, timeout=grace)
    if pending:
        verbose_router_logger.warning(
            "%s attempt release(s) still running after %ss; they keep running until they finish",
            len(pending),
            grace,
        )
