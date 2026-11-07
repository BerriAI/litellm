"""
What a routed attempt holds (io-token reservations, least-busy in-flight counts) is released exactly
once, by whichever of the attempt itself or its logging callbacks gets there first, and the release
runs to completion even when the request is cancelled mid-release.
"""

import asyncio
import contextlib
import uuid
from collections.abc import Coroutine, Generator, Sequence
from contextvars import ContextVar
from typing import Final

from litellm._logging import verbose_router_logger

ATTEMPT_SETTLEMENT_GRACE_SECONDS: Final = 5.0
_ATTEMPT_ID: Final[ContextVar[str | None]] = ContextVar("routed_attempt_id", default=None)
_RELEASING: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: a release leaves it when it finishes


def current_attempt_id() -> str | None:
    return _ATTEMPT_ID.get()


@contextlib.contextmanager
def routed_attempt() -> Generator[str, None, None]:
    attempt_id: Final = uuid.uuid4().hex
    token: Final = _ATTEMPT_ID.set(attempt_id)
    try:
        yield attempt_id
    finally:
        _ATTEMPT_ID.reset(token)


def _finished(task: "asyncio.Task[None]") -> None:
    _RELEASING.discard(task)
    if task.cancelled():
        verbose_router_logger.warning("an attempt's release was cancelled before it finished")
        return
    error: Final = task.exception()
    if error is not None:
        verbose_router_logger.warning("an attempt's release failed: %s", error)


def release_in_background(release: Coroutine[object, object, None]) -> "asyncio.Task[None]":
    task: Final = asyncio.get_running_loop().create_task(release)
    _RELEASING.add(task)
    task.add_done_callback(_finished)
    return task


async def released(releases: Sequence["asyncio.Task[None]"], grace: float | None = None) -> None:
    """Wait for ``releases`` without ever cancelling them: a cancelled waiter leaves them running."""
    if not releases:
        return
    _, pending = await asyncio.wait(releases, timeout=grace)
    if pending:
        verbose_router_logger.warning(
            "%s attempt release(s) still running after %ss; they keep running until they finish",
            len(pending),
            grace,
        )
