"""Deadline-based waits for tests, so nothing has to guess how long a background callback takes."""

import asyncio
import time
from collections.abc import Callable
from typing import Final

DEFAULT_TIMEOUT_S: Final[float] = 10.0
DEFAULT_INTERVAL_S: Final[float] = 0.02


def _fail(timeout_s: float, message: str) -> None:
    raise AssertionError(f"condition not met within {timeout_s}s: {message}")


def wait_until(
    predicate: Callable[[], bool],
    *,
    message: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> None:
    deadline: Final = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)  # sleep-ok: bounded poll interval, not a blind settle
    if not predicate():
        _fail(timeout_s, message)


async def await_until(
    predicate: Callable[[], bool],
    *,
    message: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> None:
    """Yields to the event loop between polls, so callbacks scheduled as tasks get a chance to run."""
    deadline: Final = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval_s)
    if not predicate():
        _fail(timeout_s, message)
