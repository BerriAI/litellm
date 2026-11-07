import asyncio
from collections.abc import Callable, Coroutine
from typing import Final, TypeVar

_ResultT = TypeVar("_ResultT")


class SkippableClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock a test moves forward, so timers fire without waiting for them."""

    skipped_seconds: float = 0.0

    def time(self) -> float:
        return super().time() + self.skipped_seconds

    def skip(self, seconds: float) -> None:
        self.skipped_seconds += seconds


def run_on_skippable_clock(scenario: Callable[[SkippableClockLoop], Coroutine[object, object, _ResultT]]) -> _ResultT:
    loop: Final = SkippableClockLoop()
    try:
        return loop.run_until_complete(scenario(loop))
    finally:
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()
