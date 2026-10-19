import asyncio
import selectors
from collections.abc import Callable
from typing import Final


class ManualClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock moves only when the test advances it, so timeouts fire on test-controlled conditions"""

    def __init__(self, selector: selectors.BaseSelector | None = None) -> None:
        super().__init__(selector)
        self._now = 0.0

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _AutojumpSelector(selectors.DefaultSelector):
    def __init__(self, advance: Callable[[float], None]) -> None:
        super().__init__()
        self._advance = advance

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        ready: Final = super().select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            return super().select(None)
        self._advance(timeout)
        return []


class AutojumpClockLoop(ManualClockLoop):
    """A manual-clock loop that jumps to the next timer only once no callback or I/O event is left to run"""

    def __init__(self) -> None:
        super().__init__(_AutojumpSelector(self.advance))
