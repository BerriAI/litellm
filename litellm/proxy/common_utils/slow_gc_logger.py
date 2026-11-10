"""Garbage collection passes long enough to count as event-loop holds, logged next to asyncio's own
`Executing <Task ...> took X seconds` lines so a reader can tell a collector pause from code that held the loop
(asyncio charges the pause to whichever task was running). Registered only when the loop runs in debug mode."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Mapping
from typing import Final, Literal


class SlowGcLogger:
    def __init__(
        self, threshold_seconds: float, log: Callable[[str], None], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._threshold_seconds: Final = threshold_seconds
        self._log: Final = log
        self._clock: Final = clock
        self._started = clock()

    def __call__(self, phase: Literal["start", "stop"], info: Mapping[str, int]) -> None:
        if phase == "start":
            self._started = self._clock()
            return
        seconds: Final = self._clock() - self._started
        if seconds < self._threshold_seconds:
            return
        self._log(
            f"Garbage collection gen={info['generation']} collected={info['collected']} took {seconds:.3f} seconds"
        )


def log_slow_gc_passes(threshold_seconds: float, log: Callable[[str], None]) -> None:
    gc.callbacks.append(SlowGcLogger(threshold_seconds=threshold_seconds, log=log))
