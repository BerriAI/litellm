from __future__ import annotations

import gc
import re
from itertools import count
from typing import Final

from litellm.proxy.common_utils.slow_gc_logger import SlowGcLogger


def _collect_once(step_seconds: float) -> tuple[str, ...]:
    ticks: Final = count(step=step_seconds)
    lines: Final[list[str]] = []  # mutable-ok: collects what the logger emits during one real gc.collect()
    logger: Final = SlowGcLogger(threshold_seconds=0.1, log=lines.append, clock=lambda: next(ticks))
    gc.callbacks.append(logger)
    try:
        gc.collect()
    finally:
        gc.callbacks.remove(logger)
    return tuple(lines)


def test_a_full_collection_past_the_threshold_is_logged_with_its_generation_and_length() -> None:
    lines: Final = _collect_once(step_seconds=0.25)
    assert any(re.fullmatch(r"Garbage collection gen=2 collected=\d+ took 0\.250 seconds", line) for line in lines), (
        lines
    )


def test_a_collection_under_the_threshold_is_not_logged() -> None:
    assert _collect_once(step_seconds=0.01) == ()
