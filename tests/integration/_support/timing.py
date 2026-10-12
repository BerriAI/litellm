import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Timed[T]:
    value: T
    seconds: float


def timed[T](operation: Callable[[], T]) -> Timed[T]:
    started: Final = time.monotonic()
    value: Final = operation()
    return Timed(value, time.monotonic() - started)


def assert_not_delayed_by(observed: Iterable[float], *, control: Iterable[float], injected_delay: float) -> None:
    worst: Final = max(observed)
    baseline: Final = max(control)
    assert worst < baseline + injected_delay, (
        f"slowest call took {worst:.2f}s, at least the {baseline:.2f}s control plus the {injected_delay}s injected delay"
    )


def assert_cut_short(elapsed: float, *, injected_delay: float) -> None:
    assert elapsed < injected_delay, (
        f"returned after {elapsed:.2f}s, no sooner than the {injected_delay}s injected delay"
    )


def assert_within_factor_of_control(observed: float, *, control: float, factor: float) -> None:
    assert observed <= factor * control, f"took {observed:.2f}s, over {factor}x the {control:.2f}s control"
