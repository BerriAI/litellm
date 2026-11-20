from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Final


class GatewayProcessingTiming:
    def __init__(self, started: float) -> None:
        self.started: Final = started
        self._excluded: Final[list[tuple[float, float]]] = []  # mutable-ok: request-local timing intervals
        self._timed_attempts = 0
        self._incomplete = False

    def exclude(self, started: float, ended: float) -> None:
        self._excluded.append((started, ended))

    def complete_attempt(self, timed: bool) -> None:
        if timed:
            self._timed_attempts += 1
        else:
            self._incomplete = True

    def seconds(self, ended: float) -> float | None:
        if self._incomplete or not self._timed_attempts:
            return None
        excluded = 0.0  # rebind-ok: measure of the merged intervals
        covered_until = self.started  # rebind-ok: end of the intervals already covered
        for start, stop in sorted(self._excluded):
            lower, upper = max(start, covered_until), min(stop, ended)
            excluded += max(0.0, upper - lower)
            covered_until = max(upper, covered_until)
        return max(0.0, ended - self.started - excluded)


class _ProviderAttempt:
    def __init__(self, timing: GatewayProcessingTiming) -> None:
        self.timing: Final = timing
        self.timed = False


_gateway_timing: Final[ContextVar[GatewayProcessingTiming | None]] = ContextVar(
    "gateway_processing_timing", default=None
)
_provider_attempt: Final[ContextVar[_ProviderAttempt | None]] = ContextVar("gateway_processing_attempt", default=None)


@contextmanager
def gateway_processing_window(timing: GatewayProcessingTiming) -> Generator[None, None, None]:
    token: Final = _gateway_timing.set(timing)
    try:
        yield
    finally:
        _gateway_timing.reset(token)


@contextmanager
def track_gateway_provider_attempt() -> Generator[None, None, None]:
    timing: Final = _gateway_timing.get()
    if timing is None:
        yield
        return
    attempt: Final = _ProviderAttempt(timing)
    token: Final = _provider_attempt.set(attempt)
    try:
        yield
    finally:
        _provider_attempt.reset(token)
        timing.complete_attempt(attempt.timed)


@contextmanager
def exclude_gateway_provider_time() -> Generator[None, None, None]:
    attempt: Final = _provider_attempt.get()
    if attempt is None:
        yield
        return
    started: Final = perf_counter()
    try:
        yield
    finally:
        attempt.timed = True
        attempt.timing.exclude(started, perf_counter())


@contextmanager
def exclude_gateway_retry_wait() -> Generator[None, None, None]:
    timing: Final = _gateway_timing.get()
    if timing is None:
        yield
        return
    started: Final = perf_counter()
    try:
        yield
    finally:
        timing.exclude(started, perf_counter())
