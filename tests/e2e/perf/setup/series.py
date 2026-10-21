"""Readings a background sampler takes while the traffic runs, each stamped with the wall clock
(`time.time()`, the clock the Locust records carry in `started_at`), so a ramped run can cut them into its
steps' windows after the fact."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Sample(Generic[T]):
    at: float
    value: T


@dataclass(frozen=True, slots=True)
class Series(Generic[T]):
    samples: tuple[Sample[T], ...]

    def between(self, start: float, end: float) -> tuple[T, ...]:
        """The values stamped inside `[start, end)`, in time order."""
        return tuple(sample.value for sample in self.samples if start <= sample.at < end)

    def last_before(self, at: float) -> T | None:
        """The newest value stamped before `at`, None when the series starts later."""
        earlier: Final = tuple(sample.value for sample in self.samples if sample.at < at)
        return earlier[-1] if earlier else None


class Recorder(Generic[T]):
    """What a sampler thread appends to; `series` is a frozen copy for the observation."""

    def __init__(self) -> None:
        self._samples: list[Sample[T]] = []

    def record(self, value: T) -> None:
        self._samples.append(Sample(at=time.time(), value=value))

    @property
    def series(self) -> Series[T]:
        return Series(samples=tuple(self._samples))

    def __len__(self) -> int:
        return len(self._samples)
