import bisect
from dataclasses import dataclass
from typing import Final

LATENCY_BOUNDS_MS: Final = (50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0, 120000.0)
BLOCK_COUNT_BOUNDS: Final = (1.0, 5.0, 20.0, 100.0)
ATTEMPT_BOUNDS: Final = (0.0, 1.0, 2.0, 3.0)


@dataclass(frozen=True, slots=True)
class Histogram:
    """Fixed-bucket counts: ``counts[i]`` holds values ``<= bounds[i]``, the last slot everything above"""

    bounds: tuple[float, ...]
    counts: tuple[int, ...]

    @classmethod
    def empty(cls, bounds: tuple[float, ...]) -> "Histogram":
        return cls(bounds=bounds, counts=(0,) * (len(bounds) + 1))

    @classmethod
    def of(cls, bounds: tuple[float, ...], value: float | None) -> "Histogram":
        empty: Final = cls.empty(bounds)
        if value is None:
            return empty
        index: Final = bisect.bisect_left(bounds, value)
        return cls(bounds=bounds, counts=tuple(int(slot == index) for slot in range(len(empty.counts))))

    def merge(self, other: "Histogram") -> "Histogram":
        return Histogram(bounds=self.bounds, counts=tuple(a + b for a, b in zip(self.counts, other.counts)))
