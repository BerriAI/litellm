import bisect
import math
from dataclasses import dataclass
from typing import Final

LATENCY_BOUNDS_MS: Final = (50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0, 120000.0)
BLOCK_COUNT_BOUNDS: Final = (1.0, 5.0, 20.0, 100.0)
ATTEMPT_BOUNDS: Final = (0.0, 1.0, 2.0, 3.0)


@dataclass(slots=True)
class Histogram:
    """Fixed-bucket counts: ``counts[i]`` holds values ``<= bounds[i]``, the last slot everything above.
    Negative, NaN and infinite values are counted in ``invalid`` instead of a bucket"""

    bounds: tuple[float, ...]
    counts: list[int]  # mutable-ok: incremented in place once per record on the request path
    invalid: int = 0

    @classmethod
    def empty(cls, bounds: tuple[float, ...]) -> "Histogram":
        return cls(bounds=bounds, counts=[0] * (len(bounds) + 1))

    def add(self, value: float | None) -> None:
        if value is None:
            return
        if not math.isfinite(value) or value < 0:
            self.invalid += 1
            return
        self.counts[bisect.bisect_left(self.bounds, value)] += 1

    def add_histogram(self, other: "Histogram") -> None:
        for index, count in enumerate(other.counts):
            self.counts[index] += count
        self.invalid += other.invalid
