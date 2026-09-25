"""Per-model, per-class fairness counters (minute buckets) backing ``GET /fairness/status``."""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm.proxy.utils import InternalUsageCache

FairnessMetric = Literal[
    "queued",
    "admitted_after_wait",
    "rejected_capacity",
    "rejected_queue_full",
    "rejected_deadline",
    "disconnected",
    "wait_seconds_sum",
]

FAIRNESS_METRICS: Final[tuple[FairnessMetric, ...]] = (
    "queued",
    "admitted_after_wait",
    "rejected_capacity",
    "rejected_queue_full",
    "rejected_deadline",
    "disconnected",
    "wait_seconds_sum",
)

STATS_WINDOW_SECONDS: Final = 300
_BUCKET_SECONDS: Final = 60
_BUCKET_TTL_SECONDS: Final = STATS_WINDOW_SECONDS + 2 * _BUCKET_SECONDS
_RAW_VALUES_ADAPTER: Final = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class ClassStats:
    queued: int
    admitted_after_wait: int
    rejected_capacity: int
    rejected_queue_full: int
    rejected_deadline: int
    disconnected: int
    wait_seconds_sum: float

    @property
    def avg_queue_wait_seconds(self) -> float:
        served: Final = self.admitted_after_wait + self.rejected_deadline + self.disconnected
        return self.wait_seconds_sum / served if served else 0.0


def _bucket_key(model: str, class_name: str, metric: FairnessMetric, bucket: int) -> str:
    return f"fairness_stats:{model}:{class_name}:{metric}:{bucket}"


def _as_float(raw: object) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, (str, bytes)):
        try:
            return float(raw)
        except ValueError:
            return 0.0
    return 0.0


class FairnessStats:
    def __init__(self, cache: InternalUsageCache, clock: Callable[[], float] = time.time) -> None:
        self._cache = cache
        self._clock = clock

    def _current_bucket(self) -> int:
        return int(self._clock()) // _BUCKET_SECONDS

    def _window_buckets(self) -> tuple[int, ...]:
        current: Final = self._current_bucket()
        return tuple(range(current - STATS_WINDOW_SECONDS // _BUCKET_SECONDS + 1, current + 1))

    async def record(self, model: str, class_name: str, metric: FairnessMetric, value: float = 1.0) -> None:
        await self._cache.async_increment_cache(  # pyright: ignore[reportUnknownMemberType]  # **kwargs untyped upstream
            key=_bucket_key(model, class_name, metric, self._current_bucket()),
            value=value,
            litellm_parent_otel_span=None,
            ttl=_BUCKET_TTL_SECONDS,
        )

    async def read(self, model: str, class_names: Sequence[str]) -> Mapping[str, ClassStats]:
        buckets: Final = self._window_buckets()
        keys: Final = tuple(
            _bucket_key(model, class_name, metric, bucket)
            for class_name, metric, bucket in product(class_names, FAIRNESS_METRICS, buckets)
        )
        raw_values: Final = _RAW_VALUES_ADAPTER.validate_python(
            await self._cache.async_batch_get_cache(keys=keys) or ()
        )
        values: Final = tuple(_as_float(raw) for raw in raw_values) + (0.0,) * (len(keys) - len(raw_values))
        per_class_span: Final = len(FAIRNESS_METRICS) * len(buckets)

        def total(class_index: int, metric_index: int) -> float:
            start: Final = class_index * per_class_span + metric_index * len(buckets)
            return sum(values[start : start + len(buckets)])

        return MappingProxyType(
            {
                class_name: ClassStats(
                    queued=int(total(class_index, 0)),
                    admitted_after_wait=int(total(class_index, 1)),
                    rejected_capacity=int(total(class_index, 2)),
                    rejected_queue_full=int(total(class_index, 3)),
                    rejected_deadline=int(total(class_index, 4)),
                    disconnected=int(total(class_index, 5)),
                    wait_seconds_sum=total(class_index, 6),
                )
                for class_index, class_name in enumerate(class_names)
            }
        )
