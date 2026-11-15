from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, TypeAlias

from litellm.rust_bridge._native import (
    AtomicApplied,
    AtomicOverLimit,
    GaugeOverLimit,
    GaugesAcquired,
    GuardedReply,
    WindowCount,
)

RATE_LIMITS_TARGET: Final = "rate_limits"
CounterKind: TypeAlias = Literal["requests", "tokens"]
LocalValue: TypeAlias = int | float | str | dict[str, float]


@dataclass(frozen=True, slots=True, order=True)
class RateLimitScope:
    """A rate-limit descriptor; every key of its window shares the ``{key:value}`` hash tag."""

    descriptor_key: str
    descriptor_value: str

    def counter(self, kind: CounterKind) -> RateLimitCounter:
        return RateLimitCounter(self, kind)

    @property
    def window(self) -> RateLimitWindow:
        return RateLimitWindow(self)

    @property
    def gauge(self) -> RateLimitGauge:
        return RateLimitGauge(self)


@dataclass(frozen=True, slots=True)
class RateLimitWindow:
    scope: RateLimitScope
    suffix: ClassVar[str] = "window"


@dataclass(frozen=True, slots=True, order=True)
class RateLimitCounter:
    scope: RateLimitScope
    kind: CounterKind

    @property
    def window(self) -> RateLimitWindow:
        return RateLimitWindow(self.scope)

    @property
    def suffix(self) -> str:
        return self.kind


@dataclass(frozen=True, slots=True)
class RateLimitGauge:
    scope: RateLimitScope
    suffix: ClassVar[str] = "max_parallel_requests"


RateLimitEntry: TypeAlias = RateLimitWindow | RateLimitCounter | RateLimitGauge


@dataclass(frozen=True, slots=True)
class CounterIncrement:
    counter: RateLimitCounter
    amount: int | float
    ttl: int | None


@dataclass(frozen=True, slots=True)
class AtomicCounter:
    counter: RateLimitCounter
    limit: int
    increment: int
    ttl: int
    window_size: int


AtomicReply: TypeAlias = AtomicApplied | AtomicOverLimit
AcquireReply: TypeAlias = GaugesAcquired | GaugeOverLimit


__all__ = [
    "RATE_LIMITS_TARGET",
    "AcquireReply",
    "AtomicApplied",
    "AtomicCounter",
    "AtomicOverLimit",
    "AtomicReply",
    "CounterIncrement",
    "CounterKind",
    "GaugeOverLimit",
    "GaugesAcquired",
    "GuardedReply",
    "LocalValue",
    "RateLimitCounter",
    "RateLimitEntry",
    "RateLimitGauge",
    "RateLimitScope",
    "RateLimitWindow",
    "WindowCount",
]
