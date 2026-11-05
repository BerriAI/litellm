from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

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


@dataclass(frozen=True, slots=True, order=True)
class RateLimitCounter:
    scope: RateLimitScope
    kind: CounterKind

    @property
    def window(self) -> RateLimitWindow:
        return RateLimitWindow(self.scope)


@dataclass(frozen=True, slots=True)
class RateLimitGauge:
    scope: RateLimitScope


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


@dataclass(frozen=True, slots=True)
class WindowCount:
    window_start: str
    count: int


@dataclass(frozen=True, slots=True)
class AtomicApplied:
    """Per counter, the count after the increment and the start of the window it landed in."""

    counters: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class AtomicOverLimit:
    index: int
    current: int
    limit: int


AtomicReply: TypeAlias = AtomicApplied | AtomicOverLimit


@dataclass(frozen=True, slots=True)
class GaugesAcquired:
    in_flight: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GaugeOverLimit:
    index: int
    in_flight: int
    limit: int


AcquireReply: TypeAlias = GaugesAcquired | GaugeOverLimit


@dataclass(frozen=True, slots=True)
class GuardedReply:
    applied: bool
    count: int


NativeScope: TypeAlias = tuple[str, str]
NativeCounter: TypeAlias = tuple[str, str, CounterKind]
NativeEntry: TypeAlias = tuple[Literal["window", "requests", "tokens", "max_parallel_requests"], str, str]
NativeIncrement: TypeAlias = tuple[NativeCounter, int | float, int | None]
NativeAtomicCounter: TypeAlias = tuple[NativeCounter, int, int, int, int]


NativeOverLimit: TypeAlias = tuple[int, int, int]


def native_scope(scope: RateLimitScope) -> NativeScope:
    return (scope.descriptor_key, scope.descriptor_value)


def native_counter(counter: RateLimitCounter) -> NativeCounter:
    return (counter.scope.descriptor_key, counter.scope.descriptor_value, counter.kind)


def native_entry(entry: RateLimitEntry) -> NativeEntry:
    match entry:
        case RateLimitWindow(scope):
            return ("window", scope.descriptor_key, scope.descriptor_value)
        case RateLimitGauge(scope):
            return ("max_parallel_requests", scope.descriptor_key, scope.descriptor_value)
        case RateLimitCounter(scope, kind):
            return (kind, scope.descriptor_key, scope.descriptor_value)


def native_increment(increment: CounterIncrement) -> NativeIncrement:
    return (native_counter(increment.counter), increment.amount, increment.ttl)


def native_atomic(counter: AtomicCounter) -> NativeAtomicCounter:
    return (native_counter(counter.counter), counter.limit, counter.increment, counter.ttl, counter.window_size)
