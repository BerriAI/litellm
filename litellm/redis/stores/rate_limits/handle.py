from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Protocol

from litellm.caching.redis_batch import NativePending
from litellm.redis.stores.rate_limits.types import (
    AcquireReply,
    AtomicCounter,
    AtomicReply,
    CounterIncrement,
    GuardedReply,
    LocalValue,
    RateLimitCounter,
    RateLimitEntry,
    RateLimitScope,
    WindowCount,
)
from litellm.rust_bridge._native import NativeRequestRedis, NativeSend


class RateLimitsHandle(Protocol):
    @property
    def has_redis(self) -> bool: ...
    def slot_groups(self, scopes: Sequence[RateLimitScope], force: bool) -> list[list[int]]: ...
    def get_local(self, entry: RateLimitEntry) -> LocalValue | None: ...
    def set_local(self, entry: RateLimitEntry, value: LocalValue, ttl_seconds: float | None) -> None: ...
    def get(
        self, entry: RateLimitEntry, *, backfill_ttl_seconds: float | None = None
    ) -> Awaitable[LocalValue | None]: ...
    def sliding_window(
        self, counters: Sequence[RateLimitCounter], now: int, window_size: int, send: NativeSend
    ) -> NativePending[list[WindowCount]] | None: ...
    def read_counters(
        self, counters: Sequence[RateLimitCounter], send: NativeSend
    ) -> NativePending[list[str | None]] | None: ...
    def check_and_increment(
        self, counters: Sequence[AtomicCounter], send: NativeSend
    ) -> NativePending[AtomicReply] | None: ...
    def guarded_increment(
        self,
        counter: RateLimitCounter,
        expected_window_start: str,
        amount: float,
        ttl_seconds: int | None,
        send: NativeSend,
    ) -> NativePending[GuardedReply] | None: ...
    def parallel_acquire(
        self, gauges: Sequence[tuple[RateLimitScope, int]], slot_ttl_seconds: int, slot_id: str, send: NativeSend
    ) -> NativePending[AcquireReply] | None: ...
    def parallel_release(
        self, gauges: Sequence[RateLimitScope], slot_id: str, send: NativeSend
    ) -> NativePending[list[int]] | None: ...
    def parallel_count(
        self, gauges: Sequence[RateLimitScope], slot_ttl_seconds: int, send: NativeSend
    ) -> NativePending[list[int]] | None: ...
    def token_increment(
        self, increments: Sequence[CounterIncrement], send: NativeSend
    ) -> NativePending[list[float]] | None: ...
    def token_increment_post_call(
        self, request: NativeRequestRedis, increments: Sequence[CounterIncrement]
    ) -> bool: ...
    def parallel_release_post_call(
        self, request: NativeRequestRedis, gauges: Sequence[RateLimitScope], slot_id: str
    ) -> bool: ...
    def increment(self, increment: CounterIncrement) -> Awaitable[int | float]: ...
    def increment_many(self, increments: Sequence[CounterIncrement]) -> Awaitable[None]: ...
    def increment_many_post_call(self, request: NativeRequestRedis, increments: Sequence[CounterIncrement]) -> None: ...
