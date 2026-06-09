"""Three-tier cache facade. The only thing that speaks to L1/L2/L3.

Per R3 it exposes two distinct read intents so a policy never accidentally
admits off a stale snapshot:

- :meth:`admit_via_l2` always takes the L2 atomic write path (budgets, concurrency).
- :meth:`read_for_header` serves L1-with-staleness for header hints and read-only
  display endpoints, where bounded staleness is acceptable.

High-cardinality subjects bypass L1 (the caller passes ``bypass_l1``) so a team
with 100k end-users does not churn the per-pod LRU.
"""

from typing import Protocol

from litellm.integrations.governor.plumbing.clock import Clock
from litellm.integrations.governor.plumbing.inmemory import BoundedLRUCounterCache
from litellm.integrations.governor.plumbing.postgres import PostgresCounterStore
from litellm.integrations.governor.plumbing.redis import (
    CheckIncrementResult,
    GcraResult,
    L2Store,
    ReconcileResult,
)


class CounterStore(Protocol):
    """What a policy is allowed to call. A fake implementing this is enough to
    unit-test a policy without any cache tier (R9)."""

    async def admit_via_l2(
        self,
        counter_key: str,
        window_key: str | None,
        *,
        limit: float,
        increment: float,
        window_period_s: int,
        ttl_s: int,
        bypass_l1: bool = False,
    ) -> CheckIncrementResult: ...

    async def read_for_header(
        self, counter_key: str, *, staleness_ms: int, bypass_l1: bool = False
    ) -> float | None: ...

    async def reconcile_delta(
        self, reconciled_key: str, counter_key: str, *, delta: float, dedup_ttl_s: int
    ) -> ReconcileResult: ...

    async def gcra_admit(
        self, gcra_key: str, *, period_s: int, capacity: int, burst: int, cost: int
    ) -> GcraResult: ...

    async def assert_safe_eviction_policy(self, *, has_fail_closed: bool) -> str: ...


class ThreeTierCache:
    def __init__(
        self,
        l2: L2Store,
        l1: BoundedLRUCounterCache,
        l3: PostgresCounterStore,
        clock: Clock,
    ) -> None:
        self._l2 = l2
        self._l1 = l1
        self._l3 = l3
        self._clock = clock

    async def admit_via_l2(
        self,
        counter_key: str,
        window_key: str | None,
        *,
        limit: float,
        increment: float,
        window_period_s: int,
        ttl_s: int,
        bypass_l1: bool = False,
    ) -> CheckIncrementResult:
        result = await self._l2.check_and_increment(
            counter_key,
            window_key,
            limit=limit,
            increment=increment,
            window_period_s=window_period_s,
            ttl_s=ttl_s,
        )
        if result.admitted and not bypass_l1:
            await self._l1.set(counter_key, result.value, self._clock.monotonic_s())
        return result

    async def read_for_header(
        self, counter_key: str, *, staleness_ms: int, bypass_l1: bool = False
    ) -> float | None:
        now = self._clock.monotonic_s()
        if not bypass_l1:
            entry = await self._l1.get(counter_key)
            if (
                entry is not None
                and (now - entry.read_at_monotonic_s) * 1000 <= staleness_ms
            ):
                return entry.value
        value = await self._l2.read_value(counter_key)
        if value is not None and not bypass_l1:
            await self._l1.set(counter_key, value, now)
        return value

    async def reconcile_delta(
        self, reconciled_key: str, counter_key: str, *, delta: float, dedup_ttl_s: int
    ) -> ReconcileResult:
        result = await self._l2.reconcile_actual(
            reconciled_key, counter_key, delta=delta, dedup_ttl_s=dedup_ttl_s
        )
        if result.applied and result.value is not None:
            await self._l1.set(counter_key, result.value, self._clock.monotonic_s())
        return result

    async def gcra_admit(
        self, gcra_key: str, *, period_s: int, capacity: int, burst: int, cost: int
    ) -> GcraResult:
        return await self._l2.gcra_admit(
            gcra_key, period_s=period_s, capacity=capacity, burst=burst, cost=cost
        )

    async def assert_safe_eviction_policy(self, *, has_fail_closed: bool) -> str:
        return await self._l2.assert_safe_eviction_policy(
            has_fail_closed=has_fail_closed
        )
