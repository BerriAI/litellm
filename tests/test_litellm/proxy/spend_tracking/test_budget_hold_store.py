"""Unit tests for BudgetHoldStore (in-memory backend).

The store is the heart of the orphan-proof budget design: in-flight
reservations are tracked as individually expiring holds, summed at admission,
and never folded into the committed spend counter. The key invariant is that an
orphaned hold (pod died before settle) is pruned once its TTL elapses, so it
cannot pin a budget counter forever.
"""

import time

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.spend_tracking.budget_hold_store import BudgetHoldStore


def _store(ttl_seconds: int = 3600) -> BudgetHoldStore:
    # No Redis configured -> in-memory backend.
    return BudgetHoldStore(dual_cache=DualCache(), ttl_seconds=ttl_seconds)


@pytest.mark.asyncio
async def test_place_and_total_sums_live_holds():
    store = _store()
    assert await store.place_and_total("spend:key:k", "h1", 0.4) == pytest.approx(0.4)
    assert await store.place_and_total("spend:key:k", "h2", 0.3) == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_remove_drops_only_that_hold():
    store = _store()
    await store.place_and_total("spend:key:k", "h1", 0.4)
    await store.place_and_total("spend:key:k", "h2", 0.3)

    await store.remove("spend:key:k", "h1", 0.4)

    # Only h2 remains; admitting a new hold sums 0.3 + 0.1.
    assert await store.place_and_total("spend:key:k", "h3", 0.1) == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_resize_updates_hold_cost():
    store = _store()
    await store.place_and_total("spend:key:k", "h1", 0.6)

    await store.resize("spend:key:k", "h1", 0.6, 0.4)

    assert await store.place_and_total("spend:key:k", "h2", 0.1) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_orphaned_hold_is_pruned_after_ttl():
    """A hold that is never removed (its pod was killed before settle) must be
    pruned at the next admission once its TTL has elapsed, so it stops counting
    against the budget."""
    store = _store()
    await store.place_and_total("spend:key:k", "orphan", 0.5)

    # Simulate the orphan's lifetime having elapsed.
    cost, _ = store._memory_holds["spend:key:k"]["orphan"]
    store._memory_holds["spend:key:k"]["orphan"] = (cost, time.time() - 1)

    # The next admission prunes the expired orphan; only the fresh hold counts.
    assert await store.place_and_total("spend:key:k", "fresh", 0.2) == pytest.approx(
        0.2
    )


@pytest.mark.asyncio
async def test_holds_are_isolated_per_counter():
    store = _store()
    await store.place_and_total("spend:key:k1", "h1", 0.4)
    assert await store.place_and_total("spend:key:k2", "h2", 0.3) == pytest.approx(0.3)
