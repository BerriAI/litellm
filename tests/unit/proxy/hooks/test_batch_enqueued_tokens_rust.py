"""
LIT-5273: enqueued-token accounting for batch submissions.

Covers the ``BatchEnqueuedTokenStore`` (reserve / refund / reservation
records), the metadata-driven scope resolution, and the batch-id and
response-shape helpers the v3 limiter's post-call hooks rely on.
"""

import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BatchEnqueuedTokenOverLimit,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    BatchEnqueuedTokenStore,
)
from litellm.proxy.utils import InternalUsageCache
from tests._support.rust_parallel_request_limiter import PROXY_MaxParallelRequestsHandler_v3


@pytest.fixture(autouse=True)
def _native_parallel_request_limiter(monkeypatch):
    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "true")


@pytest.fixture
def memory_store() -> BatchEnqueuedTokenStore:
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache(default_in_memory_ttl=60))
    )
    return limiter.batch_enqueued_token_store


def _scope(limit: int, key: str = "api_key") -> BatchEnqueuedTokenScope:
    return BatchEnqueuedTokenScope(key=key, value=f"{key}-{uuid.uuid4().hex}", limit=limit)


@pytest.mark.asyncio
async def test_reserve_rejects_once_allowance_is_exhausted(memory_store: BatchEnqueuedTokenStore):
    store = memory_store
    scope = _scope(limit=100)
    first = await store.reserve(tokens=80, scopes=(scope,))
    assert isinstance(first, BatchEnqueuedTokenReservation)
    second = await store.reserve(tokens=30, scopes=(scope,))
    assert second == BatchEnqueuedTokenOverLimit(scope=scope, enqueued=80)
    third = await store.reserve(tokens=20, scopes=(scope,))
    assert isinstance(third, BatchEnqueuedTokenReservation)


@pytest.mark.asyncio
async def test_reserve_is_all_or_nothing_across_scopes(memory_store: BatchEnqueuedTokenStore):
    store = memory_store
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")
    over = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
    assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)
    exact_fit = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(exact_fit, BatchEnqueuedTokenReservation)


@pytest.mark.asyncio
async def test_refund_restores_allowance_and_never_goes_negative(memory_store: BatchEnqueuedTokenStore):
    store = memory_store
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=30, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.refund(reservation)
    await store.refund(reservation)
    refill = await store.reserve(tokens=100, scopes=(scope,))
    assert isinstance(refill, BatchEnqueuedTokenReservation)
    assert isinstance(await store.reserve(tokens=1, scopes=(scope,)), BatchEnqueuedTokenOverLimit)


@pytest.mark.asyncio
async def test_reservation_record_roundtrip_pops_exactly_once(memory_store: BatchEnqueuedTokenStore):
    store = memory_store
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=40, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.save_reservation("batch_abc", reservation)
    assert await store.pop_reservation("batch_abc") == reservation
    assert await store.pop_reservation("batch_abc") is None
    assert await store.pop_reservation("batch_never_saved") is None


@pytest.mark.asyncio
async def test_zero_token_reserve_charges_nothing(memory_store: BatchEnqueuedTokenStore):
    store = memory_store
    scope = _scope(limit=100)
    empty = await store.reserve(tokens=0, scopes=(scope,))
    assert empty == BatchEnqueuedTokenReservation(tokens=0, scopes=(scope,))
    full = await store.reserve(tokens=100, scopes=(scope,))
    assert isinstance(full, BatchEnqueuedTokenReservation)


class _SingleKeyRedisFake:
    """Emulates the Redis script path one single-key call at a time, recording every call."""

    def __init__(
        self,
        fail_reserve_keys: frozenset[str] = frozenset(),
        fail_refund_keys: frozenset[str] = frozenset(),
        fail_save_keys: frozenset[str] = frozenset(),
        raise_after_landing_save_keys: frozenset[str] = frozenset(),
    ) -> None:
        self.script_calls: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self.save_ttls: tuple[int, ...] = ()
        self.counters: Mapping[str, int] = MappingProxyType({})
        self.records: Mapping[str, str] = MappingProxyType({})
        self.fail_reserve_keys = fail_reserve_keys
        self.fail_refund_keys = fail_refund_keys
        self.fail_save_keys = fail_save_keys
        self.raise_after_landing_save_keys = raise_after_landing_save_keys

    def async_register_script(self, script: str):
        kind: Final = (
            "reserve"
            if "INCRBY" in script
            else "refund"
            if "DECRBY" in script
            else "pop"
            if "GET" in script
            else "save"
        )

        async def run(keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> object:
            self.script_calls = (*self.script_calls, (kind, tuple(keys)))
            return self._run(kind, tuple(keys), tuple(args))

        return run

    def _run(self, kind: str, keys: tuple[str, ...], args: tuple[str | bytes | int | float, ...]) -> object:
        if kind == "reserve":
            if keys[0] in self.fail_reserve_keys:
                raise ConnectionError(f"simulated redis failure for {keys[0]}")
            amount, limit = int(args[0]), int(args[2])
            current: Final = self.counters.get(keys[0], 0)
            if current + amount > limit:
                return (0, current)
            self.counters = MappingProxyType({**self.counters, keys[0]: current + amount})
            return (1, current + amount)
        if kind == "refund":
            if keys[0] in self.fail_refund_keys:
                raise ConnectionError(f"simulated redis failure for {keys[0]}")
            remaining: Final = self.counters.get(keys[0], 0) - int(args[0])
            self.counters = MappingProxyType(
                {key: value for key, value in self.counters.items() if key != keys[0]}
                if remaining <= 0
                else {**self.counters, keys[0]: remaining}
            )
            return 1
        if kind == "save":
            if keys[0] in self.fail_save_keys:
                raise ConnectionError(f"simulated redis failure for {keys[0]}")
            self.records = MappingProxyType({**self.records, keys[0]: str(args[0])})
            self.save_ttls = (*self.save_ttls, int(args[1]))
            if keys[0] in self.raise_after_landing_save_keys:
                raise TimeoutError(f"simulated redis timeout after landing for {keys[0]}")
            return 1
        if kind == "pop":
            popped: Final = self.records.get(keys[0])
            if popped:
                self.records = MappingProxyType({**self.records, keys[0]: ""})
            return popped
        raise AssertionError(f"unexpected {kind} script call for keys {keys}")


@pytest.mark.asyncio
async def test_memory_refund_skips_reservations_granted_by_another_worker(memory_store: BatchEnqueuedTokenStore):
    store = memory_store
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert reservation.backend == "memory"
    assert reservation.owner

    foreign: Final = BatchEnqueuedTokenReservation(
        tokens=60, scopes=reservation.scopes, backend="memory", owner="another-worker"
    )
    await store.refund(foreign)
    assert await store.reserve(tokens=50, scopes=(scope,)) == BatchEnqueuedTokenOverLimit(scope=scope, enqueued=60)

    await store.refund(reservation)
    assert isinstance(await store.reserve(tokens=100, scopes=(scope,)), BatchEnqueuedTokenReservation)


class _OpenBreakerRedis:
    def async_register_script(self, script: str):
        async def refused(keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> object:
            raise RedisCircuitBreakerOpenError("Redis circuit breaker is open")

        return refused
