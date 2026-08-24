"""
LIT-5273: enqueued-token accounting for batch submissions.

Covers the ``BatchEnqueuedTokenStore`` (reserve / refund / reservation
records), the metadata-driven scope resolution, and the batch-id and
response-shape helpers the v3 limiter's post-call hooks rely on.
"""

import base64
import socket
import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.constants import BATCH_ENQUEUED_TOKEN_TTL_SECONDS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BatchEnqueuedTokenOverLimit,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    BatchEnqueuedTokenStore,
    batch_response_view,
    canonical_provider_batch_id,
    resolve_batch_enqueued_token_scopes,
)
from litellm.proxy.utils import InternalUsageCache


def _in_memory_store() -> BatchEnqueuedTokenStore:
    return BatchEnqueuedTokenStore(internal_usage_cache=InternalUsageCache(DualCache(default_in_memory_ttl=60)))


def _scope(limit: int, key: str = "api_key") -> BatchEnqueuedTokenScope:
    return BatchEnqueuedTokenScope(key=key, value=f"{key}-{uuid.uuid4().hex}", limit=limit)


def test_scope_resolution_reads_key_and_team_metadata():
    user = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={"batch_enqueued_token_limit": 100},
        team_id="team-1",
        team_metadata={"batch_enqueued_token_limit": "150"},
    )
    scopes = resolve_batch_enqueued_token_scopes(user)
    assert scopes == (
        BatchEnqueuedTokenScope(key="api_key", value="hashed-key", limit=100),
        BatchEnqueuedTokenScope(key="team", value="team-1", limit=150),
    )


def test_scope_resolution_returns_empty_without_opt_in():
    assert resolve_batch_enqueued_token_scopes(UserAPIKeyAuth(api_key="k")) == ()
    assert resolve_batch_enqueued_token_scopes(UserAPIKeyAuth(api_key="k", metadata={}, team_metadata=None)) == ()


@pytest.mark.parametrize("bad_value", ["not-a-number", 0, -5, None, [1000]])
def test_scope_resolution_ignores_invalid_limits(bad_value):
    user = UserAPIKeyAuth(api_key="k", metadata={"batch_enqueued_token_limit": bad_value})
    assert resolve_batch_enqueued_token_scopes(user) == ()


def test_scope_resolution_skips_team_scope_without_team_id():
    user = UserAPIKeyAuth(api_key="k", team_metadata={"batch_enqueued_token_limit": 100})
    assert resolve_batch_enqueued_token_scopes(user) == ()


@pytest.mark.asyncio
async def test_reserve_rejects_once_allowance_is_exhausted():
    store = _in_memory_store()
    scope = _scope(limit=100)
    first = await store.reserve(tokens=80, scopes=(scope,))
    assert isinstance(first, BatchEnqueuedTokenReservation)
    second = await store.reserve(tokens=30, scopes=(scope,))
    assert second == BatchEnqueuedTokenOverLimit(scope=scope, enqueued=80)
    third = await store.reserve(tokens=20, scopes=(scope,))
    assert isinstance(third, BatchEnqueuedTokenReservation)


@pytest.mark.asyncio
async def test_reserve_is_all_or_nothing_across_scopes():
    store = _in_memory_store()
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")
    over = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
    assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)
    exact_fit = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(exact_fit, BatchEnqueuedTokenReservation)


@pytest.mark.asyncio
async def test_refund_restores_allowance_and_never_goes_negative():
    store = _in_memory_store()
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=30, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.refund(reservation)
    await store.refund(reservation)
    refill = await store.reserve(tokens=100, scopes=(scope,))
    assert isinstance(refill, BatchEnqueuedTokenReservation)
    assert isinstance(await store.reserve(tokens=1, scopes=(scope,)), BatchEnqueuedTokenOverLimit)


@pytest.mark.asyncio
async def test_reservation_record_roundtrip_pops_exactly_once():
    store = _in_memory_store()
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=40, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.save_reservation("batch_abc", reservation)
    assert await store.pop_reservation("batch_abc") == reservation
    assert await store.pop_reservation("batch_abc") is None
    assert await store.pop_reservation("batch_never_saved") is None


@pytest.mark.asyncio
async def test_zero_token_reserve_charges_nothing():
    store = _in_memory_store()
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
            else "refund" if "DECRBY" in script else "pop" if "GET" in script else "save"
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
async def test_redis_reserve_issues_single_key_calls_and_rolls_back_on_over_limit():
    fake = _SingleKeyRedisFake()
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")

    over = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
    assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)
    assert tuple(kind for kind, _ in fake.script_calls) == ("reserve", "reserve", "refund")
    assert not fake.counters

    fits = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(fits, BatchEnqueuedTokenReservation)
    await store.refund(fits)
    assert not fake.counters
    assert all(len(keys) == 1 for _, keys in fake.script_calls)


@pytest.mark.asyncio
async def test_partial_redis_reserve_failure_rolls_back_and_grants_in_memory():
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")
    fake = _SingleKeyRedisFake(fail_reserve_keys=frozenset({f"batch_enqueued_tokens:team:{team_scope.value}"}))
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )

    outcome = await store.reserve(tokens=10, scopes=(key_scope, team_scope))
    assert isinstance(outcome, BatchEnqueuedTokenReservation)
    assert outcome.backend == "memory"
    assert tuple(kind for kind, _ in fake.script_calls) == ("reserve", "reserve", "refund")
    assert not fake.counters

    await store.refund(outcome)
    assert tuple(kind for kind, _ in fake.script_calls) == ("reserve", "reserve", "refund")

    refilled = await store.reserve(tokens=50, scopes=(team_scope,))
    assert isinstance(refilled, BatchEnqueuedTokenReservation)
    assert refilled.backend == "memory"


@pytest.mark.asyncio
async def test_over_limit_verdict_survives_a_failing_rollback():
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=5, key="team")
    key_counter: Final = f"batch_enqueued_tokens:api_key:{key_scope.value}"
    fake = _SingleKeyRedisFake(fail_refund_keys=frozenset({key_counter}))
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )

    outcome = await store.reserve(tokens=10, scopes=(key_scope, team_scope))
    assert outcome == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)
    assert fake.counters == {key_counter: 10}


@pytest.mark.asyncio
async def test_pop_falls_back_to_local_record_when_redis_pop_finds_nothing():
    scope = _scope(limit=100)
    record_key: Final = "batch_enqueued_token_reservation:batch_local_record"
    fake = _SingleKeyRedisFake(fail_save_keys=frozenset({record_key}))
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )

    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.save_reservation("batch_local_record", reservation)
    assert not fake.records

    popped = await store.pop_reservation("batch_local_record")
    assert popped == reservation
    await store.refund(popped)
    assert not fake.counters
    assert await store.pop_reservation("batch_local_record") is None


@pytest.mark.asyncio
async def test_local_ghost_left_by_landed_save_never_refunds_twice():
    scope = _scope(limit=100)
    record_key: Final = "batch_enqueued_token_reservation:batch_ghost"
    fake = _SingleKeyRedisFake(raise_after_landing_save_keys=frozenset({record_key}))
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )

    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.save_reservation("batch_ghost", reservation)
    assert fake.records[record_key]

    first = await store.pop_reservation("batch_ghost")
    assert first == reservation
    await store.refund(first)
    assert not fake.counters
    assert fake.records[record_key] == ""

    assert await store.pop_reservation("batch_ghost") is None
    assert (
        await store.internal_usage_cache.async_get_cache(
            key=record_key, litellm_parent_otel_span=None, local_only=True
        )
        is None
    )
    assert await store.pop_reservation("batch_ghost") is None
    assert isinstance(await store.reserve(tokens=100, scopes=(scope,)), BatchEnqueuedTokenReservation)
    assert fake.counters[f"batch_enqueued_tokens:{scope.key}:{scope.value}"] == 100


@pytest.mark.asyncio
async def test_record_ttl_shrinks_by_elapsed_time_so_stale_records_never_outlive_their_counters():
    scope = _scope(limit=100)
    fake = _SingleKeyRedisFake()
    ticks = iter((1_000.0, 1_030.5))
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60)),
        monotonic=lambda: next(ticks),
    )
    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert reservation.reserved_at_monotonic == 1_000.0
    await store.save_reservation("batch_ttl_clamp", reservation)
    assert fake.save_ttls == (BATCH_ENQUEUED_TOKEN_TTL_SECONDS - 31,)
    assert await store.pop_reservation("batch_ttl_clamp") == reservation


@pytest.mark.asyncio
async def test_memory_refund_skips_reservations_granted_by_another_worker():
    store = _in_memory_store()
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


@pytest.mark.asyncio
async def test_failed_redis_refund_leaves_local_counters_untouched():
    scope = _scope(limit=100)
    counter_key: Final = f"batch_enqueued_tokens:api_key:{scope.value}"
    fake = _SingleKeyRedisFake(fail_refund_keys=frozenset({counter_key}))
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )

    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert reservation.backend == "redis"
    store.internal_usage_cache.dual_cache.in_memory_cache.set_cache(key=counter_key, value=45)

    await store.refund(reservation)
    assert store.internal_usage_cache.dual_cache.in_memory_cache.get_cache(key=counter_key) == 45
    assert fake.counters == {counter_key: 60}


@pytest.mark.asyncio
async def test_pop_reservation_defaults_legacy_records_to_redis_backend():
    store = _in_memory_store()
    legacy = '{"tokens": 5, "scopes": [{"key": "api_key", "value": "k", "limit": 10}]}'
    store.internal_usage_cache.dual_cache.in_memory_cache.set_cache(
        key="batch_enqueued_token_reservation:batch_legacy", value=legacy
    )
    popped = await store.pop_reservation("batch_legacy")
    assert popped == BatchEnqueuedTokenReservation(
        tokens=5, scopes=(BatchEnqueuedTokenScope(key="api_key", value="k", limit=10),), backend="redis"
    )


def test_canonical_provider_batch_id_passes_raw_ids_through():
    assert canonical_provider_batch_id("batch_abc123") == "batch_abc123"


def test_canonical_provider_batch_id_decodes_unified_batch_ids():
    unified = "litellm_proxy;model_id:m-1;llm_batch_id:batch_prov_9;llm_output_file_id:file-9"
    encoded = base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")
    assert canonical_provider_batch_id(encoded) == "batch_prov_9"


def test_canonical_provider_batch_id_decodes_model_embedded_ids():
    from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

    encoded = encode_file_id_with_model(file_id="batch_prov_7", model="my-alias", id_type="batch")
    assert canonical_provider_batch_id(encoded) == "batch_prov_7"


def test_batch_response_view_accepts_batch_objects_only():
    batch = SimpleNamespace(id="batch_1", status="completed", object="batch")
    view = batch_response_view(batch)
    assert view is not None and view.id == "batch_1" and view.status == "completed"
    assert batch_response_view({"id": "chatcmpl-1", "object": "chat.completion"}) is None
    assert batch_response_view(None) is None
    assert batch_response_view("batch_1") is None


def _local_redis_port() -> int | None:
    for port in (6379,):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return port
    return None


@pytest.mark.asyncio
@pytest.mark.skipif(_local_redis_port() is None, reason="requires a local Redis on 6379 for the Lua script path")
async def test_redis_lua_path_full_lifecycle():
    from litellm.caching.redis_cache import RedisCache

    port = _local_redis_port()
    redis_cache = RedisCache(host="127.0.0.1", port=port)
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=redis_cache, default_in_memory_ttl=60))
    )
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")

    over = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
    assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)

    reservation = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert isinstance(await store.reserve(tokens=1, scopes=(key_scope, team_scope)), BatchEnqueuedTokenOverLimit)

    batch_id = f"batch_{uuid.uuid4().hex}"
    await store.save_reservation(batch_id, reservation)
    popped = await store.pop_reservation(batch_id)
    assert popped == reservation
    assert await store.pop_reservation(batch_id) is None

    await store.refund(popped)
    refill = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(refill, BatchEnqueuedTokenReservation)
    await store.refund(refill)
