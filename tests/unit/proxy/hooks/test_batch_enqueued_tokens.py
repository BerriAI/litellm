"""
LIT-5273: enqueued-token accounting for batch submissions.

Covers the ``BatchEnqueuedTokenStore`` (reserve / refund / reservation
records), the metadata-driven scope resolution, and the batch-id and
response-shape helpers the v3 limiter's post-call hooks rely on.
"""

import base64
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.constants import BATCH_ENQUEUED_TOKEN_TTL_SECONDS
from litellm.litellm_core_utils.model_latest_aliases import get_model_latest_aliases
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BatchEnqueuedTokenOverLimit,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    BatchEnqueuedTokenStore,
    batch_response_view,
    canonical_provider_batch_id,
    project_has_model_batch_enqueued_token_limits,
    project_limits_every_model,
    resolve_batch_enqueued_token_scopes,
    resolve_project_model_batch_enqueued_token_scopes,
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


def _project_user(model_limits: object, project_id: str | None = "project-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-jwt-abc",
        team_id="shared-team",
        project_id=project_id,
        project_metadata={"model_batch_enqueued_token_limit": model_limits},
    )


def test_project_model_scopes_charge_each_limited_row_model_only_its_own_tokens():
    scopes: Final = resolve_project_model_batch_enqueued_token_scopes(
        _project_user({"gpt-5.5": 100, "claude-opus-5-5": 50}),
        row_tokens_by_model={"gpt-5.5": 80, "unlimited-model": 999},
        routing_model="unlimited-model",
        batch_tokens=1079,
        model_group_of=lambda model: model,
    )
    assert scopes == (BatchEnqueuedTokenScope(key="project_model", value="project-1:gpt-5.5", limit=100, tokens=80),)


def test_project_model_scopes_charge_the_routing_model_the_whole_batch_whatever_rows_name():
    scopes: Final = resolve_project_model_batch_enqueued_token_scopes(
        _project_user({"capped-deployment": 100}),
        row_tokens_by_model={"unlisted-model": 400},
        routing_model="capped-deployment",
        batch_tokens=500,
        model_group_of=lambda model: model,
    )
    assert scopes == (
        BatchEnqueuedTokenScope(key="project_model", value="project-1:capped-deployment", limit=100, tokens=500),
    )


def test_project_model_scopes_charge_rows_of_the_routing_group_once():
    groups: Final = MappingProxyType({"mini-alias": "gpt-mini", "gpt-mini": "gpt-mini"})
    scopes: Final = resolve_project_model_batch_enqueued_token_scopes(
        _project_user({"gpt-mini": 100}),
        row_tokens_by_model={"mini-alias": 30},
        routing_model="gpt-mini",
        batch_tokens=30,
        model_group_of=groups.__getitem__,
    )
    assert scopes == (BatchEnqueuedTokenScope(key="project_model", value="project-1:gpt-mini", limit=100, tokens=30),)


def test_project_model_scopes_share_one_counter_per_model_group_under_the_tightest_limit():
    groups: Final = MappingProxyType(
        {"gpt-latest": "gpt-5.5", "gpt-5.5": "gpt-5.5", "mini": "gpt-mini", "gpt-mini": "gpt-mini"}
    )
    scopes: Final = resolve_project_model_batch_enqueued_token_scopes(
        _project_user({"gpt-latest": 100, "gpt-5.5": 70, "gpt-mini": 40}),
        row_tokens_by_model={"gpt-latest": 30, "gpt-5.5": 20, "mini": 5},
        routing_model=None,
        batch_tokens=55,
        model_group_of=groups.__getitem__,
    )
    assert scopes == (
        BatchEnqueuedTokenScope(key="project_model", value="project-1:gpt-5.5", limit=70, tokens=50),
        BatchEnqueuedTokenScope(key="project_model", value="project-1:gpt-mini", limit=40, tokens=5),
    )


@pytest.mark.parametrize("named_model", ["gpt-latest", "gpt-5.5"], ids=["only an alias", "only the group"])
def test_project_model_scope_takes_its_groups_tightest_limit_whichever_name_the_batch_uses(named_model: str) -> None:
    groups: Final = MappingProxyType({"gpt-latest": "gpt-5.5", "gpt-pinned": "gpt-5.5", "gpt-5.5": "gpt-5.5"})
    scopes: Final = resolve_project_model_batch_enqueued_token_scopes(
        _project_user({"gpt-latest": 100, "gpt-5.5": 70, "gpt-pinned": 60}),
        row_tokens_by_model={named_model: 30},
        routing_model=named_model,
        batch_tokens=30,
        model_group_of=groups.__getitem__,
    )
    assert scopes == (BatchEnqueuedTokenScope(key="project_model", value="project-1:gpt-5.5", limit=60, tokens=30),)


@pytest.mark.parametrize(
    "user",
    [
        _project_user(100),
        _project_user({"gpt-5.5": 0}),
        _project_user({"gpt-5.5": "lots"}),
        _project_user(["gpt-5.5"]),
        _project_user({"gpt-5.5": True}),
        _project_user({"gpt-5.5": "100"}),
        _project_user({"gpt-5.5": 100.0}),
        _project_user({"gpt-5.5": 100}, project_id=None),
        UserAPIKeyAuth(api_key="k", project_id="project-1"),
    ],
    ids=["scalar", "zero", "non-numeric", "list", "boolean", "numeric string", "float", "no project id", "no metadata"],
)
def test_project_model_scopes_ignore_unusable_limits(user):
    assert not project_has_model_batch_enqueued_token_limits(user)
    assert (
        resolve_project_model_batch_enqueued_token_scopes(
            user, row_tokens_by_model={"gpt-5.5": 10}, routing_model="gpt-5.5", batch_tokens=10, model_group_of=str
        )
        == ()
    )


@pytest.mark.parametrize(
    ("models", "expected"),
    [
        (("gpt-5.5",), True),
        (("gpt-latest",), True),
        (("unlisted-model",), False),
        (("gpt-5.5", "unlisted-model"), False),
    ],
    ids=["limited model", "alias of a limited group", "unlisted model", "one of the models is unlisted"],
)
def test_project_limits_every_model_only_when_each_model_has_a_limit(models, expected):
    groups: Final = MappingProxyType({"gpt-latest": "gpt-5.5"})
    assert (
        project_limits_every_model(
            _project_user({"gpt-5.5": 100}), models, model_group_of=lambda name: groups.get(name, name)
        )
        is expected
    )


def test_a_latest_alias_is_charged_under_the_project_limit_of_its_target():
    alias, target = next(iter(get_model_latest_aliases()["openai"].items()))
    user: Final = _project_user({target: 100})

    assert project_limits_every_model(user, (alias,), model_group_of=str)
    assert resolve_project_model_batch_enqueued_token_scopes(
        user, row_tokens_by_model={alias: 40}, routing_model=alias, batch_tokens=40, model_group_of=str
    ) == (BatchEnqueuedTokenScope(key="project_model", value=f"project-1:{target}", limit=100, tokens=40),)


def test_project_model_limits_are_detected_before_the_batch_file_is_read():
    assert project_has_model_batch_enqueued_token_limits(_project_user({"gpt-5.5": 100}))


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
async def test_project_model_scope_is_charged_its_own_tokens_and_refunded_after_a_round_trip():
    fake: Final = _SingleKeyRedisFake()
    store: Final = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )
    key_scope: Final = _scope(limit=1000, key="api_key")
    model_scope: Final = BatchEnqueuedTokenScope(
        key="project_model", value=f"project-{uuid.uuid4().hex}:gpt-5.5", limit=40, tokens=30
    )
    key_counter: Final = f"batch_enqueued_tokens:api_key:{key_scope.value}"
    model_counter: Final = f"batch_enqueued_tokens:project_model:{model_scope.value}"

    reservation: Final = await store.reserve(tokens=100, scopes=(key_scope, model_scope))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert fake.counters == {key_counter: 100, model_counter: 30}

    over: Final = await store.reserve(
        tokens=100,
        scopes=(key_scope, BatchEnqueuedTokenScope(key="project_model", value=model_scope.value, limit=40, tokens=20)),
    )
    assert isinstance(over, BatchEnqueuedTokenOverLimit)
    assert over.enqueued == 30
    assert fake.counters == {key_counter: 100, model_counter: 30}

    await store.save_reservation("batch_project_model", reservation)
    popped: Final = await store.pop_reservation("batch_project_model")
    assert popped == reservation
    await store.refund(popped)
    assert not fake.counters


@dataclass(frozen=True, slots=True)
class _ScopeBeforeProjectScopes:
    key: Literal["api_key", "team"]
    value: str
    limit: int


@dataclass(frozen=True, slots=True)
class _ReservationBeforeProjectScopes:
    tokens: int
    scopes: tuple[_ScopeBeforeProjectScopes, ...]


@pytest.mark.asyncio
async def test_a_pod_that_predates_project_scopes_still_reads_the_key_and_team_scopes_of_a_saved_record():
    fake: Final = _SingleKeyRedisFake()
    store: Final = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=fake, default_in_memory_ttl=60))
    )
    key_scope: Final = _scope(limit=1000, key="api_key")
    model_scope: Final = BatchEnqueuedTokenScope(key="project_model", value="project-1:gpt-5.5", limit=40, tokens=30)
    reservation: Final = await store.reserve(tokens=100, scopes=(key_scope, model_scope))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)

    await store.save_reservation("batch_mixed_versions", reservation)

    (record,) = fake.records.values()
    assert TypeAdapter(_ReservationBeforeProjectScopes).validate_json(record) == _ReservationBeforeProjectScopes(
        tokens=100, scopes=(_ScopeBeforeProjectScopes(key="api_key", value=key_scope.value, limit=1000),)
    )


@pytest.mark.asyncio
async def test_in_memory_project_model_scope_is_charged_and_refunded_its_own_tokens():
    store: Final = _in_memory_store()
    key_scope: Final = _scope(limit=1000, key="api_key")
    model_scope: Final = BatchEnqueuedTokenScope(
        key="project_model", value=f"project-{uuid.uuid4().hex}:gpt-5.5", limit=40, tokens=30
    )
    reservation: Final = await store.reserve(tokens=100, scopes=(key_scope, model_scope))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    blocked: Final = BatchEnqueuedTokenScope(key="project_model", value=model_scope.value, limit=40, tokens=20)
    assert await store.reserve(tokens=1, scopes=(blocked,)) == BatchEnqueuedTokenOverLimit(scope=blocked, enqueued=30)

    await store.refund(reservation)
    full: Final = BatchEnqueuedTokenScope(key="project_model", value=model_scope.value, limit=40, tokens=40)
    assert isinstance(await store.reserve(tokens=1, scopes=(full,)), BatchEnqueuedTokenReservation)
    assert isinstance(await store.reserve(tokens=900, scopes=(key_scope,)), BatchEnqueuedTokenReservation)
    assert isinstance(await store.reserve(tokens=101, scopes=(key_scope,)), BatchEnqueuedTokenOverLimit)


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


class _OpenBreakerRedis:
    def async_register_script(self, script: str):
        async def refused(keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> object:
            raise RedisCircuitBreakerOpenError("Redis circuit breaker is open")

        return refused


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_keeps_reservations_in_memory_without_a_warning(caplog):
    scope = _scope(limit=100)
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=_OpenBreakerRedis(), default_in_memory_ttl=60))  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    )

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        reservation = await store.reserve(tokens=60, scopes=(scope,))
        assert isinstance(reservation, BatchEnqueuedTokenReservation)
        assert reservation.backend == "memory"
        await store.save_reservation("batch_quiet", reservation)
        assert await store.pop_reservation("batch_quiet") == reservation

    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert sum("circuit breaker is open" in record.getMessage() for record in caplog.records) == 3
