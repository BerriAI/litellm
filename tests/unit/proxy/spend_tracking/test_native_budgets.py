from __future__ import annotations

import gc
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Final, TypeAlias

import pytest

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, SourcedFields
from litellm.integrations.custom_logger_v2.legacy import PROXY_REQUEST_KEYS, KwargsFields
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.models.budget import BudgetLimit, LiteLLM_BudgetTable
from litellm.models.tag import LiteLLM_TagTable
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    model_access_group_cache_key,
    project_cache_key,
    tag_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking import native_budgets
from litellm.proxy.spend_tracking.budget_reservation import (
    customer_window_counters,
    estimate_request_max_cost,
    reconcile_budget_reservation,
    release_budget_reservation,
    release_budget_reservation_on_cancel,
    release_or_invalidate_budget_reservation,
    release_unbound_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.proxy.v2_proxy_fields import PRE_CALL_SOURCES
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth, CustomerScopeKey
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget

ROUTE: Final = "/chat/completions"
BODY: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1000}
API_KEY: Final = "sk-native"
TOKEN: Final = hash_token(API_KEY)
ESTIMATE: Final = estimate_request_max_cost(request_body=BODY, route=ROUTE, llm_router=None, input_token_counts={})
KEY: Final = f"spend:key:{TOKEN}"
TEAM: Final = "spend:team:team-1"
USER: Final = "spend:user:user-1"
END_USER: Final = "spend:end_user:customer-1"


@dataclass(frozen=True, slots=True)
class Identity:
    key_budget: float | None = 100.0
    team_budget: float | None = None
    user_budget: float | None = None
    end_user_budget: float | None = None
    throttle: bool = False
    fail_closed: bool = False
    spend: Mapping[str, float] = field(default_factory=dict)

    def token(self) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key=API_KEY,
            token=TOKEN,
            max_budget=self.key_budget,
            spend=0.0,
            team_id="team-1" if self.team_budget is not None else None,
            user_id="user-1",
            end_user_id="customer-1" if self.end_user_budget is not None else None,
            end_user_max_budget=self.end_user_budget,
            tpm_limit=1000 if self.throttle else None,
            metadata={"throttle_on_budget_exceeded": True} if self.throttle else {},
        )

    def team(self) -> LiteLLM_TeamTable | None:
        return None if self.team_budget is None else LiteLLM_TeamTable(team_id="team-1", max_budget=self.team_budget)

    def user(self) -> LiteLLM_UserTable | None:
        return None if self.user_budget is None else LiteLLM_UserTable(user_id="user-1", max_budget=self.user_budget)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[SpendCounterStore]:
    spend_counter_store: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend_counter_store)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.5)
    monkeypatch.setattr(litellm, "callbacks", [])
    yield spend_counter_store
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # later tests run without a native owner


def _seed(store: SpendCounterStore, identity: Identity) -> None:
    for counter, value in identity.spend.items():
        store.set_local(counter, value)


def _snapshot(store: SpendCounterStore) -> Mapping[str, float]:
    return {key: round(float(value), 12) for key, value in store.snapshot().items()}


def _arguments(identity: Identity) -> Mapping[str, object]:
    return {
        "request_body": dict(BODY),
        "route": ROUTE,
        "llm_router": None,
        "valid_token": identity.token(),
        "team_object": identity.team(),
        "user_object": identity.user(),
        "prisma_client": None,
        "user_api_key_cache": UserApiKeyCache(),
        "proxy_logging_obj": ProxyLogging(user_api_key_cache=DualCache()),
        "end_user_id": "customer-1" if identity.end_user_budget is not None else None,
        "fail_closed_budget_enforcement": identity.fail_closed,
    }


@dataclass(frozen=True, slots=True)
class Admitted:
    reservation: dict[str, object] | None  # mutable-ok: settlement stamps finalized on the shared reservation
    refusal: Exception | None
    counters: Mapping[str, float]
    abort_status: int | None = None


def _install_native_owner(store: SpendCounterStore) -> native_budgets.NativeBudgetsLogger:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # each case installs its own owner over its own store
    litellm.callbacks.clear()
    native_budgets.install_native_budgets(store, litellm.callbacks)
    logger: Final = litellm.callbacks[0]
    assert isinstance(logger, native_budgets.NativeBudgetsLogger)
    return logger


async def _python_admit(store: SpendCounterStore, identity: Identity) -> Admitted:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # the reference run has no native owner
    _seed(store, identity)
    try:
        reservation: Final = await reserve_budget_for_request(**_arguments(identity))  # pyright: ignore[reportArgumentType]  # the shared argument table is typed per key above
    except litellm.BudgetExceededError as error:
        return Admitted(reservation=None, refusal=error, counters=_snapshot(store))
    return Admitted(reservation=reservation, refusal=None, counters=_snapshot(store))


async def _pre_call(
    logger: native_budgets.NativeBudgetsLogger, auth: UserAPIKeyAuth, data: dict[str, object]
) -> Patch | Abort | None:  # mutable-ok: the request dict the proxy hands its hooks
    fields: Final = SourcedFields(
        KwargsFields(data, PROXY_REQUEST_KEYS), PRE_CALL_SOURCES, HookArgs(auth=auth, request=data)
    )
    outcome: Final = await logger.run_rust(
        Hook.pre_call,
        frozenset({Field.budget_plan}),
        FieldView(fields),
        CallContext[object](hook=Hook.pre_call, call=None, state=None),
    )
    if isinstance(outcome, Patch):
        _ = fields.apply(outcome)
    return outcome


async def _native_admit(store: SpendCounterStore, identity: Identity, call_id: str) -> Admitted:
    logger: Final = _install_native_owner(store)
    _seed(store, identity)
    arguments: Final = _arguments(identity)
    auth: Final = arguments["valid_token"]
    assert isinstance(auth, UserAPIKeyAuth)
    plan: Final = await reserve_budget_for_request(**arguments)  # pyright: ignore[reportArgumentType]  # the shared argument table is typed per key above
    outcome: Final = await _pre_call(logger, auth, {**BODY, "litellm_call_id": call_id})
    refusal: Final = outcome.exception if isinstance(outcome, Abort) else None
    reservation: Final = plan if plan is not None and plan.get("native_call_id") is not None else None
    return Admitted(
        reservation=reservation,
        refusal=refusal,
        counters=_snapshot(store),
        abort_status=outcome.status if isinstance(outcome, Abort) else None,
    )


def _refusal_fields(error: Exception | None) -> object:
    if error is None:
        return None
    return (
        type(error),
        str(error),
        {name: getattr(error, name, None) for name in ("current_cost", "max_budget", "entity_type", "entity_id")},
        getattr(error, "status_code", None),
    )


def _entries(reservation: Mapping[str, object] | None) -> object:
    if reservation is None:
        return None
    return (
        reservation["reserved_cost"],
        reservation["entries"],
        reservation["input_cost"],
        reservation["input_tokens"],
    )


assert ESTIMATE is not None and ESTIMATE > 0

CASES: Final = {
    "key_and_team_fit": Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0}),
    "team_shrinks_to_what_is_left": Identity(team_budget=10.0, spend={KEY: 1.0, TEAM: 10.0 - ESTIMATE / 3}),
    "key_spent_refuses": Identity(key_budget=5.0, spend={KEY: 5.0}),
    "team_at_max_refuses": Identity(team_budget=10.0, spend={KEY: 0.0, TEAM: 10.0}),
    "user_on_a_personal_key": Identity(user_budget=50.0, spend={KEY: 1.0, USER: 3.0}),
    "end_user_over_budget": Identity(end_user_budget=2.0, spend={KEY: 1.0, END_USER: 2.0}),
    "throttled_key_releases_its_hold": Identity(key_budget=5.0, team_budget=100.0, throttle=True, spend={KEY: 5.0}),
    "fail_closed_refuses_any_overshoot": Identity(
        team_budget=10.0, fail_closed=True, spend={KEY: 0.0, TEAM: 10.0 - ESTIMATE / 3}
    ),
    "no_budget_reserves_nothing": Identity(key_budget=None),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", CASES.values(), ids=CASES.keys())
async def test_native_admission_matches_python(store: SpendCounterStore, identity: Identity) -> None:
    python: Final = await _python_admit(store, identity)
    store.flush_local()
    native: Final = await _native_admit(store, identity, "call-admit")

    assert _refusal_fields(native.refusal) == _refusal_fields(python.refusal)
    assert native.counters == python.counters
    assert _entries(native.reservation) == _entries(python.reservation)


@pytest.mark.asyncio
async def test_a_runtime_change_to_the_budget_exceeded_status_code_reaches_the_next_refusal(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity: Final = CASES["team_at_max_refuses"]
    seen: Final[dict[int, tuple[object, object, int | None]]] = {}  # mutable-ok: one entry per status code
    for code in (402, 429):
        monkeypatch.setattr(litellm, "budget_exceeded_status_code", code)
        store.flush_local()
        python = await _python_admit(store, identity)
        store.flush_local()
        native = await _native_admit(store, identity, f"call-status-{code}")
        seen[code] = (_refusal_fields(python.refusal), _refusal_fields(native.refusal), native.abort_status)

    assert {code: (python, status) for code, (python, _, status) in seen.items()} == {
        code: (native, code) for code, (_, native, _) in seen.items()
    }
    assert seen[402][0] != seen[429][0]


Settle: TypeAlias = Callable[[dict[str, object]], Awaitable[object]]

SETTLEMENTS: Final = {
    "success_charges_the_actual_cost": lambda reservation: reconcile_budget_reservation(reservation, 0.002),
    "failure_refunds_the_hold": release_or_invalidate_budget_reservation,
    "cancel_keeps_the_input_cost": release_budget_reservation_on_cancel,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("settle", SETTLEMENTS.values(), ids=SETTLEMENTS.keys())
async def test_native_settlement_matches_python(
    store: SpendCounterStore,
    settle: Settle,
) -> None:
    identity: Final = Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0})
    python: Final = await _python_admit(store, identity)
    assert python.reservation is not None
    await settle(python.reservation)
    await settle(python.reservation)
    settled_python: Final = _snapshot(store)
    store.flush_local()

    native: Final = await _native_admit(store, identity, "call-settle")
    assert native.reservation is not None
    await settle(native.reservation)
    await settle(native.reservation)

    assert _snapshot(store) == settled_python
    assert settled_python != python.counters


async def _db_logger_success(reservation: dict[str, object]) -> None:  # mutable-ok: the shared reservation
    _ = await reconcile_budget_reservation(reservation, 0.002, finalize=False, apply_consistent=False)
    await proxy_server.increment_spend_counters(
        token=TOKEN, team_id="team-1", user_id="user-1", response_cost=0.002, budget_reservation=reservation
    )


EXITS: Final[Mapping[str, Settle]] = {
    "db_logger_charges_the_row": _db_logger_success,
    "db_logger_releases_an_uncharged_row": release_budget_reservation,
    "db_logger_releases_after_a_failed_write": release_or_invalidate_budget_reservation,
    "failure_hook_releases": release_or_invalidate_budget_reservation,
    "client_disconnect_keeps_the_input_cost": release_budget_reservation_on_cancel,
    "request_end_releases_an_unbound_hold": release_unbound_budget_reservation,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", EXITS.values(), ids=EXITS.keys())
async def test_every_exit_settles_the_native_hold_like_python(store: SpendCounterStore, exit_path: Settle) -> None:
    identity: Final = Identity(team_budget=100.0, user_budget=100.0, spend={KEY: 1.0, TEAM: 2.0, USER: 0.5})
    python: Final = await _python_admit(store, identity)
    assert python.reservation is not None
    await exit_path(python.reservation)
    settled_python: Final = _snapshot(store)
    store.flush_local()

    native: Final = await _native_admit(store, identity, "call-exit")
    assert native.reservation is not None
    await exit_path(native.reservation)

    assert _snapshot(store) == settled_python
    assert native.reservation.get("finalized") == python.reservation.get("finalized")


@pytest.mark.asyncio
async def test_a_stream_settles_after_its_request_state_is_gone(store: SpendCounterStore) -> None:
    identity: Final = Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0})
    python: Final = await _python_admit(store, identity)
    assert python.reservation is not None
    await _db_logger_success(python.reservation)
    settled_python: Final = _snapshot(store)
    store.flush_local()

    native: Final = await _native_admit(store, identity, "call-stream")
    assert native.reservation is not None
    reservation: Final = native.reservation
    del native
    gc.collect()
    await _db_logger_success(reservation)

    assert _snapshot(store) == settled_python


DB_FLOOR: Final = 0.25
DISTURBANCES: Final[Mapping[str, float | None]] = {"counter_expired": None, "counter_would_go_negative": 1e-9}


@pytest.mark.asyncio
@pytest.mark.parametrize("disturbance", DISTURBANCES.values(), ids=DISTURBANCES.keys())
@pytest.mark.parametrize("exit_path", EXITS.values(), ids=EXITS.keys())
async def test_settling_a_counter_that_went_inconsistent_reseeds_it_like_python(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch, disturbance: float | None, exit_path: Settle
) -> None:
    async def reseed_from_db(counter_key: str) -> bool:
        store.set_local(counter_key, DB_FLOOR)
        return True

    monkeypatch.setattr(proxy_server, "reseed_spend_counter_from_db", reseed_from_db)
    identity: Final = Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0})
    settled: Final[list[Mapping[str, float]]] = []  # mutable-ok: one snapshot per side
    for native in (False, True):
        store.flush_local()
        admitted = await (_native_admit(store, identity, "call-reseed") if native else _python_admit(store, identity))
        assert admitted.reservation is not None
        if disturbance is None:
            store.flush_local()
            _seed(store, Identity(spend={TEAM: identity.spend[TEAM] + ESTIMATE}))
        else:
            store.set_local(KEY, disturbance)
        await exit_path(admitted.reservation)
        settled.append(_snapshot(store))

    assert settled[1] == settled[0]


@pytest.mark.asyncio
async def test_a_throttled_key_holds_only_its_other_counters_on_both_sides(store: SpendCounterStore) -> None:
    identity: Final = CASES["throttled_key_releases_its_hold"]
    python: Final = await _python_admit(store, identity)
    store.flush_local()
    native: Final = await _native_admit(store, identity, "call-throttle")

    held: Final = [
        None if admitted.reservation is None else [entry["counter_key"] for entry in admitted.reservation["entries"]]  # pyright: ignore[reportGeneralTypeIssues]  # the entries list the reservation carries
        for admitted in (python, native)
    ]
    assert held == [[TEAM], [TEAM]]
    assert native.counters == python.counters


class _AnyDatabase:
    pass


ORG: Final = "spend:org:org-1"
PROJECT: Final = "spend:project:project-1"
MAG: Final = "spend:model_access_group:mag-1"
MEMBER: Final = "spend:team_member:user-1:team-1"
TAG: Final = "spend:tag:tag-1"
KEY_WINDOW: Final = f"{KEY}:window:1d"


@dataclass(frozen=True, slots=True)
class Kind:
    counter: str
    entity_type: str
    token: Mapping[str, object] = field(default_factory=dict)
    cache: Mapping[str, object] = field(default_factory=dict)
    body: Mapping[str, object] = field(default_factory=dict)
    member: bool = False


def _budget(max_budget: float) -> LiteLLM_BudgetTable:
    return LiteLLM_BudgetTable(max_budget=max_budget)


KIND_BUDGET: Final = 10.0
CUSTOMER_LIMITS: Final = (BudgetLimit(budget_duration="1mo", max_budget=KIND_BUDGET),)


def _customer_kind(scope: CustomerScopeKey, entity_id: str) -> Kind:
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="customer-org-1",
        scopes=(CustomerScopeAuth(key=scope, entity_id=entity_id, budget_limits=CUSTOMER_LIMITS),),
    )
    return Kind(
        customer_window_counters(scope, entity_id, CUSTOMER_LIMITS)[0].counter_key,
        scope,
        token={"customer_hierarchy": hierarchy},
    )


KINDS: Final[Mapping[str, Kind]] = {
    "organization": Kind(
        ORG,
        "Organization",
        token={"org_id": "org-1"},
        cache={"org_id:org-1:with_budget": {"litellm_budget_table": {"max_budget": KIND_BUDGET}, "spend": 0.0}},
    ),
    "project": Kind(
        PROJECT,
        "Project",
        token={"project_id": "project-1"},
        cache={project_cache_key("project-1"): {"litellm_budget_table": {"max_budget": KIND_BUDGET}, "spend": 0.0}},
    ),
    "model_access_group": Kind(
        MAG,
        "Model access group",
        token={"matched_model_access_groups": ["mag-1"]},
        cache={
            model_access_group_cache_key("mag-1"): ModelAccessGroupBudget(
                access_group_name="mag-1", max_budget=KIND_BUDGET
            )
        },
    ),
    "team_member": Kind(
        MEMBER,
        "TeamMember",
        cache={
            team_membership_reservation_cache_key(user_id="user-1", team_id="team-1"): LiteLLM_TeamMembership(
                user_id="user-1", team_id="team-1", litellm_budget_table=_budget(KIND_BUDGET)
            )
        },
        member=True,
    ),
    "tag": Kind(
        TAG,
        "Tag",
        cache={tag_cache_key("tag-1"): LiteLLM_TagTable(tag_name="tag-1", litellm_budget_table=_budget(KIND_BUDGET))},
        body={"metadata": {"tags": ["tag-1"]}},
    ),
    "key_window": Kind(
        KEY_WINDOW,
        "Key",
        token={"budget_limits": [{"budget_duration": "1d", "max_budget": KIND_BUDGET}]},
    ),
    "customer_org": _customer_kind("customer_org", "customer-org-1"),
    "customer_team": _customer_kind("customer_team", "customer-team-1"),
    "customer_member": _customer_kind("customer_member", "member:1"),
}
KIND_SPEND: Final[Mapping[str, float]] = {
    "fits": 1.0,
    "shrinks_to_what_is_left": KIND_BUDGET - ESTIMATE / 3,
    "at_max_refuses": KIND_BUDGET,
}


async def _kind_admit(store: SpendCounterStore, kind: Kind, spend: float, native: bool) -> Admitted:
    cache: Final = UserApiKeyCache()
    for key, value in kind.cache.items():
        await cache.async_set_cache(key=key, value=value)
    token: Final = UserAPIKeyAuth(
        api_key=API_KEY,
        token=TOKEN,
        max_budget=100.0,
        spend=0.0,
        user_id="user-1",
        team_id="team-1" if kind.member else None,
        **kind.token,  # pyright: ignore[reportArgumentType]  # each kind sets the identity fields it reads
    )
    arguments: Final = {
        "request_body": {**BODY, **kind.body},
        "route": ROUTE,
        "llm_router": None,
        "valid_token": token,
        "team_object": LiteLLM_TeamTable(team_id="team-1") if kind.member else None,
        "user_object": LiteLLM_UserTable(user_id="user-1") if kind.member else None,
        "prisma_client": _AnyDatabase(),
        "user_api_key_cache": cache,
        "proxy_logging_obj": ProxyLogging(user_api_key_cache=DualCache()),
    }
    store.flush_local()
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # the reference run has no native owner
    logger: Final = _install_native_owner(store) if native else None
    store.set_local(KEY, 1.0)
    store.set_local(kind.counter, spend)
    try:
        reservation: Final = await reserve_budget_for_request(**arguments)  # pyright: ignore[reportArgumentType]  # the argument table is typed per key above
    except litellm.BudgetExceededError as error:
        return Admitted(reservation=None, refusal=error, counters=_snapshot(store))
    if logger is None:
        return Admitted(reservation=reservation, refusal=None, counters=_snapshot(store))
    outcome: Final = await _pre_call(logger, token, dict(BODY))
    return Admitted(
        reservation=reservation if reservation is not None and "native_call_id" in reservation else None,
        refusal=outcome.exception if isinstance(outcome, Abort) else None,
        counters=_snapshot(store),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("spend", KIND_SPEND.values(), ids=KIND_SPEND.keys())
@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
async def test_every_counter_kind_admits_like_python(store: SpendCounterStore, kind: Kind, spend: float) -> None:
    python: Final = await _kind_admit(store, kind, spend, native=False)
    native: Final = await _kind_admit(store, kind, spend, native=True)

    held: Final = (
        [] if python.reservation is None else [entry["entity_type"] for entry in python.reservation["entries"]]  # pyright: ignore[reportGeneralTypeIssues]  # the entries list the reservation carries
    )
    assert python.refusal is not None or held.count(kind.entity_type) >= 1, "the flag-off run must reserve this kind"
    assert _refusal_fields(native.refusal) == _refusal_fields(python.refusal)
    assert native.counters == python.counters
    assert _entries(native.reservation) == _entries(python.reservation)


WINDOW_RESETS: Final[Mapping[str, timedelta]] = {
    "window_reset_before_settlement": -timedelta(minutes=5),
    "window_still_open": timedelta(days=1),
}
CUSTOMER_KIND: Final = KINDS["customer_member"]


async def _settled_after_reset(
    store: SpendCounterStore, native: bool, reset: str, exit_path: Settle, expire: bool
) -> tuple[Mapping[str, float], object]:
    admitted: Final = await _kind_admit(store, CUSTOMER_KIND, 1.0, native=native)
    reservation: Final = admitted.reservation
    assert reservation is not None
    entries: Final = reservation["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        if entry["counter_key"] == CUSTOMER_KIND.counter:
            entry["window_reset_at"] = reset
    if expire:
        store.flush_local()
        store.set_local(KEY, 1.0 + float(reservation["reserved_cost"]))  # pyright: ignore[reportArgumentType]  # the reserved cost the reservation holds
    await exit_path(reservation)
    return _snapshot(store), reservation["entries"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", WINDOW_RESETS.values(), ids=WINDOW_RESETS.keys())
@pytest.mark.parametrize("exit_path", EXITS.values(), ids=EXITS.keys())
async def test_a_customer_window_that_reset_before_settlement_settles_like_python(
    store: SpendCounterStore, reset: timedelta, exit_path: Settle
) -> None:
    reset_at: Final = (datetime.now(timezone.utc) + reset).isoformat()
    python: Final = await _settled_after_reset(store, False, reset_at, exit_path, expire=False)
    native: Final = await _settled_after_reset(store, True, reset_at, exit_path, expire=False)

    assert native == python


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", WINDOW_RESETS.values(), ids=WINDOW_RESETS.keys())
async def test_an_expired_customer_window_is_reseeded_with_the_cost_its_window_settles_like_python(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch, reset: timedelta
) -> None:
    reseeded: Final[list[str]] = []  # mutable-ok: the counters each side reseeds

    async def reseed_from_db(counter_key: str) -> bool:
        reseeded.append(counter_key)
        store.set_local(counter_key, DB_FLOOR)
        return True

    monkeypatch.setattr(proxy_server, "reseed_spend_counter_from_db", reseed_from_db)
    settle: Final = SETTLEMENTS["success_charges_the_actual_cost"]
    reset_at: Final = (datetime.now(timezone.utc) + reset).isoformat()
    python: Final = await _settled_after_reset(store, False, reset_at, settle, expire=True)
    python_reseeded: Final = tuple(reseeded)
    reseeded.clear()
    native: Final = await _settled_after_reset(store, True, reset_at, settle, expire=True)

    assert python_reseeded == (CUSTOMER_KIND.counter,)
    assert tuple(reseeded) == python_reseeded
    assert native == python


@dataclass(slots=True)
class _SpendReads:
    db_spend: float | Exception
    db_reads: int = 0
    current_spend_calls: int = 0


def _count_spend_reads(monkeypatch: pytest.MonkeyPatch, db_spend: float | Exception) -> _SpendReads:
    reads: Final = _SpendReads(db_spend=db_spend)
    current_spend: Final = proxy_server.get_current_spend

    async def from_db(prisma_client: object, counter_key: str) -> float | None:
        reads.db_reads += 1
        if isinstance(reads.db_spend, Exception):
            raise reads.db_spend
        return reads.db_spend

    async def counted_current_spend(counter_key: str, fallback_spend: float, **kwargs: object) -> float:
        reads.current_spend_calls += 1
        return await current_spend(counter_key, fallback_spend, **kwargs)  # pyright: ignore[reportArgumentType]  # passes the keywords the caller gave

    async def cold(counter_key: str, source_cache_key: str) -> None:
        return None

    monkeypatch.setattr(SpendCounterReseed, "from_db", staticmethod(from_db))
    monkeypatch.setattr(proxy_server, "get_current_spend", counted_current_spend)
    monkeypatch.setattr(proxy_server, "ensure_spend_counter_initialized", cold)
    return reads


def _reset_reads(reads: _SpendReads) -> None:
    reads.db_reads = 0
    reads.current_spend_calls = 0


@pytest.mark.asyncio
async def test_a_warm_counter_is_read_by_the_native_without_asking_python(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: Final = _count_spend_reads(monkeypatch, 0.0)
    identity: Final = Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0})
    python: Final = await _python_admit(store, identity)
    assert reads.current_spend_calls == 2
    store.flush_local()
    _reset_reads(reads)
    native: Final = await _native_admit(store, identity, "call-warm")

    assert (reads.current_spend_calls, reads.db_reads) == (0, 0)
    assert native.counters == python.counters
    assert _entries(native.reservation) == _entries(python.reservation)


COLD_SPEND: Final[Mapping[str, float]] = {"room_left": 1.0, "reseeded_spend_refuses": 100.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("db_spend", COLD_SPEND.values(), ids=COLD_SPEND.keys())
async def test_a_cold_counter_is_reseeded_through_python_like_python(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch, db_spend: float
) -> None:
    reads: Final = _count_spend_reads(monkeypatch, db_spend)
    identity: Final = Identity(team_budget=100.0, spend={TEAM: 2.0})
    python: Final = await _python_admit(store, identity)
    python_reads: Final = (reads.current_spend_calls, reads.db_reads)
    store.flush_local()
    _reset_reads(reads)
    native: Final = await _native_admit(store, identity, "call-cold")

    assert python_reads == (2, 1)
    assert (reads.current_spend_calls, reads.db_reads) == (1, 1)
    assert _refusal_fields(native.refusal) == _refusal_fields(python.refusal)
    assert native.counters == python.counters
    assert _entries(native.reservation) == _entries(python.reservation)


@pytest.mark.asyncio
async def test_a_reseed_that_raises_fails_the_admission_like_python(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _count_spend_reads(monkeypatch, ConnectionError("database unreachable"))
    identity: Final = Identity(team_budget=100.0, spend={TEAM: 2.0})
    with pytest.raises(ConnectionError) as python:
        _ = await _python_admit(store, identity)
    python_counters: Final = _snapshot(store)
    store.flush_local()
    with pytest.raises(ConnectionError) as native:
        _ = await _native_admit(store, identity, "call-raises")

    assert str(native.value) == str(python.value)
    assert _snapshot(store) == python_counters


@pytest.mark.asyncio
async def test_python_reserves_again_once_the_native_owner_leaves_the_callbacks(
    store: SpendCounterStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity: Final = Identity(team_budget=100.0, spend={KEY: 1.0, TEAM: 2.0})
    python: Final = await _python_admit(store, identity)
    store.flush_local()
    _ = _install_native_owner(store)
    monkeypatch.setattr(litellm, "callbacks", [])
    _seed(store, identity)

    reservation: Final = await reserve_budget_for_request(**_arguments(identity))  # pyright: ignore[reportArgumentType]  # the shared argument table is typed per key above

    assert _entries(reservation) == _entries(python.reservation)
    assert _snapshot(store) == python.counters
    native_budgets.install_native_budgets(store, litellm.callbacks)
    assert native_budgets.native_owner_reserves()
