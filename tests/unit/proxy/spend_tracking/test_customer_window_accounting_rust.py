import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm.models.budget import BudgetLimit
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.budget_reservation import (
    BudgetCounter,
    customer_window_counters,
    estimate_request_max_cost,
    release_budget_reservation,
)
from litellm.proxy.spend_tracking.spend_counter_batch import PendingSpendIncrement
from litellm.proxy.utils import ProxyLogging
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerScopeAuth,
)
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from tests.unit.proxy.spend_tracking.native_budgets_replica import (
    RespRedis,
    install_native_owner,
    reserve_budget_for_request,
    resp_store,
)

pytestmark = pytest.mark.usefixtures("native_budgets_on")

LIMITS: Final = (
    BudgetLimit(budget_duration="1mo", max_budget=10),
    BudgetLimit(budget_duration="12mo", max_budget=100),
)


@pytest.fixture
def accounting_state(monkeypatch):
    backend: Final = SpendCounterStore(connect(None).spend_counters())
    writer: Final = DBSpendUpdateWriter()
    prisma: Final = MagicMock()
    prisma.spend_log_transactions = []
    prisma._spend_log_transactions_lock = asyncio.Lock()
    prisma.tool_usage_transactions = []
    prisma._tool_usage_transactions_lock = asyncio.Lock()
    prisma.db.litellm_budgetwindowspend.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "spend_counter_store", backend)
    monkeypatch.setattr(ps, "proxy_logging_obj", SimpleNamespace(db_spend_update_writer=writer))
    monkeypatch.setattr(ps, "prisma_client", prisma)
    monkeypatch.setattr(SpendCounterReseed, "_locks", OrderedDict())
    return backend, writer, prisma


@pytest.fixture
def redis_accounting_state(accounting_state, monkeypatch):
    _, writer, prisma = accounting_state
    redis: Final = RespRedis(values={})
    with resp_store(redis) as store:
        monkeypatch.setattr(ps, "spend_counter_store", store)
        _ = install_native_owner(store)
        yield redis, writer, prisma


def _held_entry(counter: BudgetCounter, reserved_cost: float) -> dict[str, float | str]:
    return {
        "counter_key": counter.counter_key,
        "entity_type": counter.entity_type,
        "entity_id": counter.entity_id,
        "reserved_cost": reserved_cost,
        "applied_adjustment": 0.0,
        **({"window_reset_at": counter.reset_at.isoformat()} if counter.reset_at is not None else {}),
    }


async def _hold(counters: Sequence[BudgetCounter], reserved_cost: float) -> tuple[float | None, ...]:
    return await ps.run_spend_counter_pipeline(
        pending=tuple(
            PendingSpendIncrement(counter_key=counter.counter_key, increment=reserved_cost) for counter in counters
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ledger", (False, True))
async def test_settlement_charges_each_window_once_and_selects_one_durable_writer(redis_accounting_state, ledger):
    backend, writer, _ = redis_accounting_state
    counters: Final = customer_window_counters("customer_member", "member", LIMITS)
    entries: Final = [_held_entry(counter, 2.0) for counter in counters]
    for counter in counters:
        await ps.spend_counter_store.set(counter.counter_key, 5.0)
    assert await _hold(counters, 2.0) == (7.0, 7.0)
    reservation: Final = {"reserved_cost": 2.0, "entries": entries, "finalized": False}
    backend.commands.clear()
    await ps.increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.75,
        budget_reservation=reservation,
        customer_counters=counters,
        windows_from_ledger=ledger,
    )
    assert tuple(backend.values[counter.counter_key] for counter in counters) == (5.75, 5.75)
    assert reservation["finalized"] is True
    rows: Final = await writer.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
    assert len(rows) == (0 if ledger else 2)
    assert all(row["spend"] == 0.75 and row["entity_id"] == "member" for row in rows)
    windows: Final = await ps.accounting_window_keys(None, None, counters)
    assert {window["window_duration"] for window in windows} == {"1mo", "12mo"}
    assert backend.count("MGET") == 1


@pytest.mark.asyncio
async def test_completion_after_reset_refunds_old_hold_and_charges_current_period(redis_accounting_state):
    backend, writer, _ = redis_accounting_state
    old: Final = customer_window_counters(
        "customer_member", "member", LIMITS, now=datetime(2000, 6, 15, tzinfo=timezone.utc)
    )
    current: Final = customer_window_counters("customer_member", "member", LIMITS)
    entries: Final = [_held_entry(counter, 2.0) for counter in old]
    for counter in old:
        await ps.spend_counter_store.set(counter.counter_key, 5.0)
    _ = await _hold(old, 2.0)
    reservation: Final = {"reserved_cost": 2.0, "entries": entries, "finalized": False}
    await ps.increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.75,
        budget_reservation=reservation,
        customer_counters=current,
    )
    assert tuple(backend.values[counter.counter_key] for counter in old) == (5.0, 5.0)
    assert tuple(backend.values[counter.counter_key] for counter in current) == (0.75, 0.75)
    assert tuple(entry["applied_adjustment"] for entry in entries) == (-2.0, -2.0)
    rows: Final = await writer.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
    assert {row["window_start"] for row in rows} == {
        counter.window_start.replace(tzinfo=None).isoformat(timespec="microseconds") for counter in current
    }


@pytest.mark.asyncio
async def test_eviction_between_reserve_and_settle_restores_durable_floor(redis_accounting_state):
    backend, _, prisma = redis_accounting_state
    counters: Final = customer_window_counters("customer_member", "member", LIMITS[:1])
    counter: Final = counters[0]
    prisma.db.litellm_budgetwindowspend.find_unique.return_value = SimpleNamespace(
        window_start=counter.window_start, spend=5.0
    )
    entries: Final = [_held_entry(counter, 2.0)]
    await ps.spend_counter_store.set(counter.counter_key, 5.0)
    _ = await _hold(counters, 2.0)
    await ps.spend_counter_store.delete((counter.counter_key,))
    await ps.increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.75,
        budget_reservation={"reserved_cost": 2.0, "entries": entries, "finalized": False},
        customer_counters=counters,
    )
    assert backend.values[counter.counter_key] == 5.75


@pytest.mark.asyncio
@pytest.mark.parametrize("limiting_duration", ("1mo", "12mo"))
async def test_concurrent_requests_obey_both_caps_with_existing_reservation_engine(accounting_state, limiting_duration):
    backend, _, prisma = accounting_state
    request_body: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1000,
    }
    estimated: Final = estimate_request_max_cost(request_body, "/chat/completions", None)
    assert estimated is not None and estimated > 0
    limits: Final = tuple(
        BudgetLimit(budget_duration=duration, max_budget=estimated * (1.5 if duration == limiting_duration else 100))
        for duration in ("1mo", "12mo")
    )
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="org",
        customer_member_id="member",
        scopes=(CustomerScopeAuth(key="customer_member", entity_id="member", budget_limits=limits),),
    )
    auth: Final = UserAPIKeyAuth().model_copy(update={"customer_hierarchy": hierarchy})
    cache: Final = UserApiKeyCache()
    logging: Final = ProxyLogging(user_api_key_cache=cache)

    async def reserve():
        return await reserve_budget_for_request(
            request_body=dict(request_body),
            route="/chat/completions",
            llm_router=None,
            valid_token=auth.model_copy(),
            team_object=None,
            user_object=None,
            prisma_client=prisma,
            user_api_key_cache=cache,
            proxy_logging_obj=logging,
            fail_closed_budget_enforcement=True,
        )

    results: Final = await asyncio.gather(reserve(), reserve(), return_exceptions=True)
    accepted: Final = tuple(result for result in results if isinstance(result, dict))
    rejected: Final = tuple(result for result in results if isinstance(result, litellm.BudgetExceededError))
    assert len(accepted) == len(rejected) == 1
    counters: Final = customer_window_counters("customer_member", "member", limits)
    assert all(
        backend.get_local(counter.counter_key) == pytest.approx(accepted[0]["reserved_cost"]) for counter in counters
    )
    await release_budget_reservation(accepted[0])
    assert all(backend.get_local(counter.counter_key) == pytest.approx(0.0) for counter in counters)
