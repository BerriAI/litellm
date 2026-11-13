import asyncio
from datetime import datetime, time, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

import litellm
import litellm.proxy.proxy_server as ps
from litellm.accounting.projection import project
from litellm.models.budget import BudgetLimit
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import (
    check_customer_hierarchy_windows,
    customer_member_cache_key,
    customer_org_alias_cache_key,
    get_customer_scopes_for_accounting,
    resolve_customer_hierarchy,
)
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.hooks.proxy_track_cost_callback import (
    ProxyDBLogger,
    _update_database_and_spend_counters,
    _update_database_and_spend_counters_in_batch,
)
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.spend_tracking.budget_reservation import (
    _counter_to_reservation_entry,
    _reserve_counters,
    customer_window_counters,
    estimate_request_max_cost,
    release_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.redis.stores.spend_counters.types import parse_customer_window_counter
from litellm.responses.streaming_iterator import drain_logging_tasks
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerHierarchyIds,
    CustomerRequestIdentity,
    CustomerScopeAuth,
)
from litellm.types.utils import ModelResponse, Usage
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

LIMITS: Final = (
    BudgetLimit(budget_duration="1mo", max_budget=10),
    BudgetLimit(budget_duration="12mo", max_budget=100),
)


@pytest.fixture
def accounting_state(monkeypatch):
    backend: Final = FakeSpendCounters()
    writer: Final = DBSpendUpdateWriter()
    prisma: Final = MagicMock()
    prisma.spend_log_transactions = []
    prisma._spend_log_transactions_lock = asyncio.Lock()
    prisma.tool_usage_transactions = []
    prisma._tool_usage_transactions_lock = asyncio.Lock()
    prisma.db.litellm_budgetwindowspend.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "spend_counter_store", backend.store())
    monkeypatch.setattr(ps, "proxy_logging_obj", SimpleNamespace(db_spend_update_writer=writer))
    monkeypatch.setattr(ps, "prisma_client", prisma)
    return backend, writer, prisma


@pytest.mark.parametrize("scope", ("customer_org", "customer_team", "customer_member"))
def test_calendar_windows_use_exact_timezone_boundary_and_independent_entity(scope):
    settings: Final = BudgetResetSettings(timezone="America/New_York", reset_time_of_day=time(12, 34, 56))
    now: Final = datetime(2026, 11, 13, tzinfo=timezone.utc)
    monthly, annual = customer_window_counters(scope, "member:one/@", LIMITS, now=now, reset_settings=settings)
    assert monthly.window_start == datetime(2026, 11, 1, 17, 34, 56, tzinfo=timezone.utc)
    assert monthly.reset_at == datetime(2026, 12, 1, 17, 34, 56, tzinfo=timezone.utc)
    assert annual.window_start == datetime(2026, 1, 1, 17, 34, 56, tzinfo=timezone.utc)
    assert annual.reset_at == datetime(2027, 1, 1, 17, 34, 56, tzinfo=timezone.utc)
    assert parse_customer_window_counter(monthly.counter_key) == (scope, "member:one/@", "1mo", monthly.window_start)
    other: Final = customer_window_counters(scope, "member:two", LIMITS, now=now, reset_settings=settings)
    assert {monthly.counter_key, annual.counter_key}.isdisjoint(counter.counter_key for counter in other)
    assert LIMITS[0].reset_at is None


def test_shared_template_reset_is_not_advanced_by_a_member():
    expired: Final = BudgetLimit(
        budget_duration="1mo", max_budget=10, reset_at=datetime(2026, 2, 1, tzinfo=timezone.utc)
    )
    november: Final = datetime(2026, 11, 13, tzinfo=timezone.utc)
    first: Final = customer_window_counters("customer_member", "one", (expired,), now=november)
    second: Final = customer_window_counters("customer_member", "two", (expired,), now=november)
    assert first[0].window_start == second[0].window_start == datetime(2026, 11, 1, tzinfo=timezone.utc)
    assert expired.reset_at == datetime(2026, 2, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("ledger", (False, True))
async def test_settlement_charges_each_window_once_and_selects_one_durable_writer(accounting_state, ledger):
    backend, writer, _ = accounting_state
    counters: Final = customer_window_counters("customer_member", "member", LIMITS)
    entries: Final = [_counter_to_reservation_entry(counter, 2.0) for counter in counters]
    for counter in counters:
        await ps.spend_counter_store.set(counter.counter_key, 5.0)
    assert await _reserve_counters(counters, entries, 2.0) == (7.0, 7.0)
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
    assert tuple(backend.redis[counter.counter_key] for counter in counters) == (5.75, 5.75)
    assert reservation["finalized"] is True
    rows: Final = await writer.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
    assert len(rows) == (0 if ledger else 2)
    assert all(row["spend"] == 0.75 and row["entity_id"] == "member" for row in rows)
    windows: Final = await ps.accounting_window_keys(None, None, counters)
    assert {window["window_duration"] for window in windows} == {"1mo", "12mo"}
    assert len([command for command in backend.commands if command.startswith("MGET")]) == 1


@pytest.mark.asyncio
async def test_completion_after_reset_refunds_old_hold_and_charges_current_period(accounting_state):
    backend, writer, _ = accounting_state
    old: Final = customer_window_counters(
        "customer_member", "member", LIMITS, now=datetime(2000, 6, 15, tzinfo=timezone.utc)
    )
    current: Final = customer_window_counters("customer_member", "member", LIMITS)
    entries: Final = [_counter_to_reservation_entry(counter, 2.0) for counter in old]
    for counter in old:
        await ps.spend_counter_store.set(counter.counter_key, 5.0)
    await _reserve_counters(old, entries, 2.0)
    reservation: Final = {"reserved_cost": 2.0, "entries": entries, "finalized": False}
    await ps.increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.75,
        budget_reservation=reservation,
        customer_counters=current,
    )
    assert tuple(backend.redis[counter.counter_key] for counter in old) == (5.0, 5.0)
    assert tuple(backend.redis[counter.counter_key] for counter in current) == (0.75, 0.75)
    assert tuple(entry["applied_adjustment"] for entry in entries) == (-2.0, -2.0)
    rows: Final = await writer.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
    assert {row["window_start"] for row in rows} == {
        counter.window_start.replace(tzinfo=None).isoformat(timespec="microseconds") for counter in current
    }


@pytest.mark.asyncio
async def test_eviction_between_reserve_and_settle_restores_durable_floor(accounting_state):
    backend, _, prisma = accounting_state
    counters: Final = customer_window_counters("customer_member", "member", LIMITS[:1])
    counter: Final = counters[0]
    prisma.db.litellm_budgetwindowspend.find_unique.return_value = SimpleNamespace(
        window_start=counter.window_start, spend=5.0
    )
    entries: Final = [_counter_to_reservation_entry(counter, 2.0)]
    await ps.spend_counter_store.set(counter.counter_key, 5.0)
    await _reserve_counters(counters, entries, 2.0)
    await ps.spend_counter_store.delete((counter.counter_key,))
    await ps.increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.75,
        budget_reservation={"reserved_cost": 2.0, "entries": entries, "finalized": False},
        customer_counters=counters,
    )
    assert backend.redis[counter.counter_key] == 5.75


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("customer_org", "customer_team", "customer_member"))
async def test_cold_counter_reseeds_exact_scope_without_scanning_logs(accounting_state, scope):
    _, _, prisma = accounting_state
    counter: Final = customer_window_counters(scope, "member:one", LIMITS)[0]
    prisma.db.litellm_budgetwindowspend.find_unique.return_value = SimpleNamespace(
        window_start=counter.window_start, spend=4.0
    )
    assert await SpendCounterReseed.from_db(prisma, counter.counter_key) == 4.0
    prisma.db.litellm_budgetwindowspend.find_unique.assert_awaited_once_with(
        where={
            "entity_type_entity_id_window_duration": {
                "entity_type": scope,
                "entity_id": "member:one",
                "window_duration": "1mo",
            }
        }
    )
    assert not prisma.db.litellm_spendlogs.group_by.called


@pytest.mark.asyncio
async def test_missing_period_is_zero_but_database_failure_remains_unknown(accounting_state):
    _, _, prisma = accounting_state
    counter: Final = customer_window_counters("customer_member", "member", LIMITS)[0]
    assert await SpendCounterReseed.from_db(prisma, counter.counter_key) == 0.0
    prisma.db.litellm_budgetwindowspend.find_unique.side_effect = ConnectionError("unavailable")
    assert await SpendCounterReseed.from_db(prisma, counter.counter_key) is None


@pytest.mark.asyncio
async def test_redis_failure_never_admits_from_incomplete_local_spend(accounting_state):
    backend, _, prisma = accounting_state
    counter: Final = customer_window_counters("customer_member", "member", LIMITS)[0]
    backend.fail = True
    backend.memory[counter.counter_key] = 0.0
    assert (
        await SpendCounterReseed.coalesced_window(
            prisma,
            ps.spend_counter_store,
            counter.counter_key,
            counter.entity_type,
            counter.entity_id,
            counter.window_duration,
            counter.window_start,
        )
        is None
    )
    prisma.db.litellm_budgetwindowspend.find_unique.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ledger", (False, True))
@pytest.mark.parametrize("disable_logs", (False, True))
@pytest.mark.parametrize("status", ("success", "failure"))
async def test_real_cost_callback_keeps_customer_scalar_and_window_cost_once(
    accounting_state,
    monkeypatch,
    ledger,
    disable_logs,
    status,
):
    backend, writer, prisma = accounting_state
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": ledger})
    monkeypatch.setattr(ps, "disable_spend_logs", disable_logs)
    now: Final = datetime.now(timezone.utc)
    counters: Final = customer_window_counters("customer_member", "member", LIMITS, now=now)
    response: Final = ModelResponse(model="gpt-4o-mini", usage=Usage(prompt_tokens=2, completion_tokens=3))
    prisma.get_request_status.return_value = status
    kwargs: Final = {
        "model": "gpt-4o-mini",
        "call_type": "acompletion",
        "litellm_call_id": "window-cost",
        "litellm_params": {
            "metadata": {
                "status": status,
                "user_api_key_customer_member_id": "member",
                "user_api_key_customer_team_id": "team",
                "user_api_key_customer_org_id": "org",
            }
        },
    }
    assert await _update_database_and_spend_counters_in_batch(
        proxy_logging_obj=ps.proxy_logging_obj,
        spend_writer=writer,
        increment_spend_counters=ps.increment_spend_counters,
        user_api_key=None,
        user_id=None,
        end_user_id=None,
        team_id=None,
        org_id=None,
        kwargs=kwargs,
        completion_response=response,
        start_time=now,
        end_time=now,
        response_cost=0.75,
        budget_reservation=None,
        request_tags=None,
        model_access_groups=None,
        project_id=None,
        customer_counters=counters,
    )
    await drain_logging_tasks()
    assert tuple(backend.redis[counter.counter_key] for counter in counters) == (0.75, 0.75)
    window_rows: Final = await writer.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
    if ledger:
        assert window_rows == ()
        assert len(prisma.spend_log_transactions) == 1
        contributions: Final = project(prisma.spend_log_transactions[0]["accounting_payload"])
        window_contributions: Final = tuple(row for row in contributions if row.table == "LiteLLM_BudgetWindowSpend")
        assert len(window_contributions) == 2
        assert all(dict(row.increments)["spend"] == 0.75 for row in window_contributions)
        scalar: Final = tuple(
            row
            for row in contributions
            if row.table
            in (
                "LiteLLM_CustomerOrganizationTable",
                "LiteLLM_CustomerTeamTable",
                "LiteLLM_CustomerMemberTable",
            )
        )
        assert len(scalar) == 3
        assert all(dict(row.increments)["spend"] == 0.75 for row in scalar)
    else:
        assert len(window_rows) == 2
        assert all(row["spend"] == 0.75 for row in window_rows)
        scalar_rows: Final = await writer.spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions()
        assert scalar_rows["customer_org_list_transactions"] == {"org": 0.75}
        assert scalar_rows["customer_team_list_transactions"] == {"team": 0.75}
        assert scalar_rows["customer_member_list_transactions"] == {"member": 0.75}


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
            valid_token=auth,
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
        backend.redis[counter.counter_key] == pytest.approx(accepted[0]["reserved_cost"]) for counter in counters
    )
    await release_budget_reservation(accepted[0])
    assert all(backend.redis[counter.counter_key] == pytest.approx(0.0) for counter in counters)


@pytest.mark.asyncio
async def test_outer_callback_resolves_cached_hierarchy_into_accounting(accounting_state, monkeypatch):
    backend, writer, prisma = accounting_state
    cache: Final = UserApiKeyCache()
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="org",
        customer_member_id="member",
        scopes=(
            CustomerScopeAuth(key="customer_org", entity_id="org"),
            CustomerScopeAuth(key="customer_member", entity_id="member", budget_limits=LIMITS),
        ),
    )
    await cache.async_set_cache(key=customer_member_cache_key("member"), value=hierarchy)
    monkeypatch.setattr(ps, "user_api_key_cache", cache)
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": True})
    monkeypatch.setattr(ps, "disable_spend_logs", True)
    now: Final = datetime.now(timezone.utc)
    assert await _update_database_and_spend_counters(
        proxy_logging_obj=ps.proxy_logging_obj,
        spend_writer=writer,
        increment_spend_counters=ps.increment_spend_counters,
        user_api_key=None,
        user_id=None,
        end_user_id=None,
        team_id=None,
        org_id=None,
        kwargs={
            "model": "gpt-4o-mini",
            "call_type": "acompletion",
            "litellm_call_id": "resolved-cost",
            "litellm_params": {
                "metadata": {"user_api_key_customer_org_id": "org", "user_api_key_customer_member_id": "member"}
            },
        },
        completion_response=ModelResponse(model="gpt-4o-mini", usage=Usage(prompt_tokens=2, completion_tokens=3)),
        start_time=now,
        end_time=now,
        response_cost=0.75,
        budget_reservation=None,
    )
    counters: Final = customer_window_counters("customer_member", "member", LIMITS)
    assert tuple(backend.redis[counter.counter_key] for counter in counters) == (0.75, 0.75)
    assert len(prisma.spend_log_transactions) == 1
    windows: Final = tuple(
        row
        for row in project(prisma.spend_log_transactions[0]["accounting_payload"])
        if row.table == "LiteLLM_BudgetWindowSpend"
    )
    assert len(windows) == 2
    prisma.db.litellm_customermembertable.find_unique.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("moved_member", (False, True))
async def test_accounting_retains_stamped_parent_ids(accounting_state, moved_member):
    _, _, prisma = accounting_state
    cache: Final = UserApiKeyCache()
    if moved_member:
        await cache.async_set_cache(
            key=customer_member_cache_key("member"),
            value=CustomerHierarchyAuth(
                customer_org_id="new-org",
                customer_team_id="new-team",
                customer_member_id="member",
                scopes=(
                    CustomerScopeAuth(key="customer_org", entity_id="new-org", budget_limits=LIMITS),
                    CustomerScopeAuth(key="customer_team", entity_id="new-team", budget_limits=LIMITS),
                    CustomerScopeAuth(key="customer_member", entity_id="member", budget_limits=LIMITS),
                ),
            ),
        )
    record: Final = SimpleNamespace(
        model_dump=lambda: {
            "spend": 1.0,
            "budget": {"budget_limits": [limit.model_dump() for limit in LIMITS]},
        }
    )
    prisma.db.litellm_customerorganizationtable.find_unique = AsyncMock(return_value=record)
    prisma.db.litellm_customerteamtable.find_unique = AsyncMock(return_value=record)
    scopes: Final = await get_customer_scopes_for_accounting(
        CustomerHierarchyIds("org", "team" if moved_member else None, "member" if moved_member else None),
        prisma,
        cache,
    )
    assert {scope.entity_id for scope in scopes} == ({"org", "team", "member"} if moved_member else {"org"})
    assert all(scope.budget_limits == LIMITS for scope in scopes)
    prisma.db.litellm_customerorganizationtable.find_unique.assert_awaited_once_with(
        where={"customer_org_id": "org"},
        include={"budget": True},
    )


@pytest.mark.asyncio
async def test_auth_rejects_redis_outage_with_503(accounting_state):
    backend, _, prisma = accounting_state
    backend.fail = True
    counter: Final = customer_window_counters("customer_member", "member", LIMITS)[0]
    backend.memory[counter.counter_key] = 0.0
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="org",
        customer_member_id="member",
        scopes=(CustomerScopeAuth(key="customer_member", entity_id="member", budget_limits=LIMITS),),
    )
    with pytest.raises(ProxyException) as error:
        await check_customer_hierarchy_windows(hierarchy)
    assert error.value.code == "503"


@pytest.mark.asyncio
@pytest.mark.parametrize("member", (False, True))
@pytest.mark.parametrize("ledger", (False, True))
@pytest.mark.parametrize("failure", (False, True))
async def test_admitted_windows_are_charged_during_policy_outage(accounting_state, monkeypatch, member, ledger, failure):
    backend, writer, prisma = accounting_state
    scopes: Final = (
        CustomerScopeAuth(key="customer_org", entity_id="org", budget_limits=LIMITS),
        *(
            (
                CustomerScopeAuth(key="customer_team", entity_id="team", budget_limits=LIMITS),
                CustomerScopeAuth(key="customer_member", entity_id="member", budget_limits=LIMITS),
            )
            if member
            else ()
        ),
    )
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=customer_member_cache_key("member") if member else customer_org_alias_cache_key("org-alias"),
        value=CustomerHierarchyAuth(
            customer_org_id="org",
            customer_team_id="team" if member else None,
            customer_member_id="member" if member else None,
            scopes=scopes,
        ),
    )
    for table in (
        prisma.db.litellm_customerorganizationtable,
        prisma.db.litellm_customerteamtable,
        prisma.db.litellm_customermembertable,
    ):
        table.find_unique = AsyncMock(side_effect=ConnectionError("policy database unavailable"))
        table.find_many = AsyncMock(side_effect=ConnectionError("policy database unavailable"))
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": ledger})
    monkeypatch.setattr(ps, "disable_spend_logs", True)
    admitted: Final = await resolve_customer_hierarchy(
        identity=CustomerRequestIdentity(customer_member_id="member")
        if member
        else CustomerRequestIdentity(customer_org_alias="org-alias"),
        prisma_client=prisma,
        user_api_key_cache=cache,
        llm_router=None,
        request_model="gpt-4o-mini",
        route="/chat/completions",
    )
    auth: Final = UserAPIKeyAuth(customer_hierarchy=admitted)
    request_data: Final = await add_litellm_data_to_request(
        data={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        request=Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []}),
        user_api_key_dict=auth,
        proxy_config=ps.ProxyConfig(),
        general_settings={},
    )
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache())
    now: Final = datetime.now(timezone.utc)
    usage: Final = Usage(prompt_tokens=2, completion_tokens=3)
    prisma.get_request_status.return_value = "failure" if failure else "success"
    if failure:
        await ProxyDBLogger().async_post_call_failure_hook(
            request_data={**request_data, "combined_usage_object": usage, "response_cost": 0.75},
            original_exception=RuntimeError("provider disconnected after billing"),
            user_api_key_dict=auth,
        )
    else:
        assert await _update_database_and_spend_counters(
            proxy_logging_obj=ps.proxy_logging_obj,
            spend_writer=writer,
            increment_spend_counters=ps.increment_spend_counters,
            user_api_key=None,
            user_id=None,
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={
                "model": "gpt-4o-mini",
                "call_type": "acompletion",
                "litellm_call_id": "policy-outage",
                "litellm_params": {"metadata": request_data["metadata"]},
            },
            completion_response=ModelResponse(model="gpt-4o-mini", usage=usage),
            start_time=now,
            end_time=now,
            response_cost=0.75,
            budget_reservation=None,
        )
    await drain_logging_tasks()
    for scope in scopes:
        for counter in customer_window_counters(scope.key, scope.entity_id, LIMITS):
            assert backend.redis[counter.counter_key] == 0.75
    queued: Final = await writer.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
    if ledger:
        assert queued == ()
        assert len(prisma.spend_log_transactions) == 1
        contributions: Final = project(prisma.spend_log_transactions[0]["accounting_payload"])
        rows: Final = tuple(row for row in contributions if row.table == "LiteLLM_BudgetWindowSpend")
        assert len(rows) == 2 * len(scopes)
        assert {dict(row.key)["entity_id"] for row in rows} == {scope.entity_id for scope in scopes}
        assert {dict(row.key)["window_duration"] for row in rows} == {"1mo", "12mo"}
        assert all(dict(row.increments)["spend"] == 0.75 for row in rows)
        scalar: Final = tuple(row for row in contributions if row.table.startswith("LiteLLM_Customer"))
        assert len(scalar) == len(scopes)
        assert all(dict(row.increments)["spend"] == 0.75 for row in scalar)
    else:
        assert len(queued) == 2 * len(scopes)
        assert {row["entity_id"] for row in queued} == {scope.entity_id for scope in scopes}
        assert {row["window_duration"] for row in queued} == {"1mo", "12mo"}
        assert all(row["spend"] == 0.75 for row in queued)
        scalar_queued: Final = await writer.spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions()
        for scope in scopes:
            assert scalar_queued[f"{scope.key}_list_transactions"] == {scope.entity_id: 0.75}
