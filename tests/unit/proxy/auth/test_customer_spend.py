from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.proxy.auth.customer_hierarchy_auth import check_customer_hierarchy_budget
from litellm.proxy.auth.customer_spend import current_customer_spend as read_customer_spend
from litellm.proxy.auth.customer_spend import customer_spend_counter, settle_customer_spend
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


async def current_customer_spend(hierarchy, store, prisma=None):
    from litellm.proxy import proxy_server

    with pytest.MonkeyPatch.context() as context:
        context.setattr(proxy_server, "spend_counter_store", store)
        context.setattr(proxy_server, "prisma_client", prisma)
        return await read_customer_spend(hierarchy, proxy_server.get_current_spend)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("customer_org", "customer_team", "customer_member"))
async def test_a_lost_counter_reseeds_committed_spend_without_refreshing_the_customer_snapshot(scope: str) -> None:
    from tests.unit.proxy.db.test_spend_counter_reseed import _FakeFindUniqueTable

    row: Final = SimpleNamespace(spend=2.5, budget_reset_at=None)
    table: Final = _FakeFindUniqueTable(row)
    table_name: Final = {
        "customer_org": "litellm_customerorganizationtable",
        "customer_team": "litellm_customerteamtable",
        "customer_member": "litellm_customermembertable",
    }[scope]
    prisma: Final = SimpleNamespace(db=SimpleNamespace(**{table_name: table}))
    snapshot: Final = CustomerHierarchyAuth(
        customer_org_id="tenant", scopes=(CustomerScopeAuth(key=scope, entity_id="id:with:colons", max_budget=2),)
    )
    shared: Final[dict[str, object]] = {}
    first: Final = FakeSpendCounters(redis=shared).store()
    second: Final = FakeSpendCounters(redis=shared).store()
    for store in (first, second):
        loaded: Final = await current_customer_spend(snapshot, store, prisma)
        with pytest.raises(litellm.BudgetExceededError):
            check_customer_hierarchy_budget(loaded)
    assert snapshot.scopes[0].spend == 0
    assert len(table.where_clauses) == 1
    await first.delete((customer_spend_counter(snapshot.scopes[0]),))
    reseeded: Final = await current_customer_spend(snapshot, second, prisma)
    with pytest.raises(litellm.BudgetExceededError):
        check_customer_hierarchy_budget(reseeded)
    assert reseeded.scopes[0].spend == row.spend
    assert len(table.where_clauses) == 2


@pytest.mark.asyncio
async def test_a_database_snapshot_ahead_of_settlement_does_not_count_the_pending_charge_twice() -> None:
    store: Final = FakeSpendCounters().store()
    initial: Final = CustomerHierarchyAuth(
        customer_org_id="tenant", scopes=(CustomerScopeAuth(key="customer_org", entity_id="tenant"),)
    )
    _ = await current_customer_spend(initial, store)
    persisted: Final = initial.model_copy(update={"scopes": (initial.scopes[0].model_copy(update={"spend": 0.5}),)})
    _ = await current_customer_spend(persisted, store)
    await settle_customer_spend({"user_api_key_customer_hierarchy": initial.model_dump(mode="json")}, 0.5, store)
    assert (await current_customer_spend(persisted, store)).scopes[0].spend == 0.5


@pytest.mark.asyncio
async def test_member_b_settlement_blocks_cached_member_a_and_an_alias_only_request_on_another_worker() -> None:
    shared: Final[dict[str, object]] = {}
    settling: Final = FakeSpendCounters(redis=shared).store()
    reading: Final = FakeSpendCounters(redis=shared).store()
    organization: Final = CustomerScopeAuth(key="customer_org", entity_id="tenant", max_budget=1)
    member_a: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        customer_member_id="a",
        scopes=(organization, CustomerScopeAuth(key="customer_member", entity_id="a")),
    )
    member_b: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        customer_member_id="b",
        scopes=(organization, CustomerScopeAuth(key="customer_member", entity_id="b")),
    )
    alias_only: Final = CustomerHierarchyAuth(customer_org_id="tenant", scopes=(organization,))
    _ = await current_customer_spend(member_a, reading)
    _ = await current_customer_spend(alias_only, reading)
    await settle_customer_spend({"user_api_key_customer_hierarchy": member_b.model_dump(mode="json")}, 1.25, settling)
    for cached in (member_a, alias_only):
        current: Final = await current_customer_spend(cached, reading)
        with pytest.raises(litellm.BudgetExceededError) as rejected:
            check_customer_hierarchy_budget(current)
        assert rejected.value.entity_type == "customer_org"
        assert rejected.value.entity_id == "tenant"
    assert (await current_customer_spend(member_a, reading)).scopes[1].spend == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("redis", (True, False))
async def test_settlement_counts_once_on_each_scope_without_modifying_the_cached_snapshot(redis: bool) -> None:
    handle: Final = FakeSpendCounters(redis={} if redis else None)
    store: Final = handle.store()
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        scopes=(
            CustomerScopeAuth(key="customer_org", entity_id="tenant", spend=1, max_budget=5),
            CustomerScopeAuth(key="customer_team", entity_id="team", spend=2),
            CustomerScopeAuth(key="customer_member", entity_id="member", spend=3),
        ),
    )
    await settle_customer_spend({"user_api_key_customer_hierarchy": hierarchy.model_dump(mode="json")}, 0.5, store)
    current: Final = await current_customer_spend(hierarchy, store)
    assert tuple(scope.spend for scope in current.scopes) == (1.5, 2.5, 3.5)
    assert tuple(scope.spend for scope in hierarchy.scopes) == (1.0, 2.0, 3.0)


@pytest.mark.asyncio
async def test_budget_reset_uses_a_new_counter_even_when_an_old_request_settles_late() -> None:
    store: Final = FakeSpendCounters().store()
    old: Final = CustomerHierarchyAuth(
        customer_org_id="tenant",
        scopes=(CustomerScopeAuth(key="customer_org", entity_id="tenant", spend=2, max_budget=1),),
    )
    reset: Final = old.model_copy(
        update={
            "scopes": (
                old.scopes[0].model_copy(
                    update={"spend": 0.0, "budget_reset_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
                ),
            )
        }
    )
    await settle_customer_spend({"user_api_key_customer_hierarchy": old.model_dump(mode="json")}, 0.5, store)
    current: Final = await current_customer_spend(reset, store)
    check_customer_hierarchy_budget(current)
    assert current.scopes[0].spend == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata", ({}, {"user_api_key_customer_hierarchy": {}}, {"user_api_key_customer_hierarchy": "bad"})
)
async def test_missing_or_invalid_customer_metadata_does_not_charge_a_counter(metadata: dict[str, object]) -> None:
    handle: Final = FakeSpendCounters()
    await settle_customer_spend(metadata, 1, handle.store())
    assert handle.redis == {}


@pytest.mark.asyncio
async def test_zero_cost_does_not_initialize_customer_counters() -> None:
    handle: Final = FakeSpendCounters()
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="tenant", scopes=())
    await settle_customer_spend(
        {"user_api_key_customer_hierarchy": hierarchy.model_dump(mode="json")}, 0, handle.store()
    )
    assert handle.redis == {}
