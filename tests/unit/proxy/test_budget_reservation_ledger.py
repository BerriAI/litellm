from typing import Final

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_EndUserTable, LiteLLM_OrganizationTable, LiteLLM_TagTable
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, model_access_group_cache_key
from litellm.redis.stores.spend_counters import SpendCounterStore, model_access_group_counter
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget
from tests.unit.proxy import test_budget_reservation as scenarios
from tests.unit.proxy.spend_counters_ledger_twin import force_ledger_windows, ledger_scenarios, rust_memory_store

globals().update(ledger_scenarios(scenarios))


@pytest.fixture(autouse=True)
def ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    force_ledger_windows(monkeypatch)


# The four counter scenarios below drove FakeSpendCounters through ``spend_counter_state``. They are redefined under
# the same name on the Rust in-memory store the ledger's counter step runs on, with the same assertions


@pytest.fixture
def spend_counter_state(monkeypatch: pytest.MonkeyPatch) -> tuple[SpendCounterStore, UserApiKeyCache]:
    key_cache: Final = UserApiKeyCache()
    monkeypatch.setattr(ps, "user_api_key_cache", key_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    return rust_memory_store(monkeypatch), key_cache


@pytest.mark.asyncio
async def test_should_seed_and_update_end_user_and_tag_counters_without_reservation(
    spend_counter_state: tuple[SpendCounterStore, UserApiKeyCache],
) -> None:
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key="end_user_id:customer-1",
        value=LiteLLM_EndUserTable(
            user_id="customer-1", blocked=False, spend=4.0, litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0)
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key="tag:paid-tag", value=LiteLLM_TagTable(tag_name="paid-tag", spend=7.0).model_dump()
    )
    await key_cache.async_set_cache(
        key="tag:other-tag", value=LiteLLM_TagTable(tag_name="other-tag", spend=2.0).model_dump()
    )

    await ps.increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.50,
        end_user_id="customer-1",
        tags=["paid-tag", "paid-tag", "other-tag", ""],
    )

    assert counter_cache.get_local("spend:end_user:customer-1") == 4.50
    assert counter_cache.get_local("spend:tag:paid-tag") == 7.50
    assert counter_cache.get_local("spend:tag:other-tag") == 2.50


def _org(organization_id: str, budget: LiteLLM_BudgetTable | None) -> LiteLLM_OrganizationTable:
    return LiteLLM_OrganizationTable(
        organization_id=organization_id,
        organization_alias="shared-org",
        budget_id="org-budget-id",
        spend=2.0,
        models=[],
        created_by="test",
        updated_by="test",
        litellm_budget_table=budget,
    )


@pytest.mark.asyncio
async def test_should_seed_org_counter_from_with_budget_cache(
    spend_counter_state: tuple[SpendCounterStore, UserApiKeyCache],
) -> None:
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key="org_id:org-counter-with-budget:with_budget",
        value=_org("org-counter-with-budget", LiteLLM_BudgetTable(max_budget=10.0)).model_dump(),
    )

    await ps.increment_spend_counters(
        token=None, team_id=None, user_id=None, org_id="org-counter-with-budget", response_cost=0.25
    )

    assert counter_cache.get_local("spend:org:org-counter-with-budget") == 2.25


@pytest.mark.asyncio
async def test_should_seed_org_counter_from_plain_org_cache(
    spend_counter_state: tuple[SpendCounterStore, UserApiKeyCache],
) -> None:
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(key="org_id:org-counter-plain", value=_org("org-counter-plain", None).model_dump())

    await ps.increment_spend_counters(
        token=None, team_id=None, user_id=None, org_id="org-counter-plain", response_cost=0.25
    )

    assert counter_cache.get_local("spend:org:org-counter-plain") == 2.25


@pytest.mark.asyncio
async def test_model_access_group_counter_accumulates_across_calls_without_a_reservation(
    spend_counter_state: tuple[SpendCounterStore, UserApiKeyCache],
) -> None:
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key=model_access_group_cache_key("premium"),
        value=ModelAccessGroupBudget(access_group_name="premium", spend=1.0, max_budget=25.0),
        model_type=ModelAccessGroupBudget,
    )
    counter_key: Final = model_access_group_counter("premium")

    await ps.increment_spend_counters(
        token=None, team_id=None, user_id=None, response_cost=0.25, model_access_groups=["premium"]
    )
    assert counter_cache.get_local(counter_key) == 1.25

    await ps.increment_spend_counters(
        token=None, team_id=None, user_id=None, response_cost=0.75, model_access_groups=["premium", "premium", ""]
    )
    assert counter_cache.get_local(counter_key) == 2.0
    assert counter_cache.get_local(model_access_group_counter("")) is None
