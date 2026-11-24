from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.models.budget import BudgetLimit
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyIds,
    CustomerScopeKey,
    app_budget_limits,
    customer_hierarchy_ids_from_request,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

_IDS: Final = {
    "user_api_key_customer_member_id": "m1",
    "user_api_key_customer_team_id": "t1",
    "user_api_key_customer_org_id": "o1",
}


def test_ids_are_read_from_whichever_metadata_bucket_the_proxy_stamped() -> None:
    expected: Final = CustomerHierarchyIds(customer_org_id="o1", customer_team_id="t1", customer_member_id="m1")
    assert customer_hierarchy_ids_from_request({"litellm_params": {"metadata": _IDS}}) == expected
    assert (
        customer_hierarchy_ids_from_request({"litellm_params": {"metadata": {"note": "x"}, "litellm_metadata": _IDS}})
        == expected
    )
    assert customer_hierarchy_ids_from_request({"litellm_params": {"metadata": {"note": "x"}}}) is None


def test_a_request_that_named_only_an_organization_charges_the_organization_alone() -> None:
    ids: Final = customer_hierarchy_ids_from_request(
        {
            "litellm_params": {
                "metadata": {"user_api_key_customer_org_id": "o1", "user_api_key_customer_member_id": None}
            }
        }
    )
    assert ids == CustomerHierarchyIds(customer_org_id="o1")
    assert ids is not None
    assert tuple(entity_id for _, entity_id in ids.spend_entities()) == ("o1",)
    assert ids.entities() == {"customer_member_id": None, "customer_team_id": None, "customer_org_id": "o1"}
    member_only: Final = {"litellm_params": {"metadata": {"user_api_key_customer_member_id": "m1"}}}
    assert customer_hierarchy_ids_from_request(member_only) is None


@pytest.mark.parametrize(
    ("scalar_maximum", "window_maximum", "expected"), ((10, 100, 10), (100, 10, 10), (0, 100, 0), (100, 0, 0))
)
def test_overlapping_app_limits_keep_the_stricter_cap_and_explicit_reset(
    scalar_maximum: float, window_maximum: float, expected: float
) -> None:
    reset: Final = datetime(2026, 12, 1, 13, tzinfo=timezone.utc)
    window: Final = BudgetLimit(budget_duration="1mo", max_budget=window_maximum, reset_at=reset)
    budget: Final = Budget(max_budget=scalar_maximum, budget_duration="1mo", budget_limits=(window,))

    limits: Final = app_budget_limits(budget)

    assert len(limits) == 1
    assert limits[0].max_budget == expected
    assert limits[0].reset_at == reset
    assert window.max_budget == window_maximum


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("app", "app_customer_org", "app_customer_member"))
async def test_overlapping_app_limits_project_one_charge_per_window(scope: CustomerScopeKey) -> None:
    from litellm.accounting.projection import project
    from litellm.proxy.proxy_server import accounting_window_keys
    from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters

    now: Final = datetime(2026, 11, 18, tzinfo=timezone.utc)
    budget: Final = Budget(
        max_budget=100,
        budget_duration="1mo",
        budget_limits=(BudgetLimit(budget_duration="1mo", max_budget=10),),
    )
    counters: Final = customer_window_counters(scope, "app:customer", app_budget_limits(budget), now=now)
    windows: Final = await accounting_window_keys(None, None, counters)
    contributions: Final = project(
        {"spend": 0.25, "startTime": now.isoformat(), "metadata": {"accounting_windows": windows}}
    )
    charges: Final = tuple(row for row in contributions if row.table == "LiteLLM_BudgetWindowSpend")

    assert tuple(dict(row.increments)["spend"] for row in charges) == (0.25,)
    assert len(counters) == len({counter.counter_key for counter in counters}) == 1


def test_distinct_app_budget_durations_keep_their_independent_limits() -> None:
    annual: Final = BudgetLimit(budget_duration="12mo", max_budget=1000)
    budget: Final = Budget(max_budget=100, budget_duration="1mo", budget_limits=(annual,))

    limits: Final = app_budget_limits(budget)

    assert tuple((limit.budget_duration, limit.max_budget) for limit in limits) == (("12mo", 1000), ("1mo", 100))
