from datetime import datetime, time, timezone
from typing import Final, cast

import pytest
from prisma.models import LiteLLM_BudgetWindowSpend

from litellm.models.budget import BudgetLimit

from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.proxy.management_helpers.customer_budget_usage import customer_budget_usage
from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, CustomerMember
from tests.unit.proxy.management_helpers.test_customer_hierarchy import Database, Record

NOW: Final = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)
SETTINGS: Final = BudgetResetSettings(timezone="America/Los_Angeles", reset_time_of_day=time(9))


def member(member_id: str) -> CustomerMember:
    return CustomerMember(
        customer_member_id=member_id,
        parent_org_id="org",
        parent_team_ids=("team",),
        spend=40,
        effective_budget=Budget(
            max_budget=50,
            budget_duration="1mo",
            budget_limits=(
                BudgetLimit(budget_duration="1mo", max_budget=10),
                BudgetLimit(budget_duration="12mo", max_budget=100),
            ),
        ),
        created_at=NOW,
        created_by="admin",
        updated_at=NOW,
        updated_by="admin",
    )


@pytest.mark.asyncio
async def test_recorded_usage_matches_enforced_windows_and_keeps_shared_members_independent() -> None:
    database: Final = Database()
    first: Final = member("member-a")
    second: Final = member("member-b")
    counters: Final = customer_window_counters(
        "customer_member", "member-a", first.effective_budget.budget_limits or (), now=NOW, reset_settings=SETTINGS
    )
    assert counters[0].window_start == datetime(2026, 11, 1, 17, tzinfo=timezone.utc)
    database.litellm_budgetwindowspend.rows = {
        "monthly": Record(
            entity_type="customer_member", entity_id="member-a", window_duration="1mo",
            window_start=counters[0].window_start, spend=8,
        ),
        "yearly": Record(
            entity_type="customer_member", entity_id="member-a", window_duration="12mo",
            window_start=counters[1].window_start, spend=62,
        ),
        "another-scope": Record(
            entity_type="customer_team", entity_id="member-b", window_duration="1mo",
            window_start=counters[0].window_start, spend=999,
        ),
        "expired": Record(
            entity_type="customer_member", entity_id="member-b", window_duration="1mo",
            window_start=datetime(2026, 10, 1, 16, tzinfo=timezone.utc), spend=100,
        ),
    }
    result: Final = await customer_budget_usage(
        (first, second),
        cast(TableActions[LiteLLM_BudgetWindowSpend], database.litellm_budgetwindowspend),
        NOW,
        SETTINGS,
    )
    assert result[0].budget_usage is not None
    assert tuple((item.spend, item.max_budget) for item in result[0].budget_usage) == ((40, 50), (8, 10), (62, 100))
    assert tuple((item.window_start, item.reset_at) for item in result[0].budget_usage[1:]) == tuple(
        (counter.window_start, counter.reset_at) for counter in counters
    )
    assert result[1].budget_usage is not None
    assert tuple(item.spend for item in result[1].budget_usage) == (40, 0, 0)
