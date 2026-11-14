import uuid
from datetime import datetime
from typing import Final

import pytest

from tests.integration._support.client import Gateway, Scenario, object_value

pytestmark: Final = pytest.mark.shares_worker_state


def _budget_reset_at(scenario: Scenario, budget_duration: str) -> datetime:
    budget_id: Final = f"integration-budget-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post(
        "/budget/new", {"budget_id": budget_id, "max_budget": 10, "budget_duration": budget_duration}
    )
    scenario.cleanups.callback(scenario.gateway.post, "/budget/delete", {"id": budget_id})
    return datetime.fromisoformat(str(created["budget_reset_at"]))


def _customer_reset_at(scenario: Scenario, budget_duration: str) -> datetime:
    customer: Final = f"integration-customer-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post(
        "/customer/new", {"user_id": customer, "max_budget": 10, "budget_duration": budget_duration}
    )
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [customer]})
    return datetime.fromisoformat(str(object_value(created["litellm_budget_table"])["budget_reset_at"]))


@pytest.mark.parametrize("budget_duration", ["12mo", "1mo", "30d", "7d", "1d"])
def test_customer_new_budget_resets_on_the_same_schedule_as_budget_new(gateway: Gateway, budget_duration: str) -> None:
    with gateway.scenario() as scenario:
        earliest: Final = _budget_reset_at(scenario, budget_duration)
        customer: Final = _customer_reset_at(scenario, budget_duration)
        latest: Final = _budget_reset_at(scenario, budget_duration)
        assert earliest <= customer <= latest
