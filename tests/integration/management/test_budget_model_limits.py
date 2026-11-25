import uuid
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, object_value
from tests.integration._support.database import read_rows

pytestmark: Final = pytest.mark.shares_worker_state

_JSON_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _budget_columns(budget_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT model_tpm_limit, model_rpm_limit FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
    )
    assert len(rows) == 1, rows
    return rows[0]


def _budget_info(gateway: Gateway, budget_id: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", "/budget/info", {"budgets": [budget_id]})
    assert response.status_code == 200, response.text
    entries: Final = _JSON_OBJECTS.validate_json(response.content)
    assert len(entries) == 1, entries
    return entries[0]


def test_budget_stores_and_returns_per_model_tpm_and_rpm_limits(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(model_tpm_limit={"gpt-4o": 400000}, model_rpm_limit={"gpt-4o": 200})

        info: Final = _budget_info(gateway, budget_id)
        assert info["model_tpm_limit"] == {"gpt-4o": 400000}, info
        assert info["model_rpm_limit"] == {"gpt-4o": 200}, info
        row: Final = _budget_columns(budget_id)
        assert row == {"model_tpm_limit": {"gpt-4o": 400000}, "model_rpm_limit": {"gpt-4o": 200}}, row

        response: Final = gateway.request(
            "POST", "/budget/update", {"budget_id": budget_id, "model_tpm_limit": {"gpt-4o": 100000, "gpt-4o-mini": 50}}
        )
        assert response.status_code == 200, response.text
        updated: Final = _budget_columns(budget_id)
        assert updated == {
            "model_tpm_limit": {"gpt-4o": 100000, "gpt-4o-mini": 50},
            "model_rpm_limit": {"gpt-4o": 200},
        }, updated


def test_customer_created_with_inline_per_model_limits_gets_a_budget_that_carries_them(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        customer: Final = f"integration-customer-{uuid.uuid4().hex}"
        created: Final = gateway.post(
            "/customer/new",
            {"user_id": customer, "model_tpm_limit": {"gpt-4o": 400000}, "model_rpm_limit": {"gpt-4o": 200}},
        )
        scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [customer]})
        budget_id: Final = created["budget_id"]
        assert isinstance(budget_id, str) and budget_id, created
        scenario.cleanups.callback(scenario.delete_budget, budget_id)

        response: Final = gateway.request("GET", f"/customer/info?end_user_id={customer}", None)
        assert response.status_code == 200, response.text
        budget: Final = object_value(
            TypeAdapter(dict[str, JsonValue]).validate_json(response.content)["litellm_budget_table"]
        )
        assert budget["model_tpm_limit"] == {"gpt-4o": 400000}, budget
        assert budget["model_rpm_limit"] == {"gpt-4o": 200}, budget


@pytest.mark.parametrize("field", ["model_tpm_limit", "model_rpm_limit"])
def test_budget_rejects_a_negative_per_model_limit(gateway: Gateway, field: str) -> None:
    budget_id: Final = f"integration-budget-{uuid.uuid4().hex}"
    response: Final = gateway.request("POST", "/budget/new", {"budget_id": budget_id, field: {"gpt-4o": -1}})
    assert response.status_code == 422, response.text
    assert read_rows('SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)) == []
