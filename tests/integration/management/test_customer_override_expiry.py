import uuid
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value

pytestmark: Final = pytest.mark.shares_worker_state

_ROUTE: Final = "/management/v1"


def _created(scenario: Scenario, collection: str, body: dict[str, object], identity: str) -> None:
    response: Final = scenario.gateway.request("POST", f"{_ROUTE}/{collection}", body)  # pyright: ignore[reportArgumentType]  # JsonValue body
    assert response.status_code == 201, response.text
    scenario.cleanups.callback(scenario.gateway.request, "DELETE", f"{_ROUTE}/{collection}/{identity}")


@pytest.mark.parametrize("collection", ("customer_members", "customer_teams"))
def test_an_override_applies_after_a_past_expiry_was_sent_without_one(gateway: Gateway, collection: str) -> None:
    suffix: Final = uuid.uuid4().hex
    org, team, row = f"override-org-{suffix}", f"override-team-{suffix}", f"override-row-{suffix}"
    past_expiry: Final = {"budget_override_expires_at": "2020-01-01T00:00:00Z"}
    with gateway.scenario() as scenario:
        _created(
            scenario,
            "customer_organizations",
            {"customer_org_id": org, "default_budgets": [{"name": "base", "max_budget": 10}]},
            org,
        )
        _created(scenario, "customer_teams", {"customer_team_id": team, "parent_org_id": org}, team)
        _created(
            scenario,
            collection,
            {"customer_member_id": row, "parent_team_ids": [team], **past_expiry}
            if collection == "customer_members"
            else {"customer_team_id": row, "parent_org_id": org, **past_expiry},
            row,
        )
        scenario.cleanups.callback(gateway.request, "PATCH", f"{_ROUTE}/{collection}/{row}", {"budget_override": None})
        overridden: Final = gateway.request(
            "PATCH", f"{_ROUTE}/{collection}/{row}", {"budget_override": {"max_budget": 50}}
        )
        assert overridden.status_code == 200, overridden.text
        body: Final = JSON_OBJECT.validate_json(overridden.content)
        assert body["budget_override_expires_at"] is None
        assert object_value(body["effective_budget"])["max_budget"] == 50
