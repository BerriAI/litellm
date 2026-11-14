import uuid
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway
from tests.integration._support.database import read_rows

pytestmark: Final = pytest.mark.shares_worker_state

_ROUTE: Final = "/management/v1"


@pytest.mark.parametrize("budget", ({"max_budget": 10}, None), ids=("with_overall_budget", "override_only"))
def test_deleting_a_customer_team_deletes_its_override_budget_row(
    gateway: Gateway, budget: dict[str, int] | None
) -> None:
    suffix: Final = uuid.uuid4().hex
    org, team = f"delete-org-{suffix}", f"delete-team-{suffix}"
    with gateway.scenario() as scenario:
        created_org: Final = gateway.request(
            "POST",
            f"{_ROUTE}/customer_organizations",
            {"customer_org_id": org, "default_budgets": [{"name": "base", "max_budget": 10}]},
        )
        assert created_org.status_code == 201, created_org.text
        scenario.cleanups.callback(gateway.request, "DELETE", f"{_ROUTE}/customer_organizations/{org}?cascade=true")
        created_team: Final = gateway.request(
            "POST",
            f"{_ROUTE}/customer_teams",
            {
                "customer_team_id": team,
                "parent_org_id": org,
                "budget_override": {"max_budget": 50},
                **({"budget": budget} if budget is not None else {}),
            },
        )
        assert created_team.status_code == 201, created_team.text
        body: Final = JSON_OBJECT.validate_json(created_team.content)
        owned: Final = tuple(
            value for value in (body["budget_id"], body["budget_override_id"]) if isinstance(value, str)
        )
        assert len(owned) == (1 if budget is None else 2)
        stored: Final = 'SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id = ANY(string_to_array(%s, \',\'))'
        assert len(read_rows(stored, (",".join(owned),))) == len(owned)
        deleted: Final = gateway.request("DELETE", f"{_ROUTE}/customer_teams/{team}")
        assert deleted.status_code == 204, deleted.text
        assert read_rows(stored, (",".join(owned),)) == []
