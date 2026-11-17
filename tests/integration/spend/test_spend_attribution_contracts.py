from contextlib import ExitStack
from itertools import chain
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    delete_key_if_present,
    eventually,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows

MEMBER_BUDGET: Final = 0.0000001
SPEND_LANDING_SECONDS: Final = 70


def test_spend_log_of_an_org_team_key_records_org_and_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization(models=[model])
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        request_id: Final = string_value(gateway.chat(model, key=key)["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT team_id, metadata::text AS metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (request_id,),
            ),
            lambda found: len(found) == 1,
            seconds=SPEND_LANDING_SECONDS,
        )
        metadata: Final = JSON_OBJECT.validate_json(string_value(rows[0]["metadata"]))
        assert metadata["user_api_key_org_id"] == organization
        assert metadata["user_api_key_team_id"] == team
        assert rows[0]["team_id"] == team


def _team_memberships(team: JsonValue, user_id: str) -> list[dict[str, JsonValue]]:
    entries: Final = object_value(team).get("team_memberships") or []
    assert isinstance(entries, list)
    return [object_value(entry) for entry in entries if object_value(entry).get("user_id") == user_id]


def _membership(gateway: Gateway, user_id: str, team_id: str) -> dict[str, JsonValue]:
    teams: Final = gateway.get("/user/info", {"user_id": user_id})["teams"]
    assert isinstance(teams, list)
    memberships: Final = list(
        chain.from_iterable(
            _team_memberships(team, user_id) for team in teams if object_value(team)["team_id"] == team_id
        )
    )
    assert len(memberships) == 1, teams
    return memberships[0]


def test_team_member_budget_blocks_the_member_after_spend_lands(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        created: Final = gateway.post(
            "/user/new",
            {"user_id": f"integration-{model}", "team_id": team, "models": [model], "max_budget": 10.0},
        )
        user: Final = string_value(created["user_id"])
        key: Final = string_value(created["key"])
        with ExitStack() as member_cleanups:
            member_cleanups.callback(scenario.delete_user, user)
            member_cleanups.callback(delete_key_if_present, gateway, key)
            gateway.post("/team/member_update", {"team_id": team, "user_id": user, "max_budget_in_team": MEMBER_BUDGET})
            membership: Final = _membership(gateway, user, team)
            scenario.cleanups.callback(scenario.delete_budget, string_value(membership["budget_id"]))
            scenario.cleanups.push(member_cleanups.pop_all())
        assert object_value(membership["litellm_budget_table"])["max_budget"] == MEMBER_BUDGET
        assert (
            gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "first"}]},
                key=key,
            ).status_code
            == 200
        )
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s', (user, team)
            ),
            lambda rows: len(rows) == 1 and isinstance(spend := rows[0]["spend"], float) and spend >= MEMBER_BUDGET,
            seconds=SPEND_LANDING_SECONDS,
        )
        blocked: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "second"}]},
            key=key,
        )
        assert blocked.status_code != 200, blocked.text
        assert "Budget has been exceeded" in blocked.text
