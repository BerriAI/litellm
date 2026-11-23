"""GET /audit lists the row a team or key update wrote, with the state before the update as an object."""

import os
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from tests.integration._support.client import Gateway, eventually, list_value, object_value
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.process import graceful_stop_seconds, owned_proxy_process
from pydantic import JsonValue

OWNED_PROXY_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
UPDATED_ROWS_SQL: Final = """SELECT id FROM "LiteLLM_AuditLog" WHERE object_id = %s AND action = 'updated'"""
STORE_BEFORE_VALUE_AS_JSON_TEXT_SQL: Final = """UPDATE "LiteLLM_AuditLog" SET before_value = to_jsonb(before_value::text)
WHERE object_id = %s AND action = 'updated'"""


def _audit_page(candidate: Gateway, object_id: str) -> httpx.Response:
    eventually(lambda: read_rows(UPDATED_ROWS_SQL, (object_id,)), lambda rows: len(rows) == 1, seconds=30)
    return candidate.request("GET", "/audit", params={"object_id": object_id})


def _before_values_of_updates(page: httpx.Response) -> list[JsonValue]:
    logs: Final = (object_value(log) for log in list_value(object_value(page.json())["audit_logs"]))
    return [log["before_value"] for log in logs if log["action"] == "updated"]


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
def test_the_audit_page_lists_a_team_update_and_a_key_update_with_the_state_before_as_an_object(
    gateway: Gateway, tmp_path: Path
) -> None:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: audit logs are an enterprise feature"
    with (
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"], "LITELLM_STORE_AUDIT_LOGS": "true"},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        candidate: Final = owned.gateway
        team: Final = scenario.team()
        token: Final = sha256(scenario.key().encode()).hexdigest()
        candidate.post("/team/update", {"team_id": team, "max_budget": 7})
        candidate.post("/key/update", {"key": token, "max_budget": 3})

        team_page: Final = _audit_page(candidate, team)
        key_page: Final = _audit_page(candidate, token)

        assert (team_page.status_code, key_page.status_code) == (200, 200), f"{team_page.text} {key_page.text}"
        team_before: Final = [object_value(value) for value in _before_values_of_updates(team_page)]
        key_before: Final = [object_value(value) for value in _before_values_of_updates(key_page)]
        assert [before["team_id"] for before in team_before] == [team], team_page.text
        assert [before["token"] for before in key_before] == [token], key_page.text


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
def test_the_audit_page_lists_an_update_row_whose_state_before_was_stored_as_json_text(
    gateway: Gateway, tmp_path: Path
) -> None:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: audit logs are an enterprise feature"
    with (
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"], "LITELLM_STORE_AUDIT_LOGS": "true"},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        candidate: Final = owned.gateway
        team: Final = scenario.team()
        candidate.post("/team/update", {"team_id": team, "max_budget": 7})
        stored: Final = _audit_page(candidate, team)
        write_rows(STORE_BEFORE_VALUE_AS_JSON_TEXT_SQL, (team,))

        page: Final = candidate.request("GET", "/audit", params={"object_id": team})
        row_id: Final = read_rows(UPDATED_ROWS_SQL, (team,))[0]["id"]
        by_id: Final = candidate.request("GET", f"/audit/{row_id}")

        assert (page.status_code, by_id.status_code) == (200, 200), f"{page.text} {by_id.text}"
        assert _before_values_of_updates(page) == _before_values_of_updates(stored), page.text
        assert object_value(by_id.json())["before_value"] == _before_values_of_updates(stored)[0], by_id.text
