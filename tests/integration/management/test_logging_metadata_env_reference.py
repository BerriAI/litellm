import json
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration.management._batch_file_caps import hashed
from pydantic import JsonValue

BULK_ROUTE: Final = "/management/v1/users/bulk"
SPACE_ID: Final = "arize_space_id"
API_KEY: Final = "arize_api_key"
REFERENCE: Final = "os.environ/ARIZE_SPACE_ID_TENANT_1"
LITERAL_VARS: Final[Mapping[str, JsonValue]] = {API_KEY: "arize-literal-key", SPACE_ID: "arize-literal-space"}


def _logging(field: str) -> dict[str, JsonValue]:
    return {
        "logging": [
            {
                "callback_name": "arize",
                "callback_type": "success",
                "callback_vars": {**LITERAL_VARS, field: REFERENCE},
            }
        ]
    }


def _refusal(field: str) -> str:
    return f"Callback param '{field}' (from key/team callback metadata) contains an 'os.environ/' reference."


def _assert_refused(response: httpx.Response, field: str) -> None:
    assert response.status_code == 400, response.text
    assert _refusal(field) in string_value(object_value(response.json()["error"])["message"]), response.text


def _team_metadata(team: str) -> JsonValue:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def _key_metadata(token: str) -> JsonValue:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(token),))
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def _drop_created_key(scenario: Scenario, response: httpx.Response) -> None:
    if response.status_code == 200 and response.json().get("key"):
        scenario.cleanups.callback(scenario.delete_key, string_value(response.json()["key"]))


FIELDS: Final = (pytest.param(SPACE_ID, id="space-id"), pytest.param(API_KEY, id="api-key"))


@pytest.mark.parametrize("field", FIELDS)
def test_team_update_refuses_an_env_reference_in_logging_metadata(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"owner": "env-reference-audit"})
        refused: Final = gateway.request("POST", "/team/update", {"team_id": team, "metadata": _logging(field)})
        _assert_refused(refused, field)
        assert _team_metadata(team) == {"owner": "env-reference-audit"}


@pytest.mark.parametrize("field", FIELDS)
def test_team_new_refuses_an_env_reference_in_logging_metadata(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        alias: Final = f"env-reference-{uuid.uuid4().hex}"
        refused: Final = gateway.request("POST", "/team/new", {"team_alias": alias, "metadata": _logging(field)})
        if refused.status_code == 200:
            scenario.cleanups.callback(scenario.delete_team, string_value(refused.json()["team_id"]))
        _assert_refused(refused, field)
        assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_alias = %s', (alias,)) == []


@pytest.mark.parametrize("field", FIELDS)
def test_key_generate_refuses_an_env_reference_in_logging_metadata(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        alias: Final = f"env-reference-{uuid.uuid4().hex}"
        refused: Final = gateway.request("POST", "/key/generate", {"key_alias": alias, "metadata": _logging(field)})
        _drop_created_key(scenario, refused)
        _assert_refused(refused, field)
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s', (alias,)) == []


@pytest.mark.parametrize("field", FIELDS)
def test_key_update_refuses_an_env_reference_in_logging_metadata(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"owner": "env-reference-audit"})
        refused: Final = gateway.request("POST", "/key/update", {"key": key, "metadata": _logging(field)})
        _assert_refused(refused, field)
        assert _key_metadata(key) == {"owner": "env-reference-audit"}


@pytest.mark.parametrize("field", FIELDS)
def test_user_new_refuses_an_env_reference_in_its_keys_logging_metadata(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        user: Final = f"env-reference-{uuid.uuid4().hex}"
        refused: Final = gateway.request("POST", "/user/new", {"user_id": user, "metadata": _logging(field)})
        if refused.status_code == 200:
            scenario.cleanups.callback(scenario.delete_user, user)
        _drop_created_key(scenario, refused)
        _assert_refused(refused, field)
        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,)) == []
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id = %s', (user,)) == []


def test_user_new_without_a_key_refuses_an_env_reference(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user: Final = f"env-reference-{uuid.uuid4().hex}"
        refused: Final = gateway.request(
            "POST", "/user/new", {"user_id": user, "metadata": _logging(SPACE_ID), "auto_create_key": False}
        )
        if refused.status_code == 200:
            scenario.cleanups.callback(scenario.delete_user, user)
        _assert_refused(refused, SPACE_ID)
        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,)) == []


@pytest.mark.parametrize("field", FIELDS)
def test_user_update_refuses_an_env_reference_in_logging_metadata(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user(metadata={"owner": "env-reference-audit"})
        refused: Final = gateway.request("POST", "/user/update", {"user_id": user, "metadata": _logging(field)})
        _assert_refused(refused, field)
        rows: Final = read_rows('SELECT metadata FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,))
        assert rows == [{"metadata": {"owner": "env-reference-audit"}}]


def _callback_settings(field: str) -> dict[str, JsonValue]:
    return {
        "callback_settings": {
            "success_callback": ["arize"],
            "callback_vars": {**LITERAL_VARS, field: REFERENCE},
        }
    }


@pytest.mark.parametrize("field", FIELDS)
def test_team_writes_refuse_an_env_reference_in_callback_settings(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"owner": "env-reference-audit"})
        updated: Final = gateway.request("POST", "/team/update", {"team_id": team, "metadata": _callback_settings(field)})
        _assert_refused(updated, field)
        patched: Final = gateway.request("PATCH", f"/team/{team}", {"metadata": _callback_settings(field)})
        _assert_refused(patched, field)
        assert "already stores" not in patched.text
        assert _team_metadata(team) == {"owner": "env-reference-audit"}
        alias: Final = f"env-reference-{uuid.uuid4().hex}"
        created: Final = gateway.request("POST", "/team/new", {"team_alias": alias, "metadata": _callback_settings(field)})
        if created.status_code == 200:
            scenario.cleanups.callback(scenario.delete_team, string_value(created.json()["team_id"]))
        _assert_refused(created, field)
        assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_alias = %s', (alias,)) == []


def test_team_key_bulk_update_reports_the_key_with_an_env_reference_as_failed(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        refused_key: Final = scenario.key(team_id=team, metadata={"owner": "env-reference-audit"})
        plain_key: Final = scenario.key(team_id=team)
        response: Final = gateway.request(
            "POST",
            "/team/key/bulk_update",
            {
                "team_id": team,
                "key_ids": [hashed(refused_key)],
                "update_fields": {"metadata": _logging(SPACE_ID)},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        assert body["successful_updates"] == [], response.text
        failed: Final = body["failed_updates"]
        assert isinstance(failed, list) and len(failed) == 1, response.text
        failure: Final = object_value(failed[0])
        assert failure["key"] == hashed(refused_key), response.text
        assert _refusal(SPACE_ID) in string_value(failure["failed_reason"]), response.text
        assert _key_metadata(refused_key) == {"owner": "env-reference-audit"}
        applied: Final = gateway.request(
            "POST",
            "/team/key/bulk_update",
            {"team_id": team, "key_ids": [hashed(plain_key)], "update_fields": {"metadata": {"owner": "x"}}},
        )
        assert applied.status_code == 200, applied.text
        assert _key_metadata(plain_key) == {"owner": "x"}


def test_a_stored_env_reference_blocks_metadata_saves_until_it_is_removed(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        legacy: Final = _logging(SPACE_ID)
        write_rows('UPDATE "LiteLLM_TeamTable" SET metadata = %s::jsonb WHERE team_id = %s', (json.dumps(legacy), team))
        patched: Final = gateway.request("PATCH", f"/team/{team}", {"metadata": {"cost_center": "1"}})
        _assert_refused(patched, SPACE_ID)
        assert f"Team {team} already stores this reference" in patched.text, patched.text
        assert _team_metadata(team) == legacy
        unrelated: Final = gateway.request("POST", "/team/update", {"team_id": team, "tpm_limit": 5000})
        assert unrelated.status_code == 200, unrelated.text
        assert _team_metadata(team) == legacy
        cleared: Final = gateway.request("POST", "/team/update", {"team_id": team, "metadata": {"logging": []}})
        assert cleared.status_code == 200, cleared.text
        assert object_value(_team_metadata(team))["logging"] == []
        repatched: Final = gateway.request("PATCH", f"/team/{team}", {"metadata": {"cost_center": "1"}})
        assert repatched.status_code == 200, repatched.text
        assert object_value(_team_metadata(team))["cost_center"] == "1"


@pytest.mark.parametrize("field", FIELDS)
def test_bulk_user_creation_refuses_only_the_row_with_an_env_reference(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        refused_user, plain_user = (f"env-reference-{uuid.uuid4().hex}" for _ in range(2))
        response: Final = gateway.request(
            "POST",
            BULK_ROUTE,
            {
                "users": [
                    {"user_id": refused_user, "metadata": _logging(field), "auto_create_key": False},
                    {"user_id": plain_user, "auto_create_key": True},
                ]
            },
        )
        if response.status_code == 200:
            for row in response.json()["data"]:
                if row["success"]:
                    scenario.cleanups.callback(scenario.delete_user, string_value(row["user_id"]))
                if row["key"]:
                    scenario.cleanups.callback(scenario.delete_key, string_value(row["key"]))
        assert response.status_code == 200, response.text
        refused, plain = response.json()["data"]
        assert (refused["user_id"], refused["success"], refused["key"]) == (refused_user, False, None), response.text
        assert _refusal(field) in string_value(refused["error"]), response.text
        assert (plain["success"], plain["error"]) == (True, None), response.text
        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (refused_user,)) == []


def test_literal_arize_values_still_save_on_every_route(gateway: Gateway) -> None:
    logging: Final = {
        "logging": [{"callback_name": "arize", "callback_type": "success", "callback_vars": dict(LITERAL_VARS)}]
    }
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        updated_team: Final = gateway.request("POST", "/team/update", {"team_id": team, "metadata": logging})
        assert updated_team.status_code == 200, updated_team.text
        created_team: Final = gateway.request("POST", "/team/new", {"metadata": logging})
        if created_team.status_code == 200:
            scenario.cleanups.callback(scenario.delete_team, string_value(created_team.json()["team_id"]))
        assert created_team.status_code == 200, created_team.text
        generated: Final = gateway.request("POST", "/key/generate", {"metadata": logging})
        _drop_created_key(scenario, generated)
        assert generated.status_code == 200, generated.text
        key: Final = scenario.key()
        updated_key: Final = gateway.request("POST", "/key/update", {"key": key, "metadata": logging})
        assert updated_key.status_code == 200, updated_key.text
        user: Final = f"env-reference-{uuid.uuid4().hex}"
        created_user: Final = gateway.request("POST", "/user/new", {"user_id": user, "metadata": logging})
        if created_user.status_code == 200:
            scenario.cleanups.callback(scenario.delete_user, user)
        _drop_created_key(scenario, created_user)
        assert created_user.status_code == 200, created_user.text
        stored: Final = object_value(_team_metadata(team))
        assert [entry["callback_name"] for entry in stored["logging"]] == ["arize"]
