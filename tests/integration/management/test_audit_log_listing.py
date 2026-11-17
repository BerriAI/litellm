"""GET /audit lists the row every management write leaves, with the state before and after as objects."""

import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from tests.integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows
from tests.integration._support.process import graceful_stop_seconds, owned_proxy_process
from pydantic import JsonValue

OWNED_PROXY_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
UPDATED_ROWS_SQL: Final = """SELECT id FROM "LiteLLM_AuditLog" WHERE object_id = %s AND action = 'updated'"""


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


AUDITED: Final = {"LITELLM_STORE_AUDIT_LOGS": "true"}
SECRET: Final = "sk-audit-secret-0123456789abcdef"
SECRET_MASKED: Final = "sk-a" + "*" * 24 + "cdef"
TEXT_THAT_LOOKS_LIKE_JSON: Final[dict[str, JsonValue]] = {
    "count": 7,
    "tags": ["a", "b"],
    "empty": "",
    "note": "x" * 5120,
    "digits": "123",
    "not_a_number": "NaN",
    "listing": "[1]",
    "nested": '{"a": 1}',
}
STORED_SQL: Final = """
SELECT id, jsonb_typeof(before_value) AS before_type, jsonb_typeof(updated_values) AS updated_type,
       before_value, updated_values
FROM "LiteLLM_AuditLog" WHERE object_id = %s AND action = %s ORDER BY updated_at, id
"""
IDS_SQL: Final = """SELECT id FROM "LiteLLM_AuditLog" WHERE object_id = %s"""


@pytest.fixture(scope="module")
def audited(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: audit logs are an enterprise feature"
    directory: Final = tmp_path_factory.mktemp("audited").resolve()
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {"DATABASE_URL": os.environ["DATABASE_URL"], **AUDITED},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as owned,
    ):
        yield owned.gateway


def _stored(object_id: str, action: str, count: int = 1) -> list[dict[str, JsonValue]]:
    return eventually(lambda: read_rows(STORED_SQL, (object_id, action)), lambda rows: len(rows) == count, seconds=30)


def _listed(page: httpx.Response) -> list[dict[str, JsonValue]]:
    assert page.status_code == 200, page.text
    return [object_value(log) for log in list_value(object_value(page.json())["audit_logs"])]


def _hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _updated_team(scenario: Scenario, **fields: JsonValue) -> str:
    team: Final = scenario.team(**fields)
    scenario.gateway.post("/team/update", {"team_id": team, "max_budget": 7})
    return team


def _updated_key(scenario: Scenario, **fields: JsonValue) -> str:
    token: Final = _hashed(scenario.key(**fields))
    scenario.gateway.post("/key/update", {"key": token, "max_budget": 3})
    return token


UPDATED: Final[Mapping[str, Callable[..., str]]] = {"team": _updated_team, "key": _updated_key}
IDENTITY_FIELD: Final = {"team": "team_id", "key": "token"}
IDENTITY_FILTER: Final = {"team": "object_team_id", "key": "object_key_hash"}


@pytest.mark.parametrize("subject", ("team", "key"))
def test_an_update_stores_the_state_before_as_an_object_with_a_secret_in_its_metadata_masked(
    audited: Gateway, subject: str
) -> None:
    with audited.scenario() as scenario:
        object_id: Final = UPDATED[subject](scenario, metadata={"api_key": SECRET})

        row: Final = _stored(object_id, "updated")[0]

        assert row["before_type"] == "object", row
        assert object_value(row["before_value"])["metadata"] == {"api_key": SECRET_MASKED}, row


@pytest.mark.parametrize("subject", ("team", "key"))
def test_an_update_row_read_by_its_id_returns_the_state_before_exactly_as_stored(
    audited: Gateway, subject: str
) -> None:
    with audited.scenario() as scenario:
        object_id: Final = UPDATED[subject](scenario)
        row: Final = _stored(object_id, "updated")[0]

        found: Final = audited.request("GET", f"/audit/{string_value(row['id'])}")

        assert found.status_code == 200, found.text
        assert object_value(found.json())["before_value"] == row["before_value"], found.text
        assert object_value(row["before_value"])[IDENTITY_FIELD[subject]] == object_id, row


@pytest.mark.parametrize("subject", ("team", "key"))
def test_filtering_the_audit_page_by_a_team_or_key_lists_its_creation_and_its_update(
    audited: Gateway, subject: str
) -> None:
    with audited.scenario() as scenario:
        object_id: Final = UPDATED[subject](scenario)
        _stored(object_id, "updated")

        page: Final = audited.request("GET", "/audit", params={IDENTITY_FILTER[subject]: object_id, "page_size": "100"})

        actions: Final = sorted(str(log["action"]) for log in _listed(page) if log["object_id"] == object_id)
        assert actions == ["created", "updated"], page.text


def test_text_inside_a_team_that_looks_like_json_is_listed_as_the_same_text(audited: Gateway) -> None:
    with audited.scenario() as scenario:
        team: Final = _updated_team(scenario, metadata=TEXT_THAT_LOOKS_LIKE_JSON)
        _stored(team, "updated")

        page: Final = audited.request("GET", "/audit", params={"object_id": team, "action": "updated"})

        assert [object_value(log["before_value"])["metadata"] for log in _listed(page)] == [
            TEXT_THAT_LOOKS_LIKE_JSON
        ], page.text


@pytest.mark.parametrize(
    "fields", ({}, {"metadata": {}}, {"metadata": None}), ids=("metadata_missing", "metadata_empty", "metadata_null")
)
def test_a_team_without_metadata_is_listed_with_empty_metadata_before_its_update(
    audited: Gateway, fields: dict[str, JsonValue]
) -> None:
    with audited.scenario() as scenario:
        team: Final = _updated_team(scenario, **fields)
        _stored(team, "updated")

        page: Final = audited.request("GET", "/audit", params={"object_id": team, "action": "updated"})

        before: Final = [object_value(log["before_value"]) for log in _listed(page)]
        assert [(state["team_id"], state.get("metadata", {})) for state in before] == [(team, {})], page.text


def test_a_key_update_is_listed_with_the_team_the_key_belonged_to_only_when_it_had_one(audited: Gateway) -> None:
    with audited.scenario() as scenario:
        team: Final = scenario.team()
        in_team: Final = _updated_key(scenario, team_id=team)
        alone: Final = _updated_key(scenario)
        _stored(in_team, "updated")
        _stored(alone, "updated")

        pages: Final = [
            audited.request("GET", "/audit", params={"object_id": token, "action": "updated"})
            for token in (in_team, alone)
        ]

        before: Final = [object_value(log["before_value"]) for page in pages for log in _listed(page)]
        assert [(state["token"], state.get("team_id")) for state in before] == [(in_team, team), (alone, None)]


def test_three_identical_team_updates_are_listed_as_three_rows_across_two_pages(audited: Gateway) -> None:
    with audited.scenario() as scenario:
        team: Final = scenario.team()
        for _ in range(3):
            audited.post("/team/update", {"team_id": team, "max_budget": 7})
        stored: Final = _stored(team, "updated", count=3)

        pages: Final = [
            audited.request(
                "GET", "/audit", params={"object_id": team, "action": "updated", "page_size": "2", "page": page}
            )
            for page in ("1", "2")
        ]

        listed: Final = [log for page in pages for log in _listed(page)]
        assert sorted(str(log["id"]) for log in listed) == sorted(str(row["id"]) for row in stored)
        assert len({str(row["id"]) for row in stored}) == 3, stored
        assert [object_value(log["before_value"])["team_id"] for log in listed] == [team, team, team]
        assert [(page.json()["total"], page.json()["total_pages"]) for page in pages] == [(3, 2), (3, 2)]


def _created_team(scenario: Scenario) -> tuple[str, str]:
    return scenario.team(), "created"


def _deleted_team(scenario: Scenario) -> tuple[str, str]:
    created: Final = scenario.gateway.post("/team/new", {"team_alias": f"audited-{uuid.uuid4().hex}"})
    team: Final = string_value(created["team_id"])
    scenario.gateway.post("/team/delete", {"team_ids": [team]})
    return team, "deleted"


def _team_with_a_member_added(scenario: Scenario) -> tuple[str, str]:
    team: Final = scenario.team()
    scenario.member(team)
    return team, "updated"


def _team_with_a_callback_set(scenario: Scenario) -> tuple[str, str]:
    team: Final = scenario.team()
    scenario.gateway.post(
        f"/team/{team}/callback",
        {
            "callback_name": "langfuse",
            "callback_type": "success",
            "callback_vars": {
                "langfuse_public_key": "pk-lf-audited",
                "langfuse_secret_key": "sk-lf-audited",
                "langfuse_host": "http://127.0.0.1:9",
            },
        },
    )
    return team, "updated"


def _created_key(scenario: Scenario) -> tuple[str, str]:
    return _hashed(scenario.key()), "created"


def _blocked_key(scenario: Scenario) -> tuple[str, str]:
    key: Final = scenario.key()
    scenario.gateway.post("/key/block", {"key": key})
    return _hashed(key), "blocked"


def _unblocked_key(scenario: Scenario) -> tuple[str, str]:
    key: Final = scenario.key()
    scenario.gateway.post("/key/block", {"key": key})
    scenario.gateway.post("/key/unblock", {"key": key})
    return _hashed(key), "unblocked"


def _rotated_key(scenario: Scenario) -> tuple[str, str]:
    key: Final = string_value(scenario.gateway.post("/key/generate", {})["key"])
    rotated: Final = string_value(scenario.gateway.post(f"/key/{key}/regenerate", {})["key"])
    scenario.cleanups.callback(scenario.delete_key, rotated)
    return _hashed(key), "rotated"


def _deleted_key(scenario: Scenario) -> tuple[str, str]:
    key: Final = string_value(scenario.gateway.post("/key/generate", {})["key"])
    scenario.gateway.post("/key/delete", {"keys": [key]})
    return _hashed(key), "deleted"


def _created_user(scenario: Scenario) -> tuple[str, str]:
    return scenario.user(), "created"


def _updated_user(scenario: Scenario) -> tuple[str, str]:
    user: Final = scenario.user()
    scenario.gateway.post("/user/update", {"user_id": user, "max_budget": 4})
    return user, "updated"


def _deleted_user(scenario: Scenario) -> tuple[str, str]:
    user: Final = f"audited-{uuid.uuid4().hex}"
    scenario.gateway.post("/user/new", {"user_id": user, "auto_create_key": False})
    deleted: Final = scenario.gateway.request("POST", "/user/delete", {"user_ids": [user]})
    assert deleted.status_code == 200, deleted.text
    return user, "deleted"


def _model(scenario: Scenario) -> str:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": f"audited-{uuid.uuid4().hex}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": f"{scenario.gateway.upstream_url}/v1",
            },
            "model_info": {},
        },
    )
    return string_value(created["model_id"])


def _created_model(scenario: Scenario) -> tuple[str, str]:
    model: Final = _model(scenario)
    scenario.cleanups.callback(scenario.delete_model, model)
    return model, "created"


def _deleted_model(scenario: Scenario) -> tuple[str, str]:
    model: Final = _model(scenario)
    scenario.gateway.post("/model/delete", {"id": model})
    return model, "deleted"


OTHER_WRITES: Final = (
    _created_team,
    _deleted_team,
    _team_with_a_member_added,
    _team_with_a_callback_set,
    _created_key,
    _blocked_key,
    _unblocked_key,
    _rotated_key,
    _deleted_key,
    _created_user,
    _updated_user,
    _deleted_user,
    _created_model,
    _deleted_model,
)


@pytest.mark.parametrize("write", OTHER_WRITES, ids=lambda write: write.__name__.lstrip("_"))
def test_every_other_management_write_is_listed_exactly_as_the_object_it_stored(
    audited: Gateway, write: Callable[[Scenario], tuple[str, str]]
) -> None:
    with audited.scenario() as scenario:
        object_id, action = write(scenario)
        row: Final = _stored(object_id, action)[0]

        page: Final = audited.request("GET", "/audit", params={"object_id": object_id, "action": action})

        assert {row["before_type"], row["updated_type"]} <= {"object", None}, row
        assert [(log["id"], log["before_value"], log["updated_values"]) for log in _listed(page)] == [
            (row["id"], row["before_value"], row["updated_values"])
        ], page.text


def test_the_audit_page_refuses_a_caller_without_a_key_and_reports_an_unknown_row_as_missing(audited: Gateway) -> None:
    without_key: Final = audited.client.get("/audit")
    unknown: Final = audited.request("GET", f"/audit/{uuid.uuid4().hex}")

    assert (without_key.status_code, unknown.status_code) == (401, 404), f"{without_key.text} {unknown.text}"


def test_an_update_of_a_team_or_key_that_does_not_exist_answers_404_and_leaves_no_audit_row(audited: Gateway) -> None:
    missing_team: Final = f"audited-missing-{uuid.uuid4().hex}"
    missing_key: Final = f"sk-audited-missing-{uuid.uuid4().hex}"
    with audited.scenario() as scenario:
        refused: Final = [
            audited.request("POST", "/team/update", {"team_id": missing_team, "max_budget": 7}),
            audited.request("POST", "/key/update", {"key": missing_key, "max_budget": 3}),
        ]
        written_after: Final = _updated_team(scenario)

        assert len(_stored(written_after, "updated")) == 1
        assert [response.status_code for response in refused] == [404, 404], [response.text for response in refused]
        assert [read_rows(IDS_SQL, (identity,)) for identity in (missing_team, _hashed(missing_key))] == [[], []]
