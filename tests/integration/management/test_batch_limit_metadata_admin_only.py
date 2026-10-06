import uuid
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, delete_key_if_present, object_value, string_value
from integration._support.database import read_rows
from integration.management._batch_file_caps import DOWNLOADS, RECORDS, UPLOADS, config_entry, hashed
from pydantic import JsonValue

TOKENS: Final = "batch_enqueued_token_limit"
LIMITS: Final = (TOKENS, RECORDS, UPLOADS, DOWNLOADS)
FILE_LIMITS: Final = (RECORDS, UPLOADS, DOWNLOADS)
EVERY_LIMIT: Final[Mapping[str, JsonValue]] = {TOKENS: 9000, RECORDS: 7, UPLOADS: 6, DOWNLOADS: 5}
BULK_ROUTE: Final = "/management/v1/users/bulk"


def _refusal(setting: str, entity: str) -> str:
    return (
        f"Only proxy admins can set {setting} on a {entity}. "
        "It limits what the holder can do with batches, so the holder cannot raise it."
    )


def _assert_refused(response: httpx.Response, setting: str, entity: str) -> None:
    assert response.status_code == 403, response.text
    error: Final = object_value(response.json()["error"])
    assert error["message"] == str({"error": _refusal(setting, entity)}), response.text
    assert error["code"] == "403", response.text


def _key_metadata(token: str) -> JsonValue:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(token),))
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def _team_metadata(team: str) -> JsonValue:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def _user_metadata(user: str) -> JsonValue:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,))
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def _user_key_metadata(user: str) -> list[JsonValue]:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_VerificationToken" WHERE user_id = %s', (user,))
    return [row["metadata"] for row in rows]


def _team_admin_key(scenario: Scenario, team: str) -> str:
    return scenario.key(user_id=scenario.member(team, role="admin"), team_id=team)


def _org_admin_key(scenario: Scenario, organization: str) -> str:
    return scenario.key(user_id=scenario.org_member(organization, role="org_admin"))


def _replaceable_key(gateway: Gateway, scenario: Scenario, **fields: JsonValue) -> str:
    key: Final = string_value(gateway.post("/key/generate", fields)["key"])
    scenario.cleanups.callback(delete_key_if_present, gateway, key)
    return key


def _drop_created_key(scenario: Scenario, response: httpx.Response) -> None:
    if response.status_code == 200 and response.json().get("key"):
        scenario.cleanups.callback(scenario.delete_key, string_value(response.json()["key"]))


def _drop_created_user(scenario: Scenario, response: httpx.Response, user: str) -> None:
    if response.status_code == 200:
        scenario.cleanups.callback(scenario.delete_user, user)
    _drop_created_key(scenario, response)


def _drop_bulk_rows(scenario: Scenario, response: httpx.Response) -> None:
    if response.status_code != 200:
        return
    for row in response.json()["data"]:
        if row["success"]:
            scenario.cleanups.callback(scenario.delete_user, string_value(row["user_id"]))
        if row["key"]:
            scenario.cleanups.callback(scenario.delete_key, string_value(row["key"]))


@pytest.mark.parametrize("setting", LIMITS)
def test_team_admin_cannot_put_a_batch_limit_on_a_new_key(gateway: Gateway, setting: str) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        caller: Final = _team_admin_key(scenario, team)
        alias: Final = f"batch-limit-{uuid.uuid4().hex}"
        refused: Final = gateway.request(
            "POST", "/key/generate", {"team_id": team, "key_alias": alias, "metadata": {setting: 5}}, key=caller
        )
        _drop_created_key(scenario, refused)
        _assert_refused(refused, setting, "key")
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s', (alias,)) == []
        plain: Final = gateway.request("POST", "/key/generate", {"team_id": team, "key_alias": alias}, key=caller)
        _drop_created_key(scenario, plain)
        assert plain.status_code == 200, plain.text
        assert _key_metadata(string_value(plain.json()["key"])) == {}


@pytest.mark.parametrize("setting", LIMITS)
def test_team_admin_cannot_put_a_batch_limit_on_an_existing_key(gateway: Gateway, setting: str) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        caller: Final = _team_admin_key(scenario, team)
        key: Final = scenario.key(team_id=team, metadata={"owner": "batch-limit-audit"})
        refused: Final = gateway.request(
            "POST", "/key/update", {"key": key, "metadata": {"owner": "batch-limit-audit", setting: 5}}, key=caller
        )
        _assert_refused(refused, setting, "key")
        assert _key_metadata(key) == {"owner": "batch-limit-audit"}


@pytest.mark.parametrize("setting", LIMITS)
def test_team_admin_cannot_put_a_batch_limit_on_a_regenerated_key(gateway: Gateway, setting: str) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        caller: Final = _team_admin_key(scenario, team)
        key: Final = _replaceable_key(gateway, scenario, team_id=team)
        refused: Final = gateway.request("POST", "/key/regenerate", {"key": key, "metadata": {setting: 5}}, key=caller)
        _drop_created_key(scenario, refused)
        _assert_refused(refused, setting, "key")
        assert _key_metadata(key) == {}


@pytest.mark.parametrize("setting", LIMITS)
def test_org_admin_cannot_put_a_batch_limit_on_an_existing_team(gateway: Gateway, setting: str) -> None:
    with gateway.scenario() as scenario:
        organization: Final = scenario.organization()
        caller: Final = _org_admin_key(scenario, organization)
        team: Final = scenario.team(organization_id=organization, metadata={"owner": "batch-limit-audit"})
        refused: Final = gateway.request(
            "POST",
            "/team/update",
            {"team_id": team, "metadata": {"owner": "batch-limit-audit", setting: 5}},
            key=caller,
        )
        _assert_refused(refused, setting, "team")
        assert _team_metadata(team) == {"owner": "batch-limit-audit"}


@pytest.mark.parametrize("setting", LIMITS)
def test_org_admin_cannot_put_a_batch_limit_on_a_new_team(gateway: Gateway, setting: str) -> None:
    with gateway.scenario() as scenario:
        organization: Final = scenario.organization()
        caller: Final = _org_admin_key(scenario, organization)
        alias: Final = f"batch-limit-{uuid.uuid4().hex}"
        refused: Final = gateway.request(
            "POST",
            "/team/new",
            {"team_alias": alias, "organization_id": organization, "metadata": {setting: 5}},
            key=caller,
        )
        if refused.status_code == 200:
            scenario.cleanups.callback(scenario.delete_team, string_value(refused.json()["team_id"]))
        _assert_refused(refused, setting, "team")
        assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_alias = %s', (alias,)) == []
        plain: Final = gateway.request(
            "POST", "/team/new", {"team_alias": alias, "organization_id": organization}, key=caller
        )
        if plain.status_code == 200:
            scenario.cleanups.callback(scenario.delete_team, string_value(plain.json()["team_id"]))
        assert plain.status_code == 200, plain.text
        assert _team_metadata(string_value(plain.json()["team_id"])) == {}


@pytest.mark.parametrize("setting", LIMITS)
def test_org_admin_cannot_put_a_batch_limit_on_a_new_users_key(gateway: Gateway, setting: str) -> None:
    with gateway.scenario() as scenario:
        organization: Final = scenario.organization()
        caller: Final = _org_admin_key(scenario, organization)
        refused_user, keyless_user, plain_user = (f"batch-limit-{uuid.uuid4().hex}" for _ in range(3))
        refused: Final = gateway.request(
            "POST",
            "/user/new",
            {"user_id": refused_user, "organization_id": organization, "metadata": {setting: 5}},
            key=caller,
        )
        _drop_created_user(scenario, refused, refused_user)
        _assert_refused(refused, setting, "key")
        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (refused_user,)) == []
        assert _user_key_metadata(refused_user) == []
        keyless: Final = gateway.request(
            "POST",
            "/user/new",
            {
                "user_id": keyless_user,
                "organization_id": organization,
                "metadata": {setting: 5},
                "auto_create_key": False,
            },
            key=caller,
        )
        _drop_created_user(scenario, keyless, keyless_user)
        assert keyless.status_code == 200, keyless.text
        assert _user_metadata(keyless_user) == {setting: 5}
        assert _user_key_metadata(keyless_user) == []
        plain: Final = gateway.request(
            "POST", "/user/new", {"user_id": plain_user, "organization_id": organization}, key=caller
        )
        _drop_created_user(scenario, plain, plain_user)
        assert plain.status_code == 200, plain.text
        assert _user_key_metadata(plain_user) == [{}]


@pytest.mark.parametrize("setting", LIMITS)
def test_bulk_user_creation_refuses_only_the_row_that_puts_a_batch_limit_on_a_key(
    gateway: Gateway, setting: str
) -> None:
    with gateway.scenario() as scenario:
        caller: Final = scenario.key(user_id=scenario.user(user_role="internal_user"), allowed_routes=[BULK_ROUTE])
        refused_user, keyless_user, plain_user = (f"batch-limit-{uuid.uuid4().hex}" for _ in range(3))
        response: Final = gateway.request(
            "POST",
            BULK_ROUTE,
            {
                "users": [
                    {"user_id": refused_user, "metadata": {setting: 5}, "auto_create_key": True},
                    {"user_id": keyless_user, "metadata": {setting: 5}},
                    {"user_id": plain_user, "auto_create_key": True},
                ]
            },
            key=caller,
        )
        _drop_bulk_rows(scenario, response)
        assert response.status_code == 200, response.text
        refused, keyless, plain = response.json()["data"]
        assert refused == {
            "user_id": refused_user,
            "user_email": None,
            "success": False,
            "teams": None,
            "key": None,
            "error": _refusal(setting, "key"),
        }, response.text
        assert (keyless["success"], keyless["key"], keyless["error"]) == (True, None, None), response.text
        assert (plain["success"], plain["error"]) == (True, None), response.text
        assert response.json()["meta"] == {"total_requested": 3, "created": 2, "failed": 1}, response.text
        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (refused_user,)) == []
        assert _user_key_metadata(refused_user) == []
        assert _user_metadata(keyless_user) == {setting: 5}
        assert _user_key_metadata(keyless_user) == []
        assert _user_key_metadata(plain_user) == [{}]
        assert _key_metadata(string_value(plain["key"])) == {}


def test_proxy_admin_sets_every_batch_limit_on_every_route(gateway: Gateway) -> None:
    limits: Final = dict(EVERY_LIMIT)
    with gateway.scenario() as scenario:
        generated: Final = scenario.key(metadata=limits)
        assert _key_metadata(generated) == limits
        updated: Final = scenario.key()
        gateway.post("/key/update", {"key": updated, "metadata": limits})
        assert _key_metadata(updated) == limits
        replaced: Final = _replaceable_key(gateway, scenario)
        regenerated: Final = string_value(gateway.post("/key/regenerate", {"key": replaced, "metadata": limits})["key"])
        scenario.cleanups.callback(scenario.delete_key, regenerated)
        assert _key_metadata(regenerated) == limits
        created_team: Final = scenario.team(metadata=limits)
        assert _team_metadata(created_team) == limits
        updated_team: Final = scenario.team()
        gateway.post("/team/update", {"team_id": updated_team, "metadata": limits})
        assert _team_metadata(updated_team) == limits
        new_user, bulk_user = (f"batch-limit-{uuid.uuid4().hex}" for _ in range(2))
        created_user: Final = gateway.request("POST", "/user/new", {"user_id": new_user, "metadata": limits})
        _drop_created_user(scenario, created_user, new_user)
        assert created_user.status_code == 200, created_user.text
        assert _user_key_metadata(new_user) == [limits]
        bulk: Final = gateway.request(
            "POST", BULK_ROUTE, {"users": [{"user_id": bulk_user, "metadata": limits, "auto_create_key": True}]}
        )
        _drop_bulk_rows(scenario, bulk)
        assert bulk.status_code == 200, bulk.text
        assert bulk.json()["meta"] == {"total_requested": 1, "created": 1, "failed": 0}, bulk.text
        assert _user_key_metadata(bulk_user) == [limits]


RESENDS: Final = (
    pytest.param({}, id="dropped"),
    pytest.param({UPLOADS: 4}, id="lowered"),
    pytest.param({UPLOADS: 6}, id="raised"),
    pytest.param({UPLOADS: "5"}, id="resent-as-a-string"),
    pytest.param({UPLOADS: None}, id="resent-as-null"),
)


@pytest.mark.parametrize("metadata", RESENDS)
def test_team_admin_may_resend_a_stored_key_limit_but_not_change_it(
    gateway: Gateway, metadata: Mapping[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        caller: Final = _team_admin_key(scenario, team)
        key: Final = scenario.key(team_id=team, metadata={UPLOADS: 5})
        alias: Final = f"batch-limit-{uuid.uuid4().hex}"
        resent: Final = gateway.request(
            "POST", "/key/update", {"key": key, "key_alias": alias, "metadata": {UPLOADS: 5}}, key=caller
        )
        assert resent.status_code == 200, resent.text
        untouched: Final = gateway.request("POST", "/key/update", {"key": key, "key_alias": f"{alias}-2"}, key=caller)
        assert untouched.status_code == 200, untouched.text
        changed: Final = gateway.request("POST", "/key/update", {"key": key, "metadata": dict(metadata)}, key=caller)
        _assert_refused(changed, UPLOADS, "key")
        assert _key_metadata(key) == {UPLOADS: 5}
        assert read_rows('SELECT key_alias FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),)) == [
            {"key_alias": f"{alias}-2"}
        ]


@pytest.mark.parametrize("metadata", RESENDS)
def test_org_admin_may_resend_a_stored_team_limit_but_not_change_it(
    gateway: Gateway, metadata: Mapping[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        organization: Final = scenario.organization()
        caller: Final = _org_admin_key(scenario, organization)
        team: Final = scenario.team(organization_id=organization, metadata={UPLOADS: 5})
        resent: Final = gateway.request(
            "POST", "/team/update", {"team_id": team, "tpm_limit": 5000, "metadata": {UPLOADS: 5}}, key=caller
        )
        assert resent.status_code == 200, resent.text
        untouched: Final = gateway.request("POST", "/team/update", {"team_id": team, "tpm_limit": 6000}, key=caller)
        assert untouched.status_code == 200, untouched.text
        changed: Final = gateway.request(
            "POST", "/team/update", {"team_id": team, "metadata": dict(metadata)}, key=caller
        )
        _assert_refused(changed, UPLOADS, "team")
        assert read_rows('SELECT metadata, tpm_limit FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)) == [
            {"metadata": {UPLOADS: 5}, "tpm_limit": 6000}
        ]


@pytest.mark.parametrize("setting", FILE_LIMITS)
def test_config_list_shows_each_file_limit_as_an_unset_integer(gateway: Gateway, setting: str) -> None:
    entry: Final = config_entry(gateway, setting)
    assert (entry["field_type"], entry["field_value"], entry["stored_in_db"], entry["editable"]) == (
        "Integer",
        None,
        None,
        True,
    ), entry


@pytest.mark.parametrize("setting", FILE_LIMITS)
def test_config_update_refuses_a_zero_file_limit(gateway: Gateway, setting: str) -> None:
    response: Final = gateway.request("POST", "/config/update", {"general_settings": {setting: 0}})
    assert response.status_code == 422, response.text
    assert response.json() == {
        "detail": [
            {
                "type": "greater_than",
                "loc": ["body", "general_settings", setting],
                "msg": "Input should be greater than 0",
            }
        ]
    }, response.text
    assert config_entry(gateway, setting)["field_value"] is None


@pytest.mark.parametrize("setting", FILE_LIMITS)
@pytest.mark.parametrize(
    ("value", "kind"),
    [
        pytest.param(0, "int", id="zero"),
        pytest.param(-1, "int", id="negative"),
        pytest.param("abc", "str", id="word"),
        pytest.param(1.5, "float", id="fraction"),
        pytest.param([5], "list", id="list"),
        pytest.param("", "str", id="empty-string"),
    ],
)
def test_config_field_update_refuses_a_malformed_file_limit(
    gateway: Gateway, value: JsonValue, kind: str, setting: str
) -> None:
    response: Final = gateway.request(
        "POST",
        "/config/field/update",
        {"field_name": setting, "field_value": value, "config_type": "general_settings"},
    )
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": {"error": f"Invalid type of field value=<class '{kind}'> passed in."}}
    assert config_entry(gateway, setting)["field_value"] is None
