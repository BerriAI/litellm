import uuid
from collections.abc import Iterator, Mapping
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
    team_admin_permissions,
)
from tests.integration._support.process import graceful_stop_seconds, owned_proxy

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

DEFAULT_ROUTES: Final = ("llm_api_routes", "/key/spend/report")
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_PROBE_PREFIX: Final = "route-probe-"
_RACE_PREFIX: Final = "route-race-"
_ROUTES_AFTER_ADMIN_REMOVAL: Final = ["llm_api_routes"]
_POLICY_SOURCE: Final = f'''import json

from litellm.proxy import proxy_server


async def route_policy(request):
    alias = request.request.key_alias or ""
    if request.operation == "update" and alias.startswith("{_RACE_PREFIX}"):
        await proxy_server.prisma_client.db.litellm_verificationtoken.update(
            where={{"token": request.existing_key.token}},
            data={{"allowed_routes": {_ROUTES_AFTER_ADMIN_REMOVAL!r}}},
        )
        return {{"decision": True}}
    if not alias.startswith("{_PROBE_PREFIX}"):
        return {{"decision": True}}
    return {{"decision": False, "message": "routes=" + json.dumps(request.effective_key.allowed_routes or [])}}
'''


def _config_with_default_routes(
    directory: Path, default_routes: list[JsonValue], general_settings: Mapping[str, JsonValue]
) -> Path:
    raw: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    configured: Final[dict[str, JsonValue]] = {
        **raw,
        "litellm_settings": {
            **object_value(raw.get("litellm_settings") or {}),
            "default_key_generate_params": {"allowed_routes": default_routes},
        },
        "general_settings": {**object_value(raw.get("general_settings") or {}), **general_settings},
    }
    config: Final = directory / "default-allowed-routes.yaml"
    config.write_text(yaml.safe_dump(configured))
    return config


@pytest.fixture(scope="module")
def defaults(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("default-allowed-routes")
    (directory / "route_policy.py").write_text(_POLICY_SOURCE)
    config: Final = _config_with_default_routes(
        directory, list(DEFAULT_ROUTES), {"custom_key_policy": "route_policy.route_policy"}
    )
    with gateway_from_environment() as gateway, owned_proxy(gateway, directory, {}, config=config) as candidate:
        yield candidate


@pytest.fixture(scope="module")
def non_string_default(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("non-string-default-allowed-routes")
    config: Final = _config_with_default_routes(directory, ["llm_api_routes", 123], {})
    with gateway_from_environment() as gateway, owned_proxy(gateway, directory, {}, config=config) as candidate:
        yield candidate


def _hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _stored_routes(gateway: Gateway, key: str) -> JsonValue:
    rows: Final = gateway.rows('SELECT allowed_routes FROM "LiteLLM_VerificationToken" WHERE token = %s', (_hash(key),))
    assert len(rows) == 1, rows
    return rows[0]["allowed_routes"]


def _created_key(scenario: Scenario, created: dict[str, JsonValue]) -> str:
    token: Final = string_value(created["key"])
    scenario.cleanups.callback(scenario.delete_key, token)
    return token


def _full_access_key(scenario: Scenario, **fields: JsonValue) -> str:
    return scenario.key(key_type="default", **fields)


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        ({}, list(DEFAULT_ROUTES)),
        ({"key_type": None}, list(DEFAULT_ROUTES)),
        ({"allowed_routes": []}, list(DEFAULT_ROUTES)),
        ({"key_type": "llm_api"}, list(DEFAULT_ROUTES)),
        ({"allowed_routes": ["info_routes"]}, ["info_routes"]),
        ({"key_type": "management"}, ["management_routes"]),
        ({"key_type": "read_only"}, ["info_routes"]),
        ({"key_type": "default"}, []),
    ),
    ids=("unset", "key_type_null", "empty_list", "llm_api", "explicit_list", "management", "read_only", "full_access"),
)
def test_admin_created_key_routes(defaults: Gateway, body: dict[str, JsonValue], expected: list[str]) -> None:
    with defaults.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        created: Final = defaults.post("/key/generate", {"user_id": user, **body})
        token: Final = _created_key(scenario, created)

        assert (created["allowed_routes"], _stored_routes(defaults, token)) == (expected, expected)


@pytest.mark.parametrize(
    ("body", "expected"),
    (({}, []), ({"key_type": "llm_api"}, ["llm_api_routes"])),
    ids=("unset", "llm_api"),
)
def test_default_with_a_non_string_route_is_ignored(
    non_string_default: Gateway, body: dict[str, JsonValue], expected: list[str]
) -> None:
    with non_string_default.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        created: Final = non_string_default.post("/key/generate", {"user_id": user, **body})
        token: Final = _created_key(scenario, created)

        assert (created["allowed_routes"], _stored_routes(non_string_default, token)) == (expected, expected)


def test_default_routes_open_own_spend_report_and_nothing_wider(defaults: Gateway) -> None:
    with defaults.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        user: Final = scenario.user(user_role="internal_user")
        token: Final = _created_key(
            scenario, defaults.post("/key/generate", {"user_id": user, "key_type": "llm_api", "models": [model]})
        )
        other: Final = _full_access_key(scenario, user_id=scenario.user(user_role="internal_user"))
        defaults.chat(model, key=token, text=f"default routes {uuid.uuid4().hex}")
        logged: Final = eventually(
            lambda: defaults.rows(
                """SELECT to_char("startTime", 'YYYY-MM-DD') AS day FROM "LiteLLM_SpendLogs" WHERE api_key = %s""",
                (_hash(token),),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        day: Final = date.fromisoformat(string_value(logged[0]["day"])).isoformat()
        window: Final = {"start_date": day, "end_date": day}

        own: Final = defaults.request("GET", "/key/spend/report", key=token, params=window)
        assert own.status_code == 200, own.text
        assert [object_value(row)["api_key"] for row in list_value(_JSON.validate_json(own.content))] == [_hash(token)]

        foreign: Final = defaults.request(
            "GET", "/key/spend/report", key=token, params={**window, "api_key": _hash(other)}
        )
        assert foreign.status_code == 403, foreign.text
        global_report: Final = defaults.request("GET", "/global/spend/report", key=token, params=window)
        assert global_report.status_code == 403, global_report.text
        minted: Final = defaults.request("POST", "/key/generate", {}, key=token)
        assert minted.status_code == 403, minted.text


@pytest.mark.parametrize("body", ({}, {"key_type": "llm_api"}), ids=("unset", "llm_api"))
def test_internal_user_own_key_routes(defaults: Gateway, body: dict[str, JsonValue]) -> None:
    with defaults.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        own_key: Final = _full_access_key(scenario, user_id=user)
        created: Final = defaults.request("POST", "/key/generate", body, key=own_key)
        assert created.status_code == 200, created.text
        token: Final = _created_key(scenario, JSON_OBJECT.validate_json(created.content))

        assert defaults.rows(
            'SELECT user_id, allowed_routes FROM "LiteLLM_VerificationToken" WHERE token = %s', (_hash(token),)
        ) == [{"user_id": user, "allowed_routes": list(DEFAULT_ROUTES)}]


def test_service_account_skips_default_and_its_team_key_gets_it(defaults: Gateway) -> None:
    with defaults.scenario() as scenario:
        team: Final = scenario.team()
        account: Final = _created_key(scenario, defaults.post("/key/service-account/generate", {"team_id": team}))
        assert _stored_routes(defaults, account) == []

        defaults.post("/team/permissions_update", {"team_id": team, "team_member_permissions": ["/key/generate"]})
        team_key: Final = _created_key(scenario, defaults.post("/key/generate", {"team_id": team}, key=account))

        assert defaults.rows(
            'SELECT user_id, team_id, allowed_routes FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (_hash(team_key),),
        ) == [{"user_id": None, "team_id": team, "allowed_routes": list(DEFAULT_ROUTES)}]


def _policy_seen_routes(response_text: str) -> JsonValue:
    message: Final = string_value(object_value(JSON_OBJECT.validate_json(response_text)["error"])["message"])
    assert message.startswith("routes="), message
    return _JSON.validate_json(message.removeprefix("routes="))


@pytest.mark.parametrize("team_admin", (False, True), ids=("unset", "team_admin_member_key"))
def test_custom_key_policy_sees_stored_routes(defaults: Gateway, team_admin: bool) -> None:
    with defaults.scenario() as scenario:
        team: Final = scenario.team()
        admin: Final = scenario.member(team, role="admin")
        member: Final = scenario.member(team)
        caller: Final = _full_access_key(scenario, user_id=admin, team_id=team) if team_admin else defaults.key
        body: Final[dict[str, JsonValue]] = {"team_id": team, "user_id": member}
        scenario.cleanups.enter_context(team_admin_permissions(defaults, ("create_member_keys",)))

        probe: Final = defaults.request(
            "POST", "/key/generate", {**body, "key_alias": f"{_PROBE_PREFIX}{uuid.uuid4().hex}"}, key=caller
        )
        assert probe.status_code == 403, probe.text
        created: Final = defaults.request(
            "POST", "/key/generate", {**body, "key_alias": f"route-twin-{uuid.uuid4().hex}"}, key=caller
        )
        assert created.status_code == 200, created.text
        token: Final = _created_key(scenario, JSON_OBJECT.validate_json(created.content))

        assert _policy_seen_routes(probe.text) == _stored_routes(defaults, token) == list(DEFAULT_ROUTES)


_ADMIN_SET_ROUTES: Final = ("llm_api_routes", "/key/spend/report")


def _owner_key_with_admin_set_routes(gateway: Gateway, scenario: Scenario) -> tuple[str, str]:
    owner_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))
    target: Final = _created_key(scenario, gateway.post("/key/generate", {}, key=owner_key))
    gateway.post("/key/update", {"key": target, "allowed_routes": list(_ADMIN_SET_ROUTES)})
    return owner_key, target


def _stored_alias_and_routes(gateway: Gateway, key: str) -> tuple[JsonValue, list[str]]:
    rows: Final = gateway.rows(
        'SELECT key_alias, allowed_routes FROM "LiteLLM_VerificationToken" WHERE token = %s', (_hash(key),)
    )
    assert len(rows) == 1, rows
    return rows[0]["key_alias"], [str(route) for route in list_value(rows[0]["allowed_routes"])]


def test_owner_resends_unchanged_routes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owner_key, target = _owner_key_with_admin_set_routes(gateway, scenario)
        alias: Final = f"resave-{uuid.uuid4().hex}"
        response: Final = gateway.request(
            "POST",
            "/key/update",
            {"key": target, "key_alias": alias, "allowed_routes": list(_ADMIN_SET_ROUTES)},
            key=owner_key,
        )

        assert response.status_code == 200, response.text
        assert _stored_alias_and_routes(gateway, target) == (alias, list(_ADMIN_SET_ROUTES))


@pytest.mark.parametrize(
    "routes",
    (
        [],
        ["llm_api_routes"],
        [*_ADMIN_SET_ROUTES, "/global/spend/report"],
        list(reversed(_ADMIN_SET_ROUTES)),
        [*_ADMIN_SET_ROUTES, _ADMIN_SET_ROUTES[-1]],
    ),
    ids=("clear", "ai_apis", "widen", "reordered", "duplicated"),
)
def test_owner_cannot_change_admin_set_routes(gateway: Gateway, routes: list[JsonValue]) -> None:
    with gateway.scenario() as scenario:
        owner_key, target = _owner_key_with_admin_set_routes(gateway, scenario)
        before: Final = _stored_alias_and_routes(gateway, target)
        response: Final = gateway.request(
            "POST", "/key/update", {"key": target, "allowed_routes": routes}, key=owner_key
        )

        assert response.status_code == 403, response.text
        assert _stored_alias_and_routes(gateway, target) == before


def test_non_owner_resending_unchanged_routes_is_refused(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        _owner_key, target = _owner_key_with_admin_set_routes(gateway, scenario)
        outsider_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))
        before: Final = _stored_alias_and_routes(gateway, target)
        response: Final = gateway.request(
            "POST",
            "/key/update",
            {"key": target, "key_alias": f"hijack-{uuid.uuid4().hex}", "allowed_routes": list(_ADMIN_SET_ROUTES)},
            key=outsider_key,
        )

        assert response.status_code == 403, response.text
        assert _stored_alias_and_routes(gateway, target) == before


def test_owner_resend_keeps_an_admin_route_removal_made_mid_update(defaults: Gateway) -> None:
    with defaults.scenario() as scenario:
        owner_key: Final = _full_access_key(scenario, user_id=scenario.user(user_role="internal_user"))
        created: Final = defaults.request("POST", "/key/generate", {}, key=owner_key)
        assert created.status_code == 200, created.text
        target: Final = _created_key(scenario, JSON_OBJECT.validate_json(created.content))
        assert _stored_routes(defaults, target) == list(DEFAULT_ROUTES)
        alias: Final = f"{_RACE_PREFIX}{uuid.uuid4().hex}"

        response: Final = defaults.request(
            "POST",
            "/key/update",
            {"key": target, "key_alias": alias, "allowed_routes": list(DEFAULT_ROUTES)},
            key=owner_key,
        )

        assert response.status_code == 200, response.text
        assert _stored_alias_and_routes(defaults, target) == (alias, _ROUTES_AFTER_ADMIN_REMOVAL)
