from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.process import OwnedState, owned_proxy

TEAM_DURATION: Final = "30d"
TEAM_LIFETIME: Final = timedelta(days=30)
CLOCK_SLACK: Final = timedelta(seconds=10)


def _key_info(gateway: Gateway, key: str) -> dict[str, JsonValue]:
    return object_value(gateway.get("/key/info", {"key": sha256(key.encode()).hexdigest()})["info"])


def _expiry(info: dict[str, JsonValue]) -> datetime:
    assert isinstance(info["expires"], str), f"Expected a finite expiry, got {info['expires']!r}"
    return datetime.fromisoformat(info["expires"])


@pytest.mark.parametrize(
    ("duration_fields", "expected_lifetime"),
    (
        pytest.param({"duration": "1d"}, timedelta(days=1), id="shorter"),
        pytest.param({}, TEAM_LIFETIME, id="omitted"),
        pytest.param({"duration": "90d"}, TEAM_LIFETIME, id="longer"),
        pytest.param({"duration": "-1"}, TEAM_LIFETIME, id="unlimited"),
        pytest.param({"duration": None}, TEAM_LIFETIME, id="null"),
        pytest.param({"duration": ""}, TEAM_LIFETIME, id="empty"),
        pytest.param({"duration": " \t"}, TEAM_LIFETIME, id="whitespace"),
    ),
)
def test_generate_team_member_key_respects_requested_duration_and_team_ceiling(
    scenario: Scenario,
    duration_fields: dict[str, JsonValue],
    expected_lifetime: timedelta,
) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team)
    requested_at: Final = datetime.now(timezone.utc)
    response: Final = gateway.request("POST", "/key/generate", {"team_id": team, "user_id": member, **duration_fields})

    assert response.status_code == 200, response.text
    expiry: Final = _expiry(JSON_OBJECT.validate_json(response.content))
    assert abs(expiry - requested_at - expected_lifetime) <= CLOCK_SLACK


def test_team_admin_key_with_inferred_owner_keeps_shorter_duration(scenario: Scenario) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team, role="admin")
    caller: Final = scenario.key(user_id=member)
    requested_at: Final = datetime.now(timezone.utc)

    created: Final = gateway.post("/key/generate", {"team_id": team, "duration": "1d"}, key=caller)

    assert created["user_id"] == member
    assert abs(_expiry(created) - requested_at - timedelta(days=1)) <= CLOCK_SLACK


@pytest.mark.parametrize("operation", ["generate", "update"])
def test_invalid_team_member_duration_returns_bad_request(scenario: Scenario, operation: str) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team)
    if operation == "generate":
        route, target = "/key/generate", {"team_id": team, "user_id": member}
    else:
        key: Final = scenario.key(team_id=team, user_id=member)
        prior_expiry: Final = _key_info(gateway, key)["expires"]
        route, target = "/key/update", {"key": sha256(key.encode()).hexdigest()}

    response: Final = gateway.request("POST", route, {**target, "duration": "tomorrow"})

    assert response.status_code == 400, response.text
    assert "Invalid duration format" in response.text
    if operation == "update":
        assert _key_info(gateway, key)["expires"] == prior_expiry


@pytest.mark.parametrize(
    ("route", "update", "days"),
    (
        pytest.param("/key/update", {"duration": "90d"}, 30, id="update-longer"),
        pytest.param("/key/update", {"duration": "-1"}, 30, id="update-unlimited"),
        pytest.param("/key/update", {"duration": None}, 30, id="update-null"),
        pytest.param("/key/update", {"duration": "1d"}, 1, id="update-shorter"),
        pytest.param("/key/update", {"duration": "-1", "user_id": None}, 30, id="update-remove-owner"),
        pytest.param("/key/bulk_update", {"duration": "90d"}, None, id="bulk-update-longer"),
        pytest.param("/team/key/bulk_update", {"duration": "90d"}, 30, id="team-bulk-update-longer"),
        pytest.param("/team/key/bulk_update", {"duration": "-1"}, 30, id="team-bulk-update-unlimited"),
        pytest.param("/team/key/bulk_update", {"duration": None}, 30, id="team-bulk-update-null"),
        pytest.param("/team/key/bulk_update", {"duration": "1d"}, 1, id="team-bulk-update-shorter"),
        pytest.param("/key/{key}/regenerate", {"duration": "90d"}, 30, id="regenerate-longer"),
        pytest.param("/key/{key}/regenerate", {"duration": "-1"}, 30, id="regenerate-unlimited"),
        pytest.param("/key/{key}/regenerate", {"duration": None}, 30, id="regenerate-null"),
        pytest.param("/key/{key}/regenerate", {"duration": "1d"}, 1, id="regenerate-shorter"),
        pytest.param("/key/regenerate", {"duration": "90d"}, 30, id="legacy-regenerate-longer"),
        pytest.param("/key/{key}/regenerate", {}, None, id="regenerate-omitted"),
    ),
)
def test_existing_team_member_key_respects_the_team_ceiling(
    scenario: Scenario, route: str, update: dict[str, JsonValue], days: int | None
) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team)
    created: Final = gateway.post("/key/generate", {"team_id": team, "user_id": member})
    key: Final = string_value(created["key"])
    key_id: Final = sha256(key.encode()).hexdigest()
    previous_expiry: Final = _expiry(_key_info(gateway, key)) if days is None else None
    requested_at: Final = datetime.now(timezone.utc)

    if route == "/key/bulk_update":
        response = gateway.request("POST", route, {"keys": [{"key": key_id, **update}]})
    elif route == "/team/key/bulk_update":
        response = gateway.request("POST", route, {"team_id": team, "key_ids": [key_id], "update_fields": update})
    elif route == "/key/{key}/regenerate":
        response = gateway.request("POST", route.format(key=key_id), update)
    else:
        response = gateway.request("POST", route, {"key": key_id, **update})

    assert response.status_code == 200, response.text
    if route in ("/key/bulk_update", "/team/key/bulk_update"):
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["failed_updates"] == []
        successful: Final = body["successful_updates"]
        assert isinstance(successful, list) and len(successful) == 1, response.text
    active_key: Final = (
        string_value(JSON_OBJECT.validate_json(response.content)["key"]) if route.endswith("/regenerate") else key
    )
    info: Final = _key_info(gateway, active_key)
    assert info["status"] == "active"
    if "user_id" in update:
        assert info["user_id"] is None
    if days is None:
        assert _expiry(info) == previous_expiry
    else:
        assert abs(_expiry(info) - requested_at - timedelta(days=days)) <= CLOCK_SLACK


@pytest.mark.parametrize("duration", ["", " \t"], ids=["empty", "whitespace"])
def test_blank_team_member_update_keeps_expiry(scenario: Scenario, duration: str) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team)
    key: Final = scenario.key(team_id=team, user_id=member, duration="1d")
    previous: Final = _key_info(gateway, key)["expires"]

    gateway.post("/key/update", {"key": sha256(key.encode()).hexdigest(), "duration": duration})

    assert _key_info(gateway, key)["expires"] == previous


@pytest.mark.parametrize(
    ("initial_duration", "expected_lifetime"),
    [
        pytest.param(None, TEAM_LIFETIME, id="no-expiry"),
        pytest.param("90d", TEAM_LIFETIME, id="later-expiry"),
        pytest.param("1d", None, id="earlier-expiry"),
    ],
)
def test_move_member_key_without_duration_applies_destination_team_ceiling(
    scenario: Scenario, initial_duration: str | None, expected_lifetime: timedelta | None
) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team)
    key: Final = scenario.key(user_id=member, duration=initial_duration)
    previous_expiry: Final = _key_info(gateway, key)["expires"]
    requested_at: Final = datetime.now(timezone.utc)

    gateway.post("/key/update", {"key": sha256(key.encode()).hexdigest(), "team_id": team})

    info: Final = _key_info(gateway, key)
    assert info["team_id"] == team
    if expected_lifetime is None:
        assert info["expires"] == previous_expiry
    else:
        assert abs(_expiry(info) - requested_at - expected_lifetime) <= CLOCK_SLACK


@pytest.mark.parametrize(
    ("initial_duration", "clamped"),
    [
        pytest.param(None, True, id="no-expiry"),
        pytest.param("90d", True, id="later-expiry"),
        pytest.param("1d", False, id="earlier-expiry"),
    ],
)
def test_assigning_a_key_owner_without_duration_applies_member_ceiling(
    scenario: Scenario, initial_duration: str | None, clamped: bool
) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team)
    created: Final = gateway.post("/key/service-account/generate", {"team_id": team, "duration": initial_duration})
    key: Final = string_value(created["key"])
    previous: Final = _key_info(gateway, key)
    assert previous["user_id"] is None
    requested_at: Final = datetime.now(timezone.utc)

    gateway.post("/key/update", {"key": sha256(key.encode()).hexdigest(), "user_id": member})

    info: Final = _key_info(gateway, key)
    assert (info["team_id"], info["user_id"]) == (team, member)
    if clamped:
        assert abs(_expiry(info) - requested_at - TEAM_LIFETIME) <= CLOCK_SLACK
    else:
        assert info["expires"] == previous["expires"]


@pytest.mark.parametrize("service_account", [True, False], ids=["service-account", "no-team-ceiling"])
def test_move_without_an_applicable_member_ceiling_preserves_expiry(scenario: Scenario, service_account: bool) -> None:
    gateway: Final = scenario.gateway
    if service_account:
        source: Final = scenario.team()
        destination = scenario.team(team_member_key_duration=TEAM_DURATION)
        created: Final = gateway.post("/key/service-account/generate", {"team_id": source, "duration": "90d"})
        key = string_value(created["key"])
        owner = None
    else:
        destination = scenario.team()
        owner = scenario.member(destination)
        key = scenario.key(user_id=owner, duration="90d")
    prior: Final = _key_info(gateway, key)
    assert prior["team_id"] != destination

    gateway.post("/key/update", {"key": sha256(key.encode()).hexdigest(), "team_id": destination})

    info: Final = _key_info(gateway, key)
    assert (info["team_id"], info["user_id"], info["expires"]) == (destination, owner, prior["expires"])


def test_team_member_key_duration_minus_one_defaults_to_no_expiry_without_a_ceiling(scenario: Scenario) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration="-1")
    member: Final = scenario.member(team)
    created: Final = gateway.post("/key/generate", {"team_id": team, "user_id": member})
    assert created["expires"] is None

    requested_at: Final = datetime.now(timezone.utc)
    key: Final = string_value(created["key"])
    gateway.post("/key/update", {"key": sha256(key.encode()).hexdigest(), "duration": "90d"})
    assert abs(_expiry(_key_info(gateway, key)) - requested_at - timedelta(days=90)) <= CLOCK_SLACK


@pytest.fixture(scope="module")
def duration_upperbound_gateway(
    session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    tmp_path: Final = tmp_path_factory.mktemp("duration-upperbound")
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = object_value(base["litellm_settings"])
    config: Final = {
        **base,
        "litellm_settings": {
            **settings,
            "default_key_generate_params": {"duration": "2d"},
            "upperbound_key_generate_params": {"duration": "7d"},
        },
    }
    path: Final = tmp_path / "duration-upperbound.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(session_gateway, tmp_path, {}, config=path, state=OwnedState.SHARED_WITH_WORKER) as candidate:
        yield candidate


@pytest.mark.shares_worker_state
@pytest.mark.parametrize(
    ("team_duration", "fields", "days"),
    [
        pytest.param(TEAM_DURATION, {}, 7, id="team-default-clamped"),
        pytest.param(TEAM_DURATION, {"duration": "90d"}, 7, id="team-cap-clamped"),
        pytest.param("-1", {}, 7, id="unlimited-team-still-has-proxy-upperbound"),
        pytest.param("  ", {}, 2, id="blank-team-setting-uses-proxy-default"),
        pytest.param(TEAM_DURATION, {"duration": "1d"}, 1, id="explicit-shorter-than-both"),
    ],
)
def test_team_member_generation_duration_precedence(
    duration_upperbound_gateway: Gateway, team_duration: str, fields: dict[str, JsonValue], days: int
) -> None:
    gateway: Final = duration_upperbound_gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_key_duration=team_duration)
        member: Final = scenario.member(team)
        requested_at: Final = datetime.now(timezone.utc)
        created: Final = gateway.post("/key/generate", {"team_id": team, "user_id": member, **fields})
        assert abs(_expiry(created) - requested_at - timedelta(days=days)) <= CLOCK_SLACK


@pytest.mark.shares_worker_state
@pytest.mark.parametrize("operation", ["generate", "update", "regenerate"])
def test_explicit_duration_below_team_ceiling_still_obeys_proxy_upperbound(
    duration_upperbound_gateway: Gateway, operation: str
) -> None:
    gateway: Final = duration_upperbound_gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
        member: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=member, duration="1d")
        prior: Final = _key_info(gateway, key)["expires"]
        key_id: Final = sha256(key.encode()).hexdigest()
        route, fields = {
            "generate": ("/key/generate", {"team_id": team, "user_id": member}),
            "update": ("/key/update", {"key": key_id}),
            "regenerate": (f"/key/{key_id}/regenerate", {}),
        }[operation]

        response: Final = gateway.request("POST", route, {**fields, "duration": "10d"})

        assert response.status_code == 400, response.text
        assert "duration" in response.text
        assert _key_info(gateway, key)["expires"] == prior


@pytest.mark.shares_worker_state
@pytest.mark.parametrize("operation", ["update", "team-bulk", "regenerate"])
def test_team_derived_expiry_obeys_proxy_upperbound(duration_upperbound_gateway: Gateway, operation: str) -> None:
    gateway: Final = duration_upperbound_gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
        member: Final = scenario.member(team)
        created: Final = gateway.post("/key/generate", {"team_id": team, "user_id": member, "duration": "1d"})
        key: Final = string_value(created["key"])
        key_id: Final = sha256(key.encode()).hexdigest()
        requested_at: Final = datetime.now(timezone.utc)
        if operation == "regenerate":
            response: Final = gateway.post(f"/key/{key_id}/regenerate", {"duration": "90d"})
            active_key = string_value(response["key"])
        else:
            active_key = key
            if operation == "team-bulk":
                bulk: Final = gateway.post(
                    "/team/key/bulk_update",
                    {"team_id": team, "key_ids": [key_id], "update_fields": {"duration": "90d"}},
                )
                assert bulk["failed_updates"] == []
            else:
                gateway.post("/key/update", {"key": key_id, "duration": "90d"})
        assert abs(_expiry(_key_info(gateway, active_key)) - requested_at - timedelta(days=7)) <= CLOCK_SLACK


@pytest.mark.shares_worker_state
def test_service_account_uses_proxy_upperbound_but_not_team_member_ceiling(
    duration_upperbound_gateway: Gateway,
) -> None:
    gateway: Final = duration_upperbound_gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_key_duration="1d")
        created_at: Final = datetime.now(timezone.utc)
        created: Final = gateway.post("/key/service-account/generate", {"team_id": team, "duration": "5d"})
        assert created["user_id"] is None
        assert abs(_expiry(created) - created_at - timedelta(days=5)) <= CLOCK_SLACK
        key: Final = string_value(created["key"])
        key_id: Final = sha256(key.encode()).hexdigest()
        updated_at: Final = datetime.now(timezone.utc)

        gateway.post("/key/update", {"key": key_id, "duration": "6d"})
        expiry: Final = _expiry(_key_info(gateway, key))
        assert abs(expiry - updated_at - timedelta(days=6)) <= CLOCK_SLACK

        rejected: Final = gateway.request("POST", "/key/update", {"key": key_id, "duration": "10d"})
        assert rejected.status_code == 400, rejected.text
        assert _expiry(_key_info(gateway, key)) == expiry
