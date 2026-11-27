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


@pytest.mark.parametrize(
    ("route", "update"),
    (
        pytest.param("/key/update", {"duration": "90d"}, id="update-longer"),
        pytest.param("/key/update", {"duration": "-1"}, id="update-unlimited"),
        pytest.param("/key/update", {"duration": "-1", "user_id": None}, id="update-remove-owner"),
        pytest.param("/key/bulk_update", {"duration": "90d"}, id="bulk-update-longer"),
        pytest.param("/team/key/bulk_update", {"duration": "90d"}, id="team-bulk-update-longer"),
        pytest.param("/key/{key}/regenerate", {"duration": "90d"}, id="regenerate-longer"),
    ),
)
def test_existing_team_member_key_cannot_be_given_a_duration_above_the_team_ceiling(
    scenario: Scenario, route: str, update: dict[str, JsonValue]
) -> None:
    gateway: Final = scenario.gateway
    team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
    member: Final = scenario.member(team)
    created: Final = gateway.post("/key/generate", {"team_id": team, "user_id": member})
    key: Final = string_value(created["key"])
    key_id: Final = sha256(key.encode()).hexdigest()
    previous_expiry: Final = _expiry(_key_info(gateway, key)) if route == "/key/bulk_update" else None
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
        string_value(JSON_OBJECT.validate_json(response.content)["key"]) if route == "/key/{key}/regenerate" else key
    )
    info: Final = _key_info(gateway, active_key)
    assert info["status"] == "active"
    if "user_id" in update:
        assert info["user_id"] is None
    if route == "/key/bulk_update":
        assert _expiry(info) == previous_expiry
    else:
        assert abs(_expiry(info) - requested_at - TEAM_LIFETIME) <= CLOCK_SLACK


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


@pytest.mark.shares_worker_state
def test_team_member_key_default_is_clamped_to_the_proxy_duration_upperbound(gateway: Gateway, tmp_path: Path) -> None:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = object_value(base["litellm_settings"])
    config: Final = {**base, "litellm_settings": {**settings, "upperbound_key_generate_params": {"duration": "7d"}}}
    path: Final = tmp_path / "duration-upperbound.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        owned_proxy(gateway, tmp_path, {}, config=path, state=OwnedState.SHARED_WITH_WORKER) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(team_member_key_duration=TEAM_DURATION)
        member: Final = scenario.member(team)
        requested_at: Final = datetime.now(timezone.utc)
        created: Final = candidate.post("/key/generate", {"team_id": team, "user_id": member})
        assert abs(_expiry(created) - requested_at - timedelta(days=7)) <= CLOCK_SLACK
