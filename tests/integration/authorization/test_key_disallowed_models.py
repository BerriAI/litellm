from __future__ import annotations

import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    list_value,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows


@dataclass(frozen=True, slots=True)
class _TeamGrant:
    team_id: str
    allowed: str
    disallowed: str


def _team_grant(scenario: Scenario) -> _TeamGrant:
    group: Final = f"integration-group-{uuid.uuid4().hex[:10]}"
    allowed: Final = scenario.model(model_info={"access_groups": [group]})
    disallowed: Final = scenario.model(model_info={"access_groups": [group]})
    return _TeamGrant(team_id=scenario.team(models=[group]), allowed=allowed, disallowed=disallowed)


def _observed_models(gateway: Gateway) -> tuple[str, ...]:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=10)
    response.raise_for_status()
    requests: Final = list_value(JSON_OBJECT.validate_json(response.content)["requests"])
    return tuple(str(object_value(object_value(entry).get("body") or {}).get("model")) for entry in requests)


def _chat(gateway: Gateway, key: str, body: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": f"disallowed {uuid.uuid4().hex}"}], **body},
        key=key,
    )


def _assert_served(gateway: Gateway, key: str, model: str) -> None:
    response: Final = _chat(gateway, key, {"model": model})
    assert response.status_code == 200, f"{model}: {response.status_code} {response.text}"


def _assert_denied(gateway: Gateway, key: str, body: dict[str, JsonValue], denied: str) -> None:
    before: Final = _observed_models(gateway)
    response: Final = _chat(gateway, key, body)
    assert response.status_code == 403, f"{response.status_code} {response.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["type"] == "key_model_access_denied", response.text
    assert error["param"] == "model", response.text
    assert denied in string_value(error["message"]), response.text
    assert _observed_models(gateway) == before


def _listed_models(gateway: Gateway, key: str) -> frozenset[str]:
    response: Final = gateway.request("GET", "/v1/models", key=key)
    assert response.status_code == 200, response.text
    entries: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    return frozenset(string_value(object_value(entry)["id"]) for entry in entries)


def _stored_disallowed_models(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT disallowed_models FROM "LiteLLM_VerificationToken" WHERE token=%s',
        (sha256(key.encode()).hexdigest(),),
    )


def test_a_key_inheriting_a_team_access_group_cannot_call_or_list_its_disallowed_models(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        grant: Final = _team_grant(scenario)
        key: Final = scenario.key(team_id=grant.team_id, disallowed_models=[grant.disallowed])
        assert _stored_disallowed_models(key) == [{"disallowed_models": [grant.disallowed]}]
        _assert_served(gateway, key, grant.allowed)
        _assert_denied(gateway, key, {"model": grant.disallowed}, grant.disallowed)
        listed: Final = eventually(
            lambda: _listed_models(gateway, key),
            lambda models: grant.allowed in models,
            seconds=30,
        )
        assert grant.allowed in listed
        assert grant.disallowed not in listed


def test_an_all_team_models_key_cannot_call_its_disallowed_models(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        grant: Final = _team_grant(scenario)
        key: Final = scenario.key(
            team_id=grant.team_id, models=["all-team-models"], disallowed_models=[grant.disallowed]
        )
        _assert_served(gateway, key, grant.allowed)
        _assert_denied(gateway, key, {"model": grant.disallowed}, grant.disallowed)


def test_disallowed_models_win_over_the_keys_own_allow_list(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        disallowed: Final = scenario.model()
        key: Final = scenario.key(models=[allowed, disallowed], disallowed_models=[disallowed])
        _assert_served(gateway, key, allowed)
        _assert_denied(gateway, key, {"model": disallowed}, disallowed)
        assert disallowed not in _listed_models(gateway, key)


def test_an_alias_or_a_fallback_cannot_reach_a_disallowed_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        grant: Final = _team_grant(scenario)
        alias: Final = f"alias-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(
            team_id=grant.team_id, aliases={alias: grant.disallowed}, disallowed_models=[grant.disallowed]
        )
        _assert_denied(gateway, key, {"model": alias}, alias)
        _assert_denied(gateway, key, {"model": grant.allowed, "fallbacks": [grant.disallowed]}, grant.disallowed)


def test_adding_and_clearing_disallowed_models_takes_effect_on_the_next_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        grant: Final = _team_grant(scenario)
        key: Final = scenario.key(team_id=grant.team_id)
        _assert_served(gateway, key, grant.disallowed)
        gateway.post("/key/update", {"key": key, "disallowed_models": [grant.disallowed]})
        _assert_denied(gateway, key, {"model": grant.disallowed}, grant.disallowed)
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert info["disallowed_models"] == [grant.disallowed], info
        gateway.post("/key/update", {"key": key, "disallowed_models": []})
        _assert_served(gateway, key, grant.disallowed)
        gateway.post("/key/update", {"key": key, "disallowed_models": [grant.disallowed]})
        _assert_denied(gateway, key, {"model": grant.disallowed}, grant.disallowed)
        gateway.post("/key/update", {"key": key, "disallowed_models": None})
        assert _stored_disallowed_models(key) == [{"disallowed_models": []}]
        _assert_served(gateway, key, grant.disallowed)


def test_a_key_owner_cannot_remove_disallowed_models_but_can_add_them(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        disallowed: Final = scenario.model()
        owner: Final = scenario.user(user_role="internal_user")
        owner_key: Final = scenario.key(user_id=owner)
        created: Final = gateway.request(
            "POST", "/key/generate", {"user_id": owner, "disallowed_models": [disallowed]}, key=owner_key
        )
        assert created.status_code == 200, f"{created.status_code} {created.text}"
        key: Final = string_value(JSON_OBJECT.validate_json(created.content)["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        cleared: Final = gateway.request("POST", "/key/update", {"key": key, "disallowed_models": []}, key=key)
        assert cleared.status_code == 403, f"{cleared.status_code} {cleared.text}"
        assert "disallowed_models" in cleared.text, cleared.text
        assert _stored_disallowed_models(key) == [{"disallowed_models": [disallowed]}]
        _assert_denied(gateway, key, {"model": disallowed}, disallowed)
        widened: Final = gateway.request(
            "POST", "/key/update", {"key": key, "disallowed_models": [disallowed, allowed]}, key=key
        )
        assert widened.status_code == 200, f"{widened.status_code} {widened.text}"
        _assert_denied(gateway, key, {"model": allowed}, allowed)


def test_an_admin_of_the_keys_team_can_remove_disallowed_models(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        group: Final = f"integration-group-{uuid.uuid4().hex[:10]}"
        disallowed: Final = scenario.model(model_info={"access_groups": [group]})
        admin: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(models=[group], members_with_roles=[{"role": "admin", "user_id": admin}])
        admin_key: Final = scenario.key(team_id=team, user_id=admin)
        key: Final = scenario.key(team_id=team, disallowed_models=[disallowed])
        _assert_denied(gateway, key, {"model": disallowed}, disallowed)
        cleared: Final = gateway.request("POST", "/key/update", {"key": key, "disallowed_models": []}, key=admin_key)
        assert cleared.status_code == 200, f"{cleared.status_code} {cleared.text}"
        assert _stored_disallowed_models(key) == [{"disallowed_models": []}]
        _assert_served(gateway, key, disallowed)


def test_a_deployment_id_cannot_reach_a_disallowed_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        deployment_id: Final = f"integration-deployment-{uuid.uuid4().hex}"
        disallowed: Final = scenario.model(model_info={"id": deployment_id})
        key: Final = scenario.key(disallowed_models=[disallowed])
        _assert_denied(gateway, key, {"model": deployment_id}, deployment_id)
        _assert_denied(gateway, key, {"model": scenario.model(), "fallbacks": [deployment_id]}, deployment_id)
