from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
import pytest
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
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.authorization._hidden_alias_budget import install_aliases, remove_aliases
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse


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


def _listed_names(gateway: Gateway, key: str, path: str, name_field: str) -> frozenset[str]:
    response: Final = gateway.request("GET", path, key=key)
    assert response.status_code == 200, response.text
    entries: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    return frozenset(string_value(object_value(entry)[name_field]) for entry in entries)


def _listed_models(gateway: Gateway, key: str) -> frozenset[str]:
    return _listed_names(gateway, key, "/v1/models", "id")


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


@pytest.mark.parametrize(
    ("path", "name_field"),
    [("/v1/models", "id"), ("/model/info", "model_name"), ("/model_group/info", "model_group")],
    ids=["v1-models", "model-info", "model-group-info"],
)
def test_a_keys_model_listings_leave_out_its_disallowed_models(gateway: Gateway, path: str, name_field: str) -> None:
    with gateway.scenario() as scenario:
        grant: Final = _team_grant(scenario)
        granted: Final = frozenset((grant.allowed, grant.disallowed))
        unrestricted: Final = scenario.key(team_id=grant.team_id)
        eventually(
            lambda: _listed_names(gateway, unrestricted, path, name_field),
            lambda names: granted <= names,
            seconds=30,
        )
        key: Final = scenario.key(team_id=grant.team_id, disallowed_models=[grant.disallowed])
        listed: Final = eventually(
            lambda: _listed_names(gateway, key, path, name_field),
            lambda names: grant.allowed in names,
            seconds=30,
        )
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


def test_a_key_alias_to_a_deployment_id_cannot_reach_a_disallowed_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        deployment_id: Final = f"integration-deployment-{uuid.uuid4().hex}"
        disallowed: Final = scenario.model(model_info={"id": deployment_id})
        alias: Final = f"alias-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(aliases={alias: deployment_id}, disallowed_models=[disallowed])
        _assert_denied(gateway, key, {"model": alias}, alias)


def test_a_key_alias_to_a_router_alias_cannot_reach_a_disallowed_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        disallowed: Final = scenario.model()
        router_alias: Final = f"router-alias-{uuid.uuid4().hex[:8]}"
        key_alias: Final = f"alias-{uuid.uuid4().hex[:8]}"
        install_aliases(gateway, {router_alias: disallowed})
        scenario.cleanups.callback(remove_aliases, gateway, frozenset({router_alias}))
        unrestricted: Final = scenario.key(aliases={key_alias: router_alias})
        served: Final = eventually(
            lambda: tuple(_chat(gateway, unrestricted, {"model": key_alias}).status_code for _ in range(6)),
            lambda statuses: frozenset(statuses) == {200},
            seconds=60,
        )
        assert frozenset(served) == {200}, served
        key: Final = scenario.key(aliases={key_alias: router_alias}, disallowed_models=[disallowed])
        _assert_denied(gateway, key, {"model": key_alias}, key_alias)


def test_a_comma_separated_fan_out_cannot_reach_a_disallowed_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        tier_upstream: Final = f"fanout-{uuid.uuid4().hex[:8]}"
        disallowed: Final = scenario.model(model=f"openai/{tier_upstream}")
        alias: Final = f"alias-{uuid.uuid4().hex[:8]}"
        aliases: Final = {alias: f"{allowed},{disallowed}"}
        served: Final = _chat(gateway, scenario.key(aliases=aliases), {"model": alias})
        assert served.status_code == 200, f"{served.status_code} {served.text}"
        assert tier_upstream in _observed_models(gateway)
        key: Final = scenario.key(aliases=aliases, disallowed_models=[disallowed])
        _assert_denied(gateway, key, {"model": alias}, alias)
        _assert_denied(gateway, key, {"model": f"{allowed}, {disallowed}"}, disallowed)


def test_a_complexity_router_cannot_send_a_key_to_a_disallowed_tier(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        keyword: Final = f"simple-{uuid.uuid4().hex}"
        tier_upstream: Final = f"tier-{uuid.uuid4().hex[:8]}"
        disallowed: Final = scenario.model(model=f"openai/{tier_upstream}")
        other_tier: Final = scenario.model()
        router: Final = scenario.model(
            model="auto_router/complexity_router",
            complexity_router_config={
                "classifier_type": "heuristic",
                "keyword_tier_rules": [{"keywords": [keyword], "tier": "SIMPLE"}],
                "tiers": {"SIMPLE": disallowed, "MEDIUM": other_tier, "COMPLEX": other_tier, "REASONING": other_tier},
            },
        )
        body: Final[dict[str, JsonValue]] = {"model": router, "messages": [{"role": "user", "content": keyword}]}
        served: Final = _chat(gateway, scenario.key(), body)
        assert served.status_code == 200, f"{served.status_code} {served.text}"
        assert tier_upstream in _observed_models(gateway)
        _assert_denied(gateway, scenario.key(disallowed_models=[disallowed]), body, disallowed)


def _deleted_disallowed_models(keys: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT disallowed_models FROM "LiteLLM_DeletedVerificationToken" WHERE token = ANY(%s) ORDER BY key_alias',
        ([sha256(key.encode()).hexdigest() for key in keys],),
    )


@pytest.mark.parametrize("count", [1, 2], ids=["one-key", "two-keys"])
def test_deleting_keys_keeps_their_disallowed_models_on_the_deleted_records(gateway: Gateway, count: int) -> None:
    prefix: Final = f"deleted-disallowed-{uuid.uuid4().hex[:8]}"
    keys: Final = tuple(
        string_value(
            gateway.post("/key/generate", {"key_alias": f"{prefix}-{index}", "disallowed_models": ["a", f"b-{index}"]})[
                "key"
            ]
        )
        for index in range(count)
    )
    deleted: Final = gateway.request("POST", "/key/delete", {"keys": list(keys)})
    assert deleted.status_code == 200, f"{deleted.status_code} {deleted.text}"
    assert _deleted_disallowed_models(keys) == [{"disallowed_models": ["a", f"b-{index}"]} for index in range(count)]
    assert [row for key in keys for row in _stored_disallowed_models(key)] == []


def test_a_team_deployments_routing_name_cannot_reach_a_disallowed_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        disallowed: Final = scenario.model(model_info={"team_id": team})
        (deployment,) = read_rows(
            """SELECT model_name FROM "LiteLLM_ProxyModelTable" WHERE model_info->>'team_public_model_name' = %s""",
            (disallowed,),
        )
        routing_name: Final = string_value(deployment["model_name"])
        assert routing_name.startswith(f"model_name_{team}_"), routing_name
        _assert_served(gateway, scenario.key(team_id=team), routing_name)
        key: Final = scenario.key(team_id=team, disallowed_models=[disallowed])
        _assert_denied(gateway, key, {"model": routing_name}, routing_name)


def _batch_row(model: str) -> str:
    return json.dumps(
        {
            "custom_id": "row-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "batch row"}], "max_tokens": 8},
        }
    )


def _batch_provider(input_file: str) -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /files": JsonResponse(
                content_type="application/json",
                body={
                    "id": "file-in-$REQUEST_ID",
                    "object": "file",
                    "purpose": "batch",
                    "bytes": len(input_file),
                    "created_at": 1,
                    "filename": "input.jsonl",
                    "status": "processed",
                },
            ),
            "GET /files/file-in-$REQUEST_ID/content": TextResponse(content_type="application/jsonl", body=input_file),
            "POST /batches": JsonResponse(
                content_type="application/json",
                body={
                    "id": "batch-$REQUEST_ID",
                    "object": "batch",
                    "endpoint": "/v1/chat/completions",
                    "input_file_id": "file-in-$REQUEST_ID",
                    "completion_window": "24h",
                    "status": "validating",
                    "created_at": 1,
                },
            ),
        },
    )


def _create_batch_from_file(gateway: Gateway, key: str, provider_model: str, input_file: str) -> httpx.Response:
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": provider_model},
        {"file": ("input.jsonl", input_file.encode(), "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, f"{uploaded.status_code} {uploaded.text}"
    return gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        key=key,
    )


def _batch_provider_model(scenario: Scenario, group: str, row_model: str | None = None) -> str:
    model: Final = f"integration-{uuid.uuid4().hex}"
    provider: Final = register_scenario(
        f"disallowed-batch-{uuid.uuid4().hex}", _batch_provider(_batch_row(model if row_model is None else row_model))
    )
    scenario.cleanups.callback(delete_scenario, provider)
    return scenario.model(name=model, model_info={"access_groups": [group]}, api_base=provider.api_base())


@pytest.mark.parametrize("key_models", [[], ["all-team-models"]], ids=["inherits-team-models", "all-team-models"])
def test_a_batch_file_row_cannot_reach_a_disallowed_model(gateway: Gateway, key_models: list[JsonValue]) -> None:
    with gateway.scenario() as scenario:
        group: Final = f"integration-group-{uuid.uuid4().hex[:10]}"
        disallowed: Final = scenario.model(model_info={"access_groups": [group]})
        serves_own_rows: Final = _batch_provider_model(scenario, group)
        serves_disallowed_rows: Final = _batch_provider_model(scenario, group, disallowed)
        key: Final = scenario.key(
            team_id=scenario.team(models=[group]), models=key_models, disallowed_models=[disallowed]
        )
        created: Final = _create_batch_from_file(gateway, key, serves_own_rows, _batch_row(serves_own_rows))
        assert created.status_code == 200, f"{created.status_code} {created.text}"
        refused: Final = _create_batch_from_file(gateway, key, serves_disallowed_rows, _batch_row(disallowed))
        assert refused.status_code == 403, f"{refused.status_code} {refused.text}"
        assert disallowed in refused.text, refused.text
