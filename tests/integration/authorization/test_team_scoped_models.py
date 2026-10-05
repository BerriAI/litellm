import uuid
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows
from pydantic import BaseModel, JsonValue


class _ObservedRequest(BaseModel):
    body: dict[str, JsonValue]


class _Observations(BaseModel):
    requests: list[_ObservedRequest]


class _TeamState(BaseModel):
    team_id: str
    models: list[str] | None = None


class _TeamInfo(BaseModel):
    team_info: _TeamState


class _ProxyError(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _ProxyError


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as client:
        client.get("/__observations").raise_for_status()
        yield client


def _observed_models(upstream: httpx.Client) -> list[JsonValue]:
    observed: Final = upstream.get("/__observations")
    observed.raise_for_status()
    return [request["body"]["model"] for request in observed.json()["requests"]]


def _observed_provider_models(upstream: httpx.Client) -> tuple[str, ...]:
    observed: Final = upstream.get("/__observations")
    assert observed.status_code == 200, observed.text
    response: Final = _Observations.model_validate_json(observed.text)
    return tuple(string_value(request.body["model"]) for request in response.requests)


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"team model {uuid.uuid4().hex}"}]},
        key=key,
    )


def _team_model(scenario: Scenario) -> tuple[str, str]:
    provider_model: Final = f"team-model-{uuid.uuid4().hex}"
    return scenario.model(model=f"openai/{provider_model}"), provider_model


def _team_info(gateway: Gateway, team_id: str) -> _TeamInfo:
    response: Final = gateway.request("GET", "/team/info", params={"team_id": team_id})
    assert response.status_code == 200, response.text
    return _TeamInfo.model_validate_json(response.text)


def _model_denied(response: httpx.Response) -> None:
    assert response.status_code == 403, response.text
    assert _ErrorResponse.model_validate_json(response.text).error.type == "team_model_access_denied", response.text


def _ids(listing: dict[str, JsonValue]) -> set[JsonValue]:
    data: Final = listing["data"]
    assert isinstance(data, list)
    return {object_value(entry)["id"] for entry in data}


def test_a_model_created_for_a_team_is_listed_in_that_teams_models(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(models=[])
        other: Final = scenario.team(models=[])
        model: Final = scenario.model(model_info={"team_id": team})
        own_models: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])["models"]
        other_models: Final = object_value(gateway.get("/team/info", {"team_id": other})["team_info"])["models"]
        assert isinstance(own_models, list) and isinstance(other_models, list)
        assert model in own_models
        assert model not in other_models


def test_a_team_model_is_listed_and_served_only_for_keys_of_its_team(gateway: Gateway, upstream: httpx.Client) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(models=[])
        other: Final = scenario.team(models=[])
        provider_model: Final = f"team-scoped-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}", model_info={"team_id": team})
        team_key: Final = scenario.key(team_id=team)
        other_key: Final = scenario.key(team_id=other)
        assert model in _ids(object_value(gateway.request("GET", "/models", key=team_key).json()))
        assert model not in _ids(object_value(gateway.request("GET", "/models", key=other_key).json()))
        served: Final = _chat(gateway, model, team_key)
        assert served.status_code == 200, served.text
        refused: Final = _chat(gateway, model, other_key)
        assert refused.status_code == 400, refused.text
        assert _observed_models(upstream) == [provider_model]


def _v2_team_public_names(gateway: Gateway, key: str, model: str) -> list[JsonValue]:
    response: Final = gateway.request("GET", "/v2/model/info", key=key, params={"model": model})
    assert response.status_code == 200, response.text
    return [entry["model_info"].get("team_public_model_name") for entry in response.json()["data"]]


def test_v2_model_info_reports_a_team_model_to_team_and_non_team_keys(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(models=[])
        model: Final = scenario.model(model_info={"team_id": team})
        assert _v2_team_public_names(gateway, scenario.key(team_id=team), model) == [model]
        assert _v2_team_public_names(gateway, scenario.key(), model) == [model]


@pytest.mark.parametrize("set_on", ["team_new", "team_update"])
def test_team_model_alias_routes_a_team_key_to_its_target(
    gateway: Gateway, upstream: httpx.Client, set_on: str
) -> None:
    with gateway.scenario() as scenario:
        provider_model: Final = f"team-alias-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        alias: Final = f"alias-{uuid.uuid4().hex}"
        team: Final = (
            scenario.team(models=[model], model_aliases={alias: model})
            if set_on == "team_new"
            else scenario.team(models=[model])
        )
        if set_on == "team_update":
            gateway.post("/team/update", {"team_id": team, "model_aliases": {alias: model}})
        key: Final = scenario.key(team_id=team, models=[model])
        response: Final = _chat(gateway, alias, key)
        assert response.status_code == 200, response.text
        assert string_value(response.json()["model"]) == alias
        assert _observed_models(upstream) == [provider_model]
        unaliased: Final = _chat(gateway, f"alias-{uuid.uuid4().hex}", key)
        assert unaliased.status_code == 403, unaliased.text
        assert unaliased.json()["error"]["type"] == "key_model_access_denied"
        assert _observed_models(upstream) == []


def test_team_model_add_and_delete_change_what_a_warmed_team_key_can_call(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model_one, provider_one = _team_model(scenario)
        model_two, provider_two = _team_model(scenario)
        model_three, provider_three = _team_model(scenario)
        team: Final = scenario.team(models=[model_one])
        key: Final = scenario.key(team_id=team)

        first: Final = _chat(gateway, model_one, key)
        assert first.status_code == 200, first.text
        initially_denied: Final = _chat(gateway, model_two, key)
        _model_denied(initially_denied)
        observed_before_add: Final = _observed_provider_models(upstream)
        assert observed_before_add == (provider_one,), repr(observed_before_add)

        added: Final = gateway.request("POST", "/team/model/add", {"team_id": team, "models": [model_two, model_three]})
        assert added.status_code == 200, added.text
        added_state: Final = _TeamState.model_validate_json(added.text)
        expected_added: Final = sorted((model_one, model_two, model_three))
        assert sorted(added_state.models or ()) == expected_added, added.text
        added_info: Final = _team_info(gateway, team)
        assert added_info.team_info.models == expected_added, repr(added_info)
        assert read_rows('SELECT models FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)) == [
            {"models": expected_added}
        ], added.text

        second: Final = _chat(gateway, model_two, key)
        assert second.status_code == 200, second.text
        third: Final = _chat(gateway, model_three, key)
        assert third.status_code == 200, third.text
        deleted: Final = gateway.request("POST", "/team/model/delete", {"team_id": team, "models": [model_two]})
        assert deleted.status_code == 200, deleted.text
        deleted_state: Final = _TeamState.model_validate_json(deleted.text)
        expected_remaining: Final = sorted((model_one, model_three))
        assert sorted(deleted_state.models or ()) == expected_remaining, deleted.text
        deleted_info: Final = _team_info(gateway, team)
        assert deleted_info.team_info.models == expected_remaining, repr(deleted_info)
        assert read_rows('SELECT models FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)) == [
            {"models": expected_remaining}
        ], deleted.text

        denied_after_delete: Final = _chat(gateway, model_two, key)
        _model_denied(denied_after_delete)
        still_allowed_one: Final = _chat(gateway, model_one, key)
        assert still_allowed_one.status_code == 200, still_allowed_one.text
        still_allowed_three: Final = _chat(gateway, model_three, key)
        assert still_allowed_three.status_code == 200, still_allowed_three.text
        observed_after_add_and_delete: Final = _observed_provider_models(upstream)
        assert observed_before_add + observed_after_add_and_delete == (
            provider_one,
            provider_two,
            provider_three,
            provider_one,
            provider_three,
        ), repr((observed_before_add, observed_after_add_and_delete))


def test_team_model_add_on_an_unrestricted_team_keeps_every_other_model(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model_one, provider_one = _team_model(scenario)
        model_two, provider_two = _team_model(scenario)
        team: Final = scenario.team(models=[])
        key: Final = scenario.key(team_id=team)

        before_add: Final = _chat(gateway, model_two, key)
        assert before_add.status_code == 200, before_add.text

        added: Final = gateway.request("POST", "/team/model/add", {"team_id": team, "models": [model_one]})
        assert added.status_code == 200, added.text
        expected: Final = sorted(("all-proxy-models", model_one))
        added_state: Final = _TeamState.model_validate_json(added.text)
        assert sorted(added_state.models or ()) == expected, added.text
        added_info: Final = _team_info(gateway, team)
        assert added_info.team_info.models == expected, repr(added_info)
        assert read_rows(
            'SELECT models FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [{"models": expected}], added.text
        still_unrestricted: Final = _chat(gateway, model_two, key)
        assert still_unrestricted.status_code == 200, still_unrestricted.text
        added_model: Final = _chat(gateway, model_one, key)
        assert added_model.status_code == 200, added_model.text
        observed_after_add: Final = _observed_provider_models(upstream)
        assert observed_after_add == (provider_two, provider_two, provider_one), repr(observed_after_add)
