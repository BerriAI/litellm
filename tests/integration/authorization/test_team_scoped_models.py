import uuid
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, object_value, string_value
from pydantic import JsonValue


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as client:
        client.get("/__observations").raise_for_status()
        yield client


def _observed_models(upstream: httpx.Client) -> list[JsonValue]:
    observed: Final = upstream.get("/__observations")
    observed.raise_for_status()
    return [request["body"]["model"] for request in observed.json()["requests"]]


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"team model {uuid.uuid4().hex}"}]},
        key=key,
    )


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
    response: Final = gateway.request("GET", "/v2/model/info", key=key, params={"model_name": model})
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
