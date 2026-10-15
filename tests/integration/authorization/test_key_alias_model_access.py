import uuid
from typing import Final

import httpx

from tests.integration._support.client import Gateway, eventually, object_value, string_value


def _listed_model_ids(response: httpx.Response) -> frozenset[str]:
    entries: Final = response.json()["data"]
    assert isinstance(entries, list), response.text
    return frozenset(string_value(object_value(entry)["id"]) for entry in entries)


def _listed_and_callable(gateway: Gateway, key: str, model: str, alias: str) -> None:
    """Every id /v1/models lists for this key must be callable by the same key."""
    response: Final = eventually(
        lambda: gateway.request("GET", "/v1/models", key=key),
        lambda value: value.status_code == 200 and _listed_model_ids(value) == frozenset({model, alias}),
        return_last_on_timeout=True,
    )
    assert response.status_code == 200, response.text
    listed: Final = _listed_model_ids(response)
    assert listed == frozenset({model, alias}), response.text
    for model_id in sorted(listed):
        called: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model_id, "messages": [{"role": "user", "content": "ping"}]},
            key=key,
        )
        assert called.status_code == 200, f"listed id {model_id} is not callable: {called.status_code} {called.text}"


def test_key_alias_listed_by_v1_models_is_callable(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = f"integration-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], aliases={alias: model})
        _listed_and_callable(gateway, key, model, alias)


def test_team_key_alias_listed_by_v1_models_is_callable(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_id: Final = scenario.team(models=[model])
        alias: Final = f"integration-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(team_id=team_id, aliases={alias: model})
        _listed_and_callable(gateway, key, model, alias)


def test_key_alias_to_model_outside_key_allowlist_is_hidden_and_denied(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        hidden: Final = scenario.model()
        alias: Final = f"integration-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[allowed], aliases={alias: hidden})
        response: Final = eventually(
            lambda: gateway.request("GET", "/v1/models", key=key),
            lambda value: value.status_code == 200 and _listed_model_ids(value) == frozenset({allowed}),
            return_last_on_timeout=True,
        )
        assert response.status_code == 200, response.text
        assert _listed_model_ids(response) == frozenset({allowed}), response.text
        called: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": alias, "messages": [{"role": "user", "content": "ping"}]},
            key=key,
        )
        assert called.status_code == 403, called.text
        assert "key_model_access_denied" in called.text, called.text
