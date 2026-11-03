import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final, Literal

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, delete_key_if_present, object_value, string_value
from tests.integration._support.database import read_rows

KEY_ALIAS: Final = "mistral-7b"
ModelAccess = Literal["all-team-models", "single-model"]
AccessLevel = Literal["key", "team"]
ModelEndpoint = Literal["/v1/models", "/model/info"]


def _owned_key(
    gateway: Gateway, scenario: Scenario, fields: dict[str, JsonValue], *, key: str | None = None
) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", "/key/generate", fields, key=key)
    assert response.status_code == 200, response.text
    created: Final = object_value(response.json())
    scenario.cleanups.callback(delete_key_if_present, gateway, string_value(created["key"]))
    return created


def _data(gateway: Gateway, path: str, key: str, params: dict[str, str] | None = None) -> list[JsonValue]:
    response: Final = gateway.request("GET", path, key=key, params=params)
    assert response.status_code == 200, response.text
    data: Final = object_value(response.json())["data"]
    assert isinstance(data, list)
    return data


def _verification_row(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT user_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def test_concurrent_key_generation_persists_every_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        with ThreadPoolExecutor(max_workers=10) as pool:
            keys: Final = list(pool.map(lambda _: scenario.key(models=[model]), range(10)))
        assert len(set(keys)) == 10
        for key in keys:
            assert len(_verification_row(key)) == 1


def test_generated_key_exposes_hashed_token_and_timestamps(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        created: Final = _owned_key(gateway, scenario, {})
        key: Final = string_value(created["key"])
        assert created["token"] is not None
        assert created["token"] != key
        assert created["token"] == sha256(key.encode()).hexdigest()
        assert created["token_id"] is not None
        assert created["created_at"] is not None
        assert created["updated_at"] is not None


def test_key_info_serves_admin_and_self_and_hides_unknown_keys(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        digest: Final = sha256(key.encode()).hexdigest()
        admin: Final = gateway.request("GET", "/key/info", params={"key": key})
        assert admin.status_code == 200, admin.text
        assert object_value(admin.json())["key"] == key
        explicit: Final = gateway.request("GET", "/key/info", params={"key": key}, key=key)
        assert explicit.status_code == 200, explicit.text
        assert object_value(explicit.json())["key"] == key
        implicit: Final = gateway.request("GET", "/key/info", key=key)
        assert implicit.status_code == 200, implicit.text
        assert object_value(implicit.json())["key"] == digest
        unknown: Final = gateway.request("GET", "/key/info", params={"key": f"sk-{uuid.uuid4()}"}, key=key)
        assert unknown.status_code == 404, unknown.text


def test_model_info_is_filtered_to_the_models_a_key_can_use(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        admin_models: Final = _data(gateway, "/model/info", gateway.key)
        user_models: Final = _data(gateway, "/model/info", key)
        assert len(admin_models) > len(user_models)
        assert [object_value(entry)["model_name"] for entry in user_models] == [model]


def test_proxy_admin_user_key_deletes_a_key_owned_by_another_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        admin_user: Final = scenario.user(user_role="proxy_admin")
        owner: Final = scenario.user(user_role="internal_user")
        admin_key: Final = string_value(_owned_key(gateway, scenario, {"user_id": admin_user})["key"])
        victim: Final = string_value(_owned_key(gateway, scenario, {"user_id": owner})["key"])
        deleted: Final = gateway.request("POST", "/key/delete", {"keys": [victim]}, key=admin_key)
        assert deleted.status_code == 200, deleted.text
        assert _verification_row(victim) == []


@pytest.mark.parametrize("model_endpoint", ["/v1/models", "/model/info"])
@pytest.mark.parametrize("access_level", ["key", "team"])
@pytest.mark.parametrize("model_access", ["all-team-models", "single-model"])
def test_key_model_list_follows_key_or_team_access(
    gateway: Gateway, model_access: ModelAccess, access_level: AccessLevel, model_endpoint: ModelEndpoint
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        granted: Final[list[JsonValue]] = [] if model_access == "all-team-models" else [model]
        team: Final = scenario.team(models=granted if access_level == "team" else [])
        key: Final = scenario.key(
            team_id=team,
            models=granted if access_level == "key" else [],
            aliases={KEY_ALIAS: model},
        )
        data: Final = _data(gateway, model_endpoint, key)
        if model_access == "all-team-models":
            assert len(data) > 1
            if model_endpoint == "/v1/models":
                assert all(isinstance(object_value(entry)["id"], str) for entry in data)
                assert model in {object_value(entry)["id"] for entry in data}
            else:
                assert model in {object_value(entry)["model_name"] for entry in data}
        elif model_endpoint == "/v1/models":
            assert {object_value(entry)["id"] for entry in data} == {model, KEY_ALIAS}
        else:
            assert [object_value(entry)["model_name"] for entry in data] == [model]


def test_internal_user_cannot_reassign_its_key_to_another_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first: Final = scenario.user(user_role="internal_user")
        second: Final = scenario.user(user_role="internal_user")
        key: Final = string_value(_owned_key(gateway, scenario, {"user_id": first})["key"])
        own_key: Final = string_value(_owned_key(gateway, scenario, {}, key=key)["key"])
        assert _verification_row(own_key) == [{"user_id": first}]
        update: Final = gateway.request("POST", "/key/update", {"key": own_key, "user_id": second}, key=key)
        assert update.status_code == 403, update.text
        assert _verification_row(own_key) == [{"user_id": first}]


def test_internal_user_cannot_delete_another_users_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first: Final = scenario.user(user_role="internal_user")
        second: Final = scenario.user(user_role="internal_user")
        victim: Final = string_value(_owned_key(gateway, scenario, {"user_id": first})["key"])
        attacker: Final = string_value(_owned_key(gateway, scenario, {"user_id": second})["key"])
        deleted: Final = gateway.request("POST", "/key/delete", {"keys": [victim]}, key=attacker)
        assert deleted.status_code == 403, deleted.text
        assert _verification_row(victim) == [{"user_id": first}]
