from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from tests.unit.proxy.router_entry_point_stubs import route_entry_points_to_stubs

client = TestClient(app)


@pytest.fixture
def authenticated_client(monkeypatch):
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-9876"
    )
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "sunset-model": {
                "deprecation_date": "2020-01-01",
                "litellm_provider": "openai",
            },
            "future-model": {
                "deprecation_date": "2099-01-01",
                "litellm_provider": "openai",
            },
        },
    )
    router = route_entry_points_to_stubs(MagicMock())
    router.get_model_list.return_value = [
        {
            "model_name": "sunset-alias",
            "litellm_params": {"model": "sunset-model"},
            "model_info": {"id": "1"},
        },
        {
            "model_name": "future-alias",
            "litellm_params": {"model": "future-model"},
            "model_info": {"id": "2"},
        },
    ]
    monkeypatch.setattr(proxy_server, "llm_router", router)
    yield client
    app.dependency_overrides.pop(user_api_key_auth, None)


def test_should_bucket_configured_models_by_urgency(authenticated_client):
    response = authenticated_client.get("/model/deprecations")

    assert response.status_code == 200
    payload = response.json()
    assert [m["model_name"] for m in payload["deprecated"]] == ["sunset-alias"]
    assert [m["model_name"] for m in payload["upcoming"]] == ["future-alias"]
    assert payload["imminent"] == []
    assert payload["warn_within_days"] == 30
    assert payload["deprecated"][0]["days_until_deprecation"] < 0


def test_should_rebucket_with_warn_within_days_override(authenticated_client):
    response = authenticated_client.get("/v1/model/deprecations", params={"warn_within_days": 40000})

    assert response.status_code == 200
    payload = response.json()
    assert [m["model_name"] for m in payload["imminent"]] == ["future-alias"]
    assert payload["upcoming"] == []
    assert payload["warn_within_days"] == 40000


@pytest.fixture
def client_with_cost_map_successors(monkeypatch):
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-9876"
    )
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "retiring-model": {
                "deprecation_date": "2020-01-01",
                "litellm_provider": "openai",
                "successor_model": "fresh-model",
            },
            "orphan-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"},
        },
    )
    router = route_entry_points_to_stubs(MagicMock())
    router.get_model_list.return_value = [
        {
            "model_name": "retiring-alias",
            "litellm_params": {"model": "retiring-model"},
            "model_info": {"id": "1"},
        },
        {
            "model_name": "retiring-overridden",
            "litellm_params": {"model": "retiring-model"},
            "model_info": {"id": "2", "successor_model": "my-own-pick"},
        },
        {
            "model_name": "shared-name",
            "litellm_params": {"model": "retiring-model", "api_key": "fake"},
            "model_info": {"id": "4"},
        },
        {
            "model_name": "shared-name",
            "litellm_params": {"model": "retiring-model", "api_key": "fake"},
            "model_info": {"id": "5", "successor_model": "my-own-pick"},
        },
        {
            "model_name": "dated-by-hand",
            "litellm_params": {"model": "retiring-model", "api_key": "fake"},
            "model_info": {"id": "6", "deprecation_date": "2020-06-01"},
        },
        {
            "model_name": "orphan-alias",
            "litellm_params": {"model": "orphan-model"},
            "model_info": {"id": "3"},
        },
    ]
    monkeypatch.setattr(proxy_server, "llm_router", router)
    yield client
    app.dependency_overrides.pop(user_api_key_auth, None)


def test_cost_map_successor_is_reported_unless_the_deployment_names_its_own(client_with_cost_map_successors):
    response = client_with_cost_map_successors.get("/model/deprecations")

    assert response.status_code == 200, response.text
    successors = {m["model_name"]: m["successor_model"] for m in response.json()["deprecated"]}
    assert successors == {
        "retiring-alias": "fresh-model",
        "retiring-overridden": "my-own-pick",
        "shared-name": "my-own-pick",
        "dated-by-hand": "fresh-model",
        "orphan-alias": None,
    }
