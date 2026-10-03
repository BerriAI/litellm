from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app

client = TestClient(app)


@pytest.fixture
def authenticated_client(monkeypatch):
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234"
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
    router = MagicMock()
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
    response = authenticated_client.get(
        "/v1/model/deprecations", params={"warn_within_days": 40000}
    )

    assert response.status_code == 200
    payload = response.json()
    assert [m["model_name"] for m in payload["imminent"]] == ["future-alias"]
    assert payload["upcoming"] == []
    assert payload["warn_within_days"] == 40000
