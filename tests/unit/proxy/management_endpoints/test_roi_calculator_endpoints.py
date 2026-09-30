import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.roi_calculator_endpoints import (
    _estimator_models_from_deployments,
    get_roi_config_repository,
    router,
)
from litellm.proxy.roi_calculator.estimator import estimator_options
from litellm.types.roi_calculator import ROISettings

_JSON_HEADERS: Final = MappingProxyType({"content-type": "application/json"})


def _assert_json_round_trip(value: object) -> None:
    serialized: Final = json.dumps(value)
    decoded: Final[object] = cast(object, json.loads(serialized))
    assert decoded == value


class _Parameter:
    def __init__(self, param_value: object) -> None:
        self.param_value: Final = param_value


class _ConfigRepository:
    def __init__(self) -> None:
        self.values: Mapping[str, object] = MappingProxyType({})

    async def get_param(self, param_name: str) -> _Parameter | None:
        value: Final = self.values.get(param_name)
        return _Parameter(value) if value is not None else None

    async def set_param(self, param_name: str, param_value: object) -> object:
        _assert_json_round_trip(param_value)
        self.values = MappingProxyType({**self.values, param_name: param_value})
        return self.values[param_name]


def _client(role: LitellmUserRoles, repository: _ConfigRepository) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    return TestClient(app)


def test_router_group_uses_underlying_model_metadata_for_reasoning_option() -> None:
    import litellm

    supported_model: Final = next(
        model
        for model, metadata in litellm.model_cost.items()
        if metadata.get("supports_none_reasoning_effort") is True
    )
    deployments: Final = (
        {
            "model_name": "roi-estimator",
            "litellm_params": {"model": "custom-deployment"},
            "model_info": {"base_model": supported_model},
        },
    )

    estimator_models: Final = _estimator_models_from_deployments(deployments)

    assert estimator_models == ((supported_model, None),)
    assert estimator_options(estimator_models) == {"reasoning_effort": "none"}


def test_non_admin_cannot_read_roi_settings() -> None:
    client: Final = _client(LitellmUserRoles.INTERNAL_USER, _ConfigRepository())

    response: Final = client.get("/roi-calculator/settings")

    assert response.status_code == 403


def test_view_only_admin_cannot_change_roi_settings() -> None:
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, _ConfigRepository())

    response: Final = client.put(
        "/roi-calculator/settings",
        content='{"repos":["org/repo"]}',
        headers=_JSON_HEADERS,
    )

    assert response.status_code == 403


def test_github_token_is_never_returned_and_url_change_clears_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "roi-calculator-test-salt-key-0123456789")
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)

    saved: Final = client.put(
        "/roi-calculator/settings",
        content=('{"github_token":"private-test-token","repos":["org/repo"],"estimator_model":"test-estimator"}'),
        headers=_JSON_HEADERS,
    )

    assert saved.status_code == 200
    assert saved.json()["has_github_token"] is True
    assert "private-test-token" not in saved.text
    stored_settings: Final = TypeAdapter(ROISettings).validate_python(repository.values["roi_calculator_settings"])
    encrypted_token: Final = stored_settings.github_token.get_secret_value()
    assert encrypted_token != "private-test-token"
    assert "private-test-token" not in encrypted_token

    updated: Final = client.put(
        "/roi-calculator/settings",
        content='{"github_api_url":"https://github.enterprise.test/api/v3"}',
        headers=_JSON_HEADERS,
    )

    assert updated.status_code == 200
    assert updated.json()["has_github_token"] is False


def test_github_api_url_must_use_https() -> None:
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)

    response: Final = client.put(
        "/roi-calculator/settings",
        content='{"github_api_url":"http://github.enterprise.test/api/v3"}',
        headers=_JSON_HEADERS,
    )

    assert response.status_code == 422
    assert not repository.values


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
@pytest.mark.parametrize(
    "method,path,body",
    [
        ("POST", "/roi-calculator/sync", {}),
        ("DELETE", "/roi-calculator/sync", {}),
        ("POST", "/roi-calculator/setup/reset", {}),
        ("POST", "/roi-calculator/connections/test", {}),
        ("PUT", "/roi-calculator/identity-map", {"github_login": "alice", "email": "alice@example.com"}),
    ],
)
def test_all_writes_require_full_admin(role: LitellmUserRoles, method: str, path: str, body: Mapping[str, str]) -> None:
    client: Final = _client(role, _ConfigRepository())
    assert client.request(method, path, json=body).status_code == 403


def test_schedule_and_estimator_key_persist_without_exposing_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "roi-calculator-test-salt-key-0123456789")
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)
    saved: Final = client.put(
        "/roi-calculator/settings", json={"estimator_key": "sk-test-secret", "update_interval_minutes": 60}
    )
    assert saved.status_code == 200
    assert saved.json()["has_estimator_key"] is True
    assert saved.json()["update_interval_minutes"] == 60
    assert "sk-test-secret" not in saved.text
    assert "sk-test-secret" not in str(repository.values)
    updated: Final = client.put("/roi-calculator/settings", json={"estimator_key": None, "update_interval_minutes": 0})
    assert updated.json()["has_estimator_key"] is False
    assert updated.json()["update_interval_minutes"] == 0


def test_sample_preview_does_not_change_live_settings_or_report() -> None:
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, repository)
    response: Final = client.get("/roi-calculator/report", params={"mode": "demo"})
    assert response.status_code == 200
    assert response.json()["report"]["mode"] == "demo"
    assert response.json()["report"]["metrics"]["cost_per_hour"] > 0
    assert not repository.values
    assert client.get("/roi-calculator/report").json()["report"] is None


@pytest.mark.parametrize("interval", [0.1, 1, 4.99])
def test_schedule_rejects_intervals_under_five_minutes(interval: float) -> None:
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, _ConfigRepository())
    assert client.put("/roi-calculator/settings", json={"update_interval_minutes": interval}).status_code == 422
