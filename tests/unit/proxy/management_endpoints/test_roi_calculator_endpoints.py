from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.roi_calculator_endpoints import (
    get_roi_config_repository,
    router,
)
from litellm.types.roi_calculator import ROISettings

_JSON_HEADERS: Final = MappingProxyType({"content-type": "application/json"})


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
        self.values = MappingProxyType({**self.values, param_name: param_value})
        return self.values[param_name]


def _client(role: LitellmUserRoles, repository: _ConfigRepository) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    return TestClient(app)


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
        content=(
            '{"github_token":"private-test-token","repos":["org/repo"],'
            '"estimator_model":"test-estimator"}'
        ),
        headers=_JSON_HEADERS,
    )

    assert saved.status_code == 200
    assert saved.json()["has_github_token"] is True
    assert "private-test-token" not in saved.text
    stored_settings: Final = TypeAdapter(ROISettings).validate_python(
        repository.values["roi_calculator_settings"]
    )
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
