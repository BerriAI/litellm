from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.alerting_endpoints.model_deprecation_alerts import alert_state_repository, router
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.config_repository import ConfigParam, ConfigRepository
from litellm.types.proxy.auth.user_roles import LitellmUserRoles


class MemoryConfig(ConfigRepository):
    def __init__(self) -> None:
        self.rows: dict[str, object] = {}

    async def get_param(self, param_name: str) -> ConfigParam | None:
        value = self.rows.get(param_name)
        return None if value is None else ConfigParam(param_name, value)

    async def set_param(self, param_name: str, param_value: object) -> ConfigParam:
        self.rows[param_name] = param_value
        return ConfigParam(param_name, param_value)


@pytest.fixture()
def client() -> Iterator[tuple[TestClient, UserAPIKeyAuth]]:
    app = FastAPI()
    app.include_router(router)
    user = UserAPIKeyAuth(user_id="admin-one", user_role=LitellmUserRoles.PROXY_ADMIN)
    repository = MemoryConfig()
    app.dependency_overrides[user_api_key_auth] = lambda: user
    app.dependency_overrides[alert_state_repository] = lambda: repository
    with TestClient(app) as http:
        yield http, user


IDENTITY = {"model": "anthropic/claude-sonnet-4-5", "deprecation_date": "2026-11-30"}
UPDATE = {**IDENTITY, "status": "resolved", "last_seen_date": "2026-11-04", "last_seen_requests": 123}
PATH = "/alerting/model-deprecations/state"


def test_new_alert_is_open(client: tuple[TestClient, UserAPIKeyAuth]) -> None:
    http, _ = client
    response = http.get(PATH, params=IDENTITY)
    assert response.status_code == 200
    assert response.json()["status"] == "open"


def test_resolution_is_shared_and_keeps_the_usage_snapshot(client: tuple[TestClient, UserAPIKeyAuth]) -> None:
    http, user = client
    response = http.put(PATH, json=UPDATE)
    assert response.status_code == 200
    assert response.json()["updated_by"] == "admin-one"
    user.user_id = "admin-two"
    observed = http.get(PATH, params=IDENTITY).json()
    assert observed["status"] == "resolved"
    assert observed["last_seen_date"] == "2026-11-04"
    assert observed["last_seen_requests"] == 123
    assert observed["updated_by"] == "admin-one"


def test_reopen_does_not_change_another_models_resolution(client: tuple[TestClient, UserAPIKeyAuth]) -> None:
    http, _ = client
    http.put(PATH, json=UPDATE)
    other = {**UPDATE, "model": "openai/gpt-4-turbo"}
    http.put(PATH, json=other)
    assert http.put(PATH, json={**UPDATE, "status": "open"}).json()["status"] == "open"
    assert http.get(PATH, params={**IDENTITY, "model": other["model"]}).json()["status"] == "resolved"


@pytest.mark.parametrize("method", ["get", "put"])
def test_internal_users_cannot_read_or_change_global_alerts(
    client: tuple[TestClient, UserAPIKeyAuth], method: str
) -> None:
    http, user = client
    user.user_role = LitellmUserRoles.INTERNAL_USER
    response = http.get(PATH, params=IDENTITY) if method == "get" else http.put(PATH, json=UPDATE)
    assert response.status_code == 403


def test_viewer_can_read_but_cannot_resolve_at_the_same_path(client: tuple[TestClient, UserAPIKeyAuth]) -> None:
    http, user = client
    user.user_role = LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    assert http.get(PATH, params=IDENTITY).status_code == 200
    assert http.put(PATH, json=UPDATE).status_code == 403
    assert http.get(PATH, params=IDENTITY).json()["status"] == "open"


@pytest.mark.parametrize(
    "field,value", [("status", "unknown"), ("last_seen_requests", -1), ("deprecation_date", "invalid"), ("model", "")]
)
def test_invalid_updates_are_rejected(client: tuple[TestClient, UserAPIKeyAuth], field: str, value: object) -> None:
    http, _ = client
    assert http.put(PATH, json={**UPDATE, field: value}).status_code == 422
    assert http.get(PATH, params=IDENTITY).json()["status"] == "open"


def test_tags_are_scoped_to_model_dates_and_the_visible_keys(client: tuple[TestClient, UserAPIKeyAuth]) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litellm.proxy.alerting_endpoints.model_deprecation_alerts import alert_tag_repository

    http, _ = client
    grouped = AsyncMock(return_value=[{"api_key": "reporting-one", "tag": "support"}])
    http.app.dependency_overrides[alert_tag_repository] = lambda: SimpleNamespace(
        table=SimpleNamespace(group_by=grouped)
    )
    response = http.get(
        "/alerting/model-deprecations/tags",
        params={
            "model": "openai/gpt-4-turbo",
            "api_keys": ["reporting-one", "reporting-two"],
            "start_date": "2026-10-06",
            "end_date": "2026-11-04",
        },
    )
    assert response.status_code == 200
    assert response.json() == [{"api_key": "reporting-one", "tag": "support"}]
    assert grouped.call_args.kwargs["where"] == {
        "api_key": {"in": ["reporting-one", "reporting-two"]},
        "model": "openai/gpt-4-turbo",
        "date": {"gte": "2026-10-06", "lte": "2026-11-04"},
        "api_requests": {"gt": 0},
        "tag": {"not": None},
    }


@pytest.mark.parametrize(
    "role,expected", [(LitellmUserRoles.INTERNAL_USER, 403), (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, 200)]
)
def test_tag_access_matches_alert_visibility(
    client: tuple[TestClient, UserAPIKeyAuth], role: LitellmUserRoles, expected: int
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litellm.proxy.alerting_endpoints.model_deprecation_alerts import alert_tag_repository

    http, user = client
    user.user_role = role
    grouped = AsyncMock(return_value=[])
    http.app.dependency_overrides[alert_tag_repository] = lambda: SimpleNamespace(
        table=SimpleNamespace(group_by=grouped)
    )
    response = http.get(
        "/alerting/model-deprecations/tags",
        params={"model": "test-model", "api_keys": ["one"], "start_date": "2026-10-06", "end_date": "2026-11-04"},
    )
    assert response.status_code == expected
    if expected == 403:
        grouped.assert_not_called()


def test_tag_query_rejects_more_than_100_keys_before_database_access(client: tuple[TestClient, UserAPIKeyAuth]) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litellm.proxy.alerting_endpoints.model_deprecation_alerts import (
        alert_tag_repository,
    )

    http, _ = client
    grouped = AsyncMock(return_value=[])
    http.app.dependency_overrides[alert_tag_repository] = lambda: SimpleNamespace(
        table=SimpleNamespace(group_by=grouped)
    )
    response = http.get(
        "/alerting/model-deprecations/tags",
        params={
            "model": "test-model",
            "api_keys": [f"reporting-{index}" for index in range(101)],
            "start_date": "2026-10-07",
            "end_date": "2026-11-05",
        },
    )
    assert response.status_code == 422
    grouped.assert_not_called()
