from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.alerting_endpoints.alerts import router
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.database import get_database
from litellm.repositories.database import Database
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from tests._support.native_database import native_database


class StoredAlert(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int = 5
    detection_key: str = "digest"
    type: str = "spend"
    resource_type: str = "team"
    resource_id: str = "example-team"
    status: str = "open"
    window_start: datetime = datetime(2026, 1, 8, tzinfo=timezone.utc)
    window_end: datetime = datetime(2026, 1, 9, tzinfo=timezone.utc)
    evidence: object = {"value": 20, "baseline": 2, "unit": "usd", "sample_size": 50}
    created_at: datetime = datetime(2026, 1, 9, tzinfo=timezone.utc)
    updated_at: datetime = datetime(2026, 1, 9, tzinfo=timezone.utc)
    updated_by: str | None = None


@pytest.fixture
def table() -> MagicMock:
    table: Final = MagicMock()
    table.find_many = AsyncMock(return_value=[StoredAlert()])
    table.find_unique = AsyncMock(return_value=None)
    table.update = AsyncMock(return_value=None)
    return table


@pytest.fixture
def app(table: MagicMock) -> FastAPI:
    app: Final = FastAPI()
    app.include_router(router)
    prisma: Final = MagicMock()
    prisma.writer_db.litellm_alert = table
    app.dependency_overrides[get_database] = lambda: Database(prisma, native_database())
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
)
@pytest.mark.parametrize("method,path", (("GET", "/alerts"), ("GET", "/alerts/5"), ("PATCH", "/alerts/5")))
def test_permissions_precede_database_access(
    app: FastAPI, table: MagicMock, role: LitellmUserRoles, method: str, path: str
) -> None:
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[get_database] = lambda: None
    with TestClient(app) as client:
        response: Final = client.request(method, path, json={"status": "dismissed"} if method == "PATCH" else None)
    assert response.status_code == (503 if role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY and method == "GET" else 403)
    table.find_many.assert_not_awaited()
    table.find_unique.assert_not_awaited()
    table.update.assert_not_awaited()


@pytest.mark.parametrize(
    "query", ("limit=0", "limit=101", "cursor=0", "cursor=9223372036854775808", "cursor=oops", "status=invalid")
)
def test_invalid_page_never_reads_database(client: TestClient, table: MagicMock, query: str) -> None:
    assert client.get(f"/alerts?{query}").status_code == 422
    table.find_many.assert_not_awaited()


def test_viewer_reads_saved_alert_with_string_id(app: FastAPI, table: MagicMock) -> None:
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    )
    with TestClient(app) as client:
        response: Final = client.get("/alerts?status=open&limit=1")
    assert response.status_code == 200, response.text
    expected: Final = StoredAlert().model_dump(mode="json", exclude={"detection_key"})
    assert response.json() == {"data": [{**expected, "id": "5"}], "next_cursor": None}


@pytest.mark.parametrize("method", ("GET", "PATCH"))
def test_missing_alert_returns_not_found(client: TestClient, method: str) -> None:
    response: Final = client.request(method, "/alerts/5", json={"status": "resolved"} if method == "PATCH" else None)
    assert response.status_code == 404
    assert response.json() == {"detail": "Alert not found"}


@pytest.mark.parametrize("body", ({"status": "unknown"}, {"status": "open", "evidence": {}}, {}))
def test_status_update_rejects_other_changes(client: TestClient, table: MagicMock, body: object) -> None:
    assert client.patch("/alerts/5", json=body).status_code == 422
    table.update.assert_not_awaited()
