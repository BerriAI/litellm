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
from litellm.proxy.deps.repositories import get_alert_monitoring_repository
from litellm.repositories.database import Database
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository
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
def history_query() -> AsyncMock:
    return AsyncMock(return_value=[])


@pytest.fixture
def app(table: MagicMock, history_query: AsyncMock) -> FastAPI:
    app: Final = FastAPI()
    app.include_router(router)
    prisma: Final = MagicMock()
    prisma.writer_db.litellm_alert = table
    prisma.writer_db.litellm_config.find_unique = AsyncMock(return_value=None)
    prisma.writer_db.tx.return_value.__aenter__.return_value.query_raw = history_query
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
    "query",
    ("limit=0", "limit=101", "cursor=0", "cursor=9223372036854775808", "cursor=oops", "status=invalid", "type=invalid"),
)
def test_invalid_page_never_reads_database(client: TestClient, table: MagicMock, query: str) -> None:
    assert client.get(f"/alerts?{query}").status_code == 422
    table.find_many.assert_not_awaited()


@pytest.mark.parametrize(
    "role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.PROXY_ADMIN)
)
@pytest.mark.parametrize("method", ("GET", "PATCH"))
def test_monitoring_authorization_precedes_missing_database(app: FastAPI, role: LitellmUserRoles, method: str) -> None:
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[get_database] = lambda: None
    with TestClient(app) as client:
        response: Final = client.request(
            method, "/alerts/monitoring", json={"enabled": False} if method == "PATCH" else None
        )
    allowed: Final = role == LitellmUserRoles.PROXY_ADMIN or (
        role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY and method == "GET"
    )
    assert response.status_code == (503 if allowed else 403), response.text


@pytest.mark.parametrize(
    "body", ({"enabled": True, "alerts_used": 0}, {"enabled": True, "alert_limit": None}, {"enabled": "false"}, {})
)
def test_monitoring_only_accepts_a_boolean_opt_out(
    client: TestClient, history_query: AsyncMock, body: dict[str, object]
) -> None:
    response: Final = client.patch("/alerts/monitoring", json=body)
    assert response.status_code == 422, response.text
    history_query.assert_not_awaited()


@pytest.mark.parametrize("enabled", (False, True))
def test_deprecation_preference_updates_without_changing_performance(app: FastAPI, enabled: bool) -> None:
    prisma: Final = MagicMock()
    prisma.writer_db.litellm_config.find_unique = AsyncMock(return_value=MagicMock(param_value=enabled))
    prisma.writer_db.litellm_config.upsert = AsyncMock()
    transaction: Final = prisma.writer_db.tx.return_value.__aenter__.return_value
    transaction.query_raw = AsyncMock(side_effect=((), ({"enabled": False, "alerts_used": 3},)))
    transaction.execute_raw = AsyncMock()
    app.dependency_overrides[get_alert_monitoring_repository] = lambda: PerformanceAlertRepository(prisma)
    with TestClient(app) as client:
        response: Final = client.patch("/alerts/monitoring", json={"model_deprecations_enabled": enabled})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "enabled": False,
        "model_deprecations_enabled": enabled,
        "alerts_used": 3,
        "alert_limit": 5,
        "disabled_by_config": False,
    }
    prisma.writer_db.litellm_config.upsert.assert_awaited_once_with(
        where={"param_name": "model_deprecation_alerts_enabled"},
        data={
            "create": {"param_name": "model_deprecation_alerts_enabled", "param_value": str(enabled).lower()},
            "update": {"param_value": str(enabled).lower()},
        },
    )
    transaction.execute_raw.assert_not_awaited()


def test_config_opt_out_cannot_be_enabled_from_the_api(app: FastAPI) -> None:
    prisma: Final = MagicMock()
    prisma.writer_db.litellm_config.find_unique = AsyncMock(return_value=None)
    transaction: Final = prisma.writer_db.tx.return_value.__aenter__.return_value
    transaction.query_raw = AsyncMock(return_value=[{"enabled": True, "alerts_used": 0}])
    transaction.execute_raw = AsyncMock()
    app.dependency_overrides[get_alert_monitoring_repository] = lambda: PerformanceAlertRepository(
        prisma, configured=False
    )
    with TestClient(app) as client:
        status: Final = client.get("/alerts/monitoring")
        response: Final = client.patch("/alerts/monitoring", json={"enabled": True})
    assert status.json() == {
        "enabled": False,
        "alerts_used": 0,
        "alert_limit": 5,
        "disabled_by_config": True,
        "model_deprecations_enabled": True,
    }
    assert response.status_code == 409, response.text
    transaction.execute_raw.assert_not_awaited()


@pytest.mark.parametrize("alert_id", ("0", "-1", "01", "1.0", "oops", "9223372036854775808", "1" * 20))
@pytest.mark.parametrize("method", ("GET", "PATCH"))
def test_invalid_id_never_accesses_database(client: TestClient, table: MagicMock, alert_id: str, method: str) -> None:
    response: Final = client.request(
        method, f"/alerts/{alert_id}", json={"status": "resolved"} if method == "PATCH" else None
    )
    assert response.status_code == 422
    table.find_unique.assert_not_awaited()
    table.update.assert_not_awaited()


@pytest.mark.parametrize("alert_id", (9007199254740993, 9223372036854775807))
def test_large_ids_remain_exact_in_reads_updates_and_pagination(
    client: TestClient, table: MagicMock, alert_id: int
) -> None:
    row: Final = StoredAlert(id=alert_id)
    table.find_unique.return_value = row
    table.update.return_value = row.model_copy(update={"status": "resolved", "updated_by": "admin-user"})
    table.find_many.return_value = [row, StoredAlert(id=alert_id - 1)]
    detail: Final = client.get(f"/alerts/{alert_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["id"] == str(alert_id)
    table.find_unique.assert_awaited_once_with(where={"id": alert_id})
    updated: Final = client.patch(f"/alerts/{alert_id}", json={"status": "resolved"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["id"] == str(alert_id)
    assert updated.json()["status"] == "resolved"
    table.update.assert_awaited_once_with(
        where={"id": alert_id}, data={"status": "resolved", "updated_by": "admin-user"}
    )
    page: Final = client.get("/alerts?limit=1")
    assert page.status_code == 200, page.text
    assert page.json()["next_cursor"] == str(alert_id)
    following: Final = client.get("/alerts", params={"cursor": page.json()["next_cursor"], "limit": 1})
    assert following.status_code == 200, following.text
    table.find_many.assert_awaited_with(where={"status": "open", "id": {"lt": alert_id}}, order={"id": "desc"}, take=2)


def test_openapi_keeps_ids_and_cursors_as_strings(client: TestClient) -> None:
    spec: Final = client.get("/openapi.json").json()
    paths: Final = spec["paths"]
    for method in ("get", "patch"):
        parameter: Final = next(p for p in paths["/alerts/{alert_id}"][method]["parameters"] if p["name"] == "alert_id")
        assert parameter["schema"]["type"] == "string"
    cursor: Final = next(p for p in paths["/alerts"]["get"]["parameters"] if p["name"] == "cursor")
    assert {s["type"] for s in cursor["schema"]["anyOf"]} == {"string", "null"}
    schemas: Final = spec["components"]["schemas"]
    assert schemas["Alert"]["properties"]["id"]["type"] == "string"
    assert {s["type"] for s in schemas["AlertViewPage"]["properties"]["next_cursor"]["anyOf"]} == {"string", "null"}


def test_viewer_reads_saved_alert_with_string_id(app: FastAPI, table: MagicMock) -> None:
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    )
    with TestClient(app) as client:
        response: Final = client.get("/alerts?status=open&limit=1")
    assert response.status_code == 200, response.text
    expected: Final = StoredAlert().model_dump(mode="json", exclude={"detection_key"})
    assert response.json() == {
        "data": [{**expected, "id": "5", "resource_name": None, "history": []}],
        "next_cursor": None,
    }


@pytest.mark.parametrize("method", ("GET", "PATCH"))
def test_missing_alert_returns_not_found(client: TestClient, method: str) -> None:
    response: Final = client.request(method, "/alerts/5", json={"status": "resolved"} if method == "PATCH" else None)
    assert response.status_code == 404
    assert response.json() == {"detail": "Alert not found"}


@pytest.mark.parametrize("body", ({"status": "unknown"}, {"status": "open", "evidence": {}}, {}))
def test_status_update_rejects_other_changes(client: TestClient, table: MagicMock, body: object) -> None:
    assert client.patch("/alerts/5", json=body).status_code == 422
    table.update.assert_not_awaited()


def test_empty_alert_page_does_not_read_usage(client: TestClient, table: MagicMock, history_query: AsyncMock) -> None:
    table.find_many.return_value = []
    response: Final = client.get("/alerts")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}
    history_query.assert_not_awaited()


@pytest.mark.parametrize(
    "kind,unit,expected",
    (
        ("spend", "usd", 12),
        ("traffic", "requests", 20),
        ("latency", "milliseconds", 250),
        ("reliability", "ratio", 0.25),
    ),
)
def test_detail_returns_usage_history_without_replacing_saved_evidence(
    client: TestClient, table: MagicMock, history_query: AsyncMock, kind: str, unit: str, expected: float
) -> None:
    evidence: Final = {"value": 0.5, "baseline": 0.1, "unit": unit, "sample_size": 50}
    table.find_unique.return_value = StoredAlert(type=kind, evidence=evidence)
    history_query.return_value = [
        {
            "id": "5",
            "resource_name": "Research",
            "date": "2026-01-08",
            "spend": 12,
            "requests": 20,
            "failed": 5,
            "response_ms": 3750,
            "timed": 15,
        }
    ]
    response: Final = client.get("/alerts/5")
    assert response.status_code == 200
    assert response.json()["resource_name"] == "Research"
    assert response.json()["history"] == [{"date": "2026-01-08", "value": expected}]
    assert response.json()["evidence"] == evidence


@pytest.mark.parametrize("failure", (TimeoutError("private row value"), ValueError("private row value")))
def test_history_failure_keeps_alert_readable_and_logs_only_error_type(
    client: TestClient, table: MagicMock, history_query: AsyncMock, caplog: pytest.LogCaptureFixture, failure: Exception
) -> None:
    table.find_unique.return_value = StoredAlert()
    history_query.side_effect = failure
    response: Final = client.get("/alerts/5")
    assert response.status_code == 200
    assert response.json()["evidence"] == StoredAlert().evidence
    assert response.json()["history"] is None
    assert f"Alert Usage history unavailable ({type(failure).__name__})" in caplog.text
    assert "private row value" not in caplog.text


@pytest.mark.parametrize("kind,unit", (("latency", "milliseconds"), ("reliability", "ratio")))
def test_no_samples_leave_a_gap_in_history(
    client: TestClient, table: MagicMock, history_query: AsyncMock, kind: str, unit: str
) -> None:
    table.find_unique.return_value = StoredAlert(
        type=kind, evidence={"value": 0.5, "baseline": 0.1, "unit": unit, "sample_size": 50}
    )
    history_query.return_value = [
        {
            "id": "5",
            "resource_name": "Research",
            "date": "2026-01-08",
            "spend": 0,
            "requests": 0,
            "failed": 0,
            "response_ms": 0,
            "timed": 0,
        }
    ]
    response: Final = client.get("/alerts/5")
    assert response.status_code == 200
    assert response.json()["history"] == []
