from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Final

from typing_extensions import LiteralString

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient, Response

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.telemetry.endpoints import (
    TelemetrySettingsResponse,
    router,
    telemetry_consent,
    telemetry_runtime_dependency,
    telemetry_sink,
    telemetry_store,
)
from litellm.proxy.telemetry.runtime import TelemetryRuntime
from litellm.proxy.telemetry.settings import TelemetrySettings
from litellm.proxy.telemetry.store import TelemetryStore
from litellm.telemetry.consent import TelemetryConsent
from litellm.telemetry.records import AttemptRecord, InstanceInfo, RequestRecord, TelemetryGroup, UIAction, UIEvent
from litellm.telemetry.sink import TelemetrySink
from tests.unit.proxy.telemetry.fake_database import SettingsDatabase


@dataclass
class _ReportTable:
    rows: tuple[Mapping[str, object], ...]
    queried_with: tuple[object, ...] | None = None

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        self.queried_with = args
        return self.rows

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        return 0


def _row(report_id: str, window_end: float) -> Mapping[str, object]:
    return {"id": report_id, "window_start": window_end - 60, "window_end": window_end, "report": {"requests": []}}


def _client(role: LitellmUserRoles, store: TelemetryStore | None) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[telemetry_store] = lambda: store
    return TestClient(app)


def test_a_full_page_returns_a_cursor_at_its_last_report() -> None:
    table: Final = _ReportTable(rows=(_row("a", 100.0), _row("b", 100.0)))
    response: Final = _client(LitellmUserRoles.PROXY_ADMIN, TelemetryStore(table, retention_days=30)).get(
        "/telemetry/reports", params={"after": 40.0, "after_id": "z", "limit": 2}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "reports": [_row("a", 100.0), _row("b", 100.0)],
        "next_after": 100.0,
        "next_after_id": "b",
    }
    assert table.queried_with == (40.0, "z", 2)


def test_a_short_page_is_the_last_one() -> None:
    table: Final = _ReportTable(rows=(_row("a", 100.0),))
    response: Final = _client(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, TelemetryStore(table, retention_days=30)).get(
        "/telemetry/reports", params={"limit": 2}
    )
    assert response.status_code == 200, response.text
    assert (response.json()["next_after"], response.json()["next_after_id"]) == (None, None)
    assert table.queried_with == (0.0, "", 2)


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.ORG_ADMIN, LitellmUserRoles.TEAM])
def test_non_admin_roles_cannot_export_reports(role: LitellmUserRoles) -> None:
    table: Final = _ReportTable(rows=(_row("a", 100.0),))
    response: Final = _client(role, TelemetryStore(table, retention_days=30)).get("/telemetry/reports")
    assert response.status_code == 403, response.text
    assert table.queried_with is None


def test_exporting_without_a_local_store_is_an_error() -> None:
    response: Final = _client(LitellmUserRoles.PROXY_ADMIN, None).get("/telemetry/reports")
    assert response.status_code == 500, response.text


@dataclass
class _UIEventSink:
    events: list[UIEvent] = field(default_factory=list)  # mutable-ok: records what the route forwarded

    def set_instance(self, info: InstanceInfo) -> None:
        pass

    def record_request(self, record: RequestRecord) -> None:
        pass

    def record_attempt(self, record: AttemptRecord) -> None:
        pass

    def record_ui_event(self, event: UIEvent) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        pass


def _ui_client(role: LitellmUserRoles, sink: TelemetrySink | None) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[telemetry_sink] = lambda: sink
    return TestClient(app)


@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.INTERNAL_USER])
def test_any_signed_in_ui_user_records_ui_events_into_the_sink(role: LitellmUserRoles) -> None:
    sink: Final = _UIEventSink()
    response: Final = _ui_client(role, sink).post(
        "/telemetry/ui_events", json={"page": "models-and-endpoints", "action": "click", "target": "tab=health"}
    )
    assert response.status_code == 204, response.text
    assert sink.events == [UIEvent(page="models-and-endpoints", action=UIAction.CLICK, target="tab=health")]


@pytest.mark.parametrize(
    "body",
    [
        {"page": "teams/abc-123", "action": "view"},
        {"page": "Teams", "action": "view"},
        {"page": "", "action": "view"},
        {"page": "teams", "action": "hover"},
        {"page": "teams", "action": "click", "target": "user@example.com"},
        {"page": "teams", "action": "click", "target": "tab=sk-abc123"},
        {"page": "teams", "action": "click", "target": "key=team-abc"},
        {"page": "teams", "action": "click", "target": f"tab={'a' * 45}"},
        {"page": "team-1234", "action": "view"},
        {"page": "a" * 49, "action": "view"},
        {"page": "teams", "action": "view", "team_id": "abc"},
    ],
)
def test_ui_events_outside_the_allowlisted_shape_are_rejected(body: Mapping[str, object]) -> None:
    sink: Final = _UIEventSink()
    response: Final = _ui_client(LitellmUserRoles.PROXY_ADMIN, sink).post("/telemetry/ui_events", json=body)
    assert response.status_code == 422, response.text
    assert sink.events == []


def test_ui_events_are_accepted_and_dropped_while_telemetry_is_off() -> None:
    response: Final = _ui_client(LitellmUserRoles.PROXY_ADMIN, None).post(
        "/telemetry/ui_events", json={"page": "teams", "action": "view"}
    )
    assert response.status_code == 204, response.text


def test_the_proxy_app_serves_the_export_route_from_its_own_telemetry_runtime() -> None:
    from litellm.proxy.proxy_server import app, telemetry_runtime

    assert telemetry_runtime.store is None
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    try:
        response: Final = TestClient(app).get("/telemetry/reports")
    finally:
        _ = app.dependency_overrides.pop(user_api_key_auth)
    assert response.status_code == 500, response.text
    assert response.json() == {"detail": CommonProxyErrors.db_not_connected_error.value}


def test_the_proxy_app_accepts_ui_events_while_its_telemetry_runtime_is_off() -> None:
    from litellm.proxy.proxy_server import app, telemetry_runtime

    assert telemetry_runtime.sink is None
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    try:
        response: Final = TestClient(app).post("/telemetry/ui_events", json={"page": "teams", "action": "view"})
    finally:
        _ = app.dependency_overrides.pop(user_api_key_auth)
    assert response.status_code == 204, response.text


async def _settings_client(
    role: LitellmUserRoles, settings: TelemetrySettings, db: SettingsDatabase | None
) -> tuple[AsyncClient, TelemetryRuntime]:
    runtime: Final = TelemetryRuntime()
    await runtime.start(litellm_version="1.0.0", settings=settings, db=lambda: db, register=lambda _logger: None)
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id="admin")
    app.dependency_overrides[telemetry_runtime_dependency] = lambda: runtime
    app.dependency_overrides[telemetry_store] = lambda: TelemetryStore(db, 30) if db is not None else None
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://proxy"), runtime


def _settings(response: Response) -> TelemetrySettingsResponse:
    assert response.status_code == 200, response.text
    return TelemetrySettingsResponse.model_validate_json(response.text)


def _enabled(response: Response) -> tuple[str, ...]:
    return tuple(info.group.value for info in _settings(response).groups if info.enabled)


@pytest.mark.asyncio
async def test_an_admin_can_store_groups_and_reads_them_back_with_a_sample_report() -> None:
    db: Final = SettingsDatabase()
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN, TelemetrySettings(), db)
    async with client:
        saved: Final = await client.put("/telemetry/settings", json={"groups": ["heartbeat", "request_success"]})
        read: Final = await client.get("/telemetry/settings")
    await runtime.stop()
    assert saved.status_code == 200, saved.text
    assert _enabled(read) == ("heartbeat", "request_success")
    body: Final = _settings(read)
    assert (body.editable, body.destination, body.report_is_sample) == (True, "local_table", True)
    assert body.report is not None and body.report["requests"]


@pytest.mark.parametrize(
    ("groups", "status"),
    [(["token_info"], 400), (["heartbeat", "everything"], 400)],
)
@pytest.mark.asyncio
async def test_invalid_groups_are_rejected_and_nothing_is_stored(groups: list[str], status: int) -> None:
    db: Final = SettingsDatabase()
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN, TelemetrySettings(), db)
    async with client:
        response: Final = await client.put("/telemetry/settings", json={"groups": groups})
    await runtime.stop()
    assert response.status_code == status, response.text
    assert db.stored_groups is None


@pytest.mark.asyncio
async def test_unknown_fields_in_the_update_are_rejected() -> None:
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN, TelemetrySettings(), SettingsDatabase())
    async with client:
        response: Final = await client.put("/telemetry/settings", json={"groups": [], "endpoint": "https://x"})
    await runtime.stop()
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_a_view_only_admin_can_read_but_not_change_the_settings() -> None:
    db: Final = SettingsDatabase()
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, TelemetrySettings(), db)
    async with client:
        read: Final = await client.get("/telemetry/settings")
        write: Final = await client.put("/telemetry/settings", json={"groups": ["heartbeat"]})
    await runtime.stop()
    assert (read.status_code, write.status_code) == (200, 403)
    assert _settings(read).editable is False
    assert db.stored_groups is None


@pytest.mark.asyncio
async def test_internal_users_cannot_read_the_settings() -> None:
    client, runtime = await _settings_client(LitellmUserRoles.INTERNAL_USER, TelemetrySettings(), SettingsDatabase())
    async with client:
        response: Final = await client.get("/telemetry/settings")
    await runtime.stop()
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_env_vars_lock_the_settings_and_are_listed_without_their_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_ENDPOINT", "https://user:secret@telemetry.example")
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"]}')
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN, TelemetrySettings(disabled=True), db)
    async with client:
        read: Final = await client.get("/telemetry/settings")
        write: Final = await client.put("/telemetry/settings", json={"groups": []})
    await runtime.stop()
    assert write.status_code == 409
    body: Final = _settings(read)
    assert (body.vetoed, body.editable, _enabled(read)) == (True, False, ())
    assert body.environment_variables == ("LITELLM_TELEMETRY_DISABLED", "LITELLM_TELEMETRY_ENDPOINT")
    assert "secret" not in read.text


@pytest.mark.parametrize(
    ("groups", "enabled"),
    [
        (frozenset({TelemetryGroup.HEARTBEAT, TelemetryGroup.PAGE_NAVIGATION}), True),
        (frozenset({TelemetryGroup.HEARTBEAT, TelemetryGroup.REQUEST_SUCCESS}), False),
        (frozenset[TelemetryGroup](), False),
    ],
)
def test_any_ui_user_can_ask_whether_page_navigation_events_are_wanted(
    groups: frozenset[TelemetryGroup], enabled: bool
) -> None:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    app.dependency_overrides[telemetry_consent] = lambda: TelemetryConsent(groups)
    response: Final = TestClient(app).get("/telemetry/ui_events/enabled")
    assert response.status_code == 200, response.text
    assert response.json() == {"enabled": enabled}


@pytest.mark.asyncio
async def test_a_vetoed_proxy_with_a_database_still_reports_the_local_table_as_its_destination() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"]}')
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN, TelemetrySettings(disabled=True), db)
    async with client:
        read: Final = await client.get("/telemetry/settings")
    await runtime.stop()
    assert (_settings(read).vetoed, _settings(read).destination) == (True, "local_table")


@pytest.mark.asyncio
async def test_a_failing_settings_read_is_a_503_instead_of_a_crash() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"]}')
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN, TelemetrySettings(), db)
    db.fail_reads = True
    async with client:
        response: Final = await client.get("/telemetry/settings")
    await runtime.stop()
    assert response.status_code == 503, response.text
    assert "db down" not in response.text


@pytest.mark.asyncio
async def test_the_settings_report_the_fixed_five_minute_window() -> None:
    client, runtime = await _settings_client(LitellmUserRoles.PROXY_ADMIN, TelemetrySettings(), SettingsDatabase())
    async with client:
        read: Final = await client.get("/telemetry/settings")
    await runtime.stop()
    assert _settings(read).flush_interval_seconds == 300
