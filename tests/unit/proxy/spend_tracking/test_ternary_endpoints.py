from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal

import polars as pl
import pytest
from pydantic import TypeAdapter
from fastapi.testclient import TestClient

import litellm
import litellm.proxy.proxy_server as ps
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.ternary.export_engine import FocusFrame
from litellm.integrations.ternary.ternary_logger import TernaryLogger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.ternary_endpoints import (
    TERNARY_SETTINGS_PARAM_NAME,
    TernarySettingsStore,
    register_ternary_logger_from_db,
    ternary_exporter,
    ternary_settings_store,
    ternary_usage_source,
)
from litellm.repositories.config_repository import ConfigParam
from litellm.types.proxy.ternary_endpoints import (
    TernaryDryRunResponse,
    TernaryExportResponse,
    TernarySettings,
    TernarySettingsView,
)

API_KEY = "tern_live_secret_key_value"
_TAGS = TypeAdapter(Mapping[str, str])
SETTINGS = TernarySettings(api_key=API_KEY, connection_id="conn-123", base_url="https://api.ternary.app")


class MemoryConfig:
    def __init__(self) -> None:
        self.rows: dict[str, object] = {}

    async def get_param(self, param_name: str) -> ConfigParam | None:
        if param_name not in self.rows:
            return None
        return ConfigParam(param_name=param_name, param_value=self.rows[param_name])

    async def set_param(self, param_name: str, param_value: object) -> ConfigParam:
        self.rows[param_name] = param_value
        return ConfigParam(param_name=param_name, param_value=param_value)

    async def delete_param(self, param_name: str) -> bool:
        return self.rows.pop(param_name, None) is not None


def _frame(columns: Mapping[str, Sequence[object]]) -> FocusFrame:
    return pl.DataFrame(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # polars is absent from the type-check env
        dict(columns)
    )


class UsageSource:
    def __init__(self, columns: Mapping[str, Sequence[object]]) -> None:
        self.columns = columns
        self.limits: list[int | None] = []

    async def get_usage_data(self, *, limit: int | None) -> FocusFrame:
        self.limits.append(limit)
        return _frame({name: values[:limit] for name, values in self.columns.items()})


class RecordingExporter:
    def __init__(self, error: Exception | None = None) -> None:
        self.logger = TernaryLogger(api_key=API_KEY, connection_id="conn-123", base_url="https://api.ternary.app")
        self.error = error
        self.calls: list[tuple[int | None, datetime | None, datetime | None]] = []

    def scheduled_window(self, now: datetime) -> FocusTimeWindow:
        return self.logger.scheduled_window(now)

    async def export_usage_data(
        self, *, limit: int | None, start_time_utc: datetime | None, end_time_utc: datetime | None
    ) -> None:
        self.calls.append((limit, start_time_utc, end_time_utc))
        if self.error is not None:
            raise self.error


def _usage_columns() -> Mapping[str, Sequence[object]]:
    return {
        "date": ["2026-10-13", "2026-10-14", "2026-10-14"],
        "spend": [Decimal("1.25"), Decimal("2.5"), Decimal("4.0")],
        "api_key": ["k1", "k2", "k3"],
        "api_key_alias": ["a1", "a2", "a3"],
        "model": ["gpt-5", "gpt-5-mini", "claude-opus-5-5"],
        "model_group": ["gpt-5", "gpt-5-mini", "claude-opus-5-5"],
        "custom_llm_provider": ["openai", "openai", "anthropic"],
        "team_id": ["t1", None, "t2"],
        "team_alias": ["T1", None, "T2"],
        "api_requests": [1, 2, 3],
        "prompt_tokens": [10, 20, 30],
        "completion_tokens": [1, 2, 3],
        "cache_read_input_tokens": [0, 5, 0],
        "cache_creation_input_tokens": [0, 0, 7],
    }


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-ternary-endpoint-tests")


@pytest.fixture
def config() -> MemoryConfig:
    return MemoryConfig()


@pytest.fixture
def store(config: MemoryConfig) -> TernarySettingsStore:
    return TernarySettingsStore(config)


@pytest.fixture
def role() -> list[LitellmUserRoles]:
    return [LitellmUserRoles.PROXY_ADMIN]


@pytest.fixture
def client(store: TernarySettingsStore, role: list[LitellmUserRoles]) -> Iterator[TestClient]:
    overrides = {
        ps.user_api_key_auth: lambda: UserAPIKeyAuth(user_role=role[0], user_id="u"),
        ternary_settings_store: lambda: store,
    }
    ps.app.dependency_overrides.update(overrides)
    try:
        yield TestClient(ps.app)
    finally:
        for dependency in overrides:
            ps.app.dependency_overrides.pop(dependency, None)


@pytest.fixture
def clean_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])


def _use(dependency: Callable[..., object], value: object) -> None:
    ps.app.dependency_overrides[dependency] = lambda: value


@pytest.fixture(autouse=True)
def drop_extra_overrides() -> Iterator[None]:
    yield
    for dependency in (ternary_usage_source, ternary_exporter):
        ps.app.dependency_overrides.pop(dependency, None)


@pytest.mark.asyncio
async def test_store_round_trips_settings_with_the_api_key_encrypted_at_rest(
    store: TernarySettingsStore, config: MemoryConfig
):
    await store.save(SETTINGS)

    stored = config.rows[TERNARY_SETTINGS_PARAM_NAME]
    assert API_KEY not in json.dumps(stored)
    assert await store.get() == SETTINGS


def test_init_then_get_shows_masked_settings(client: TestClient):
    init = client.post("/ternary/init", json={"api_key": API_KEY, "connection_id": "conn-123"})
    assert init.status_code == 200, init.text
    assert "Restart the proxy" in init.text

    view = TernarySettingsView.model_validate_json(client.get("/ternary/settings").content)

    assert (view.connection_id, view.base_url, view.frequency, view.status) == (
        "conn-123",
        "https://api.ternary.app",
        "daily",
        "configured",
    )
    assert view.api_key_masked is not None
    assert view.api_key_masked != API_KEY
    assert view.api_key_masked[:2] == API_KEY[:2]


def test_get_when_not_configured_returns_empty_view(client: TestClient):
    response = client.get("/ternary/settings")

    assert response.status_code == 200
    assert TernarySettingsView.model_validate_json(response.content) == TernarySettingsView()


@pytest.mark.parametrize(
    "body",
    [
        {"api_key": API_KEY, "connection_id": "conn/123"},
        {"api_key": API_KEY, "connection_id": "conn-123", "base_url": "http://api.ternary.app"},
    ],
)
def test_init_rejects_settings_the_destination_would_reject(
    client: TestClient, config: MemoryConfig, body: Mapping[str, object]
):
    response = client.post("/ternary/init", json=body)

    assert response.status_code == 400
    assert config.rows == {}


@pytest.mark.parametrize(
    "body",
    [
        {"api_key": " ", "connection_id": "conn-123"},
        {"api_key": API_KEY, "connection_id": "conn-123", "frequency": "hourly"},
        {"api_key": API_KEY, "connection_id": "conn-123", "frequency": "interval"},
    ],
)
def test_init_rejects_invalid_request_bodies(client: TestClient, config: MemoryConfig, body: Mapping[str, object]):
    assert client.post("/ternary/init", json=body).status_code == 422
    assert config.rows == {}


@pytest.mark.asyncio
async def test_update_keeps_the_stored_api_key_when_omitted(client: TestClient, store: TernarySettingsStore):
    await store.save(SETTINGS)

    response = client.put(
        "/ternary/settings", json={"connection_id": "conn-456", "frequency": "interval", "interval_seconds": 60}
    )

    assert response.status_code == 200, response.text
    assert await store.get() == SETTINGS.model_copy(
        update={"connection_id": "conn-456", "frequency": "interval", "interval_seconds": 60}
    )


@pytest.mark.asyncio
async def test_update_validates_the_merged_settings(client: TestClient, store: TernarySettingsStore):
    await store.save(SETTINGS)

    to_interval_without_seconds = client.put("/ternary/settings", json={"frequency": "interval"})
    to_insecure_url = client.put("/ternary/settings", json={"base_url": "http://example.com"})

    assert to_interval_without_seconds.status_code == 400
    assert to_insecure_url.status_code == 400
    assert await store.get() == SETTINGS


def test_update_without_settings_is_not_found(client: TestClient):
    assert client.put("/ternary/settings", json={"connection_id": "conn-456"}).status_code == 404


@pytest.mark.asyncio
async def test_update_with_no_fields_is_rejected(client: TestClient, store: TernarySettingsStore):
    await store.save(SETTINGS)

    assert client.put("/ternary/settings", json={}).status_code == 400


@pytest.mark.asyncio
async def test_delete_removes_settings_and_deregisters_the_logger(
    client: TestClient, store: TernarySettingsStore, clean_callbacks: None
):
    await store.save(SETTINGS)
    litellm.logging_callback_manager.add_litellm_callback(
        TernaryLogger(api_key=API_KEY, connection_id="conn-123", base_url="https://api.ternary.app")
    )

    response = client.delete("/ternary/delete")

    assert response.status_code == 200
    assert await store.get() is None
    assert not any(isinstance(cb, TernaryLogger) for cb in litellm.callbacks)


def test_delete_without_settings_is_not_found(client: TestClient):
    assert client.delete("/ternary/delete").status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/ternary/init", {"api_key": API_KEY, "connection_id": "conn-123"}),
        ("put", "/ternary/settings", {"connection_id": "conn-456"}),
        ("delete", "/ternary/delete", None),
        ("post", "/ternary/dry-run", {}),
        ("post", "/ternary/export", {}),
    ],
)
@pytest.mark.parametrize("role", [[LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY], [LitellmUserRoles.INTERNAL_USER]])
def test_writes_and_exports_are_proxy_admin_only(
    client: TestClient, config: MemoryConfig, method: str, path: str, body: Mapping[str, object] | None
):
    _use(ternary_usage_source, UsageSource(_usage_columns()))
    exporter = RecordingExporter()
    _use(ternary_exporter, exporter)

    response = client.request(method, path, json=body)

    assert response.status_code == 403
    assert config.rows == {}
    assert exporter.calls == []


@pytest.mark.parametrize("role", [[LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY]])
def test_admin_viewer_can_read_settings(client: TestClient):
    assert client.get("/ternary/settings").status_code == 200


@pytest.mark.parametrize("role", [[LitellmUserRoles.INTERNAL_USER]])
def test_internal_user_cannot_read_settings(client: TestClient):
    assert client.get("/ternary/settings").status_code == 403


def test_dry_run_previews_the_rows_ternary_would_receive(client: TestClient):
    source = UsageSource(_usage_columns())
    _use(ternary_usage_source, source)

    response = client.post("/ternary/dry-run", json={"limit": 2})

    assert response.status_code == 200, response.text
    body = TernaryDryRunResponse.model_validate_json(response.content)
    assert source.limits == [2]
    assert body.summary.model_dump() == {
        "total_records": 2,
        "estimated_spend": 3.75,
        "total_tokens": 33,
        "unique_teams": 1,
        "unique_models": 2,
        "days": ["2026-10-13", "2026-10-14"],
    }
    assert [row["spend"] for row in body.usage_data] == [1.25, 2.5]
    tags = [_TAGS.validate_json(str(row["Tags"])) for row in body.normalized_data]
    assert [t["prompt_tokens"] for t in tags] == ["10", "20"]
    assert [t["cache_read_input_tokens"] for t in tags] == ["0", "5"]


def test_dry_run_needs_no_settings(client: TestClient, config: MemoryConfig):
    _use(ternary_usage_source, UsageSource(_usage_columns()))

    assert client.post("/ternary/dry-run", json={}).status_code == 200
    assert config.rows == {}


def test_export_without_bounds_sends_the_scheduled_window(client: TestClient):
    exporter = RecordingExporter()
    _use(ternary_exporter, exporter)
    before = datetime.now(timezone.utc)

    response = client.post("/ternary/export", json={})

    after = datetime.now(timezone.utc)
    assert response.status_code == 200, response.text
    [(limit, start, end)] = exporter.calls
    assert limit is None
    assert start is not None
    assert end is not None
    assert start == exporter.scheduled_window(before).start_time
    assert (start.hour, start.minute, start.second, start.microsecond) == (0, 0, 0, 0)
    assert (before.date() - start.date()).days == 2
    assert before <= end <= after
    assert TernaryExportResponse.model_validate_json(response.content).start_time_utc == start


def test_export_passes_explicit_bounds_through(client: TestClient):
    exporter = RecordingExporter()
    _use(ternary_exporter, exporter)

    response = client.post(
        "/ternary/export",
        json={"start_time_utc": "2026-09-01T00:00:00", "end_time_utc": "2026-09-03T00:00:00Z", "limit": 100},
    )

    assert response.status_code == 200, response.text
    assert exporter.calls == [
        (100, datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 9, 3, tzinfo=timezone.utc))
    ]


@pytest.mark.parametrize(
    "body",
    [
        {"start_time_utc": "2026-09-01T00:00:00Z"},
        {"start_time_utc": "2026-09-03T00:00:00Z", "end_time_utc": "2026-09-01T00:00:00Z"},
    ],
)
def test_export_rejects_incomplete_or_reversed_bounds(client: TestClient, body: Mapping[str, object]):
    exporter = RecordingExporter()
    _use(ternary_exporter, exporter)

    assert client.post("/ternary/export", json=body).status_code == 422
    assert exporter.calls == []


def test_export_reports_failed_days_as_a_bad_gateway(client: TestClient):
    _use(ternary_exporter, RecordingExporter(RuntimeError("Ternary export: 1 day(s) failed to upload: 2026-10-14")))

    response = client.post("/ternary/export", json={})

    assert response.status_code == 502
    assert "2026-10-14" in response.text


def test_export_without_settings_is_not_found(client: TestClient, clean_callbacks: None):
    assert client.post("/ternary/export", json={}).status_code == 404


@pytest.mark.asyncio
async def test_exporter_is_built_from_stored_settings_when_none_is_registered(
    store: TernarySettingsStore, clean_callbacks: None
):
    await store.save(SETTINGS.model_copy(update={"frequency": "interval", "interval_seconds": 120}))

    exporter = await ternary_exporter(store)

    assert isinstance(exporter, TernaryLogger)
    assert exporter.frequency == "interval"
    assert exporter.interval_seconds == 120


@pytest.mark.asyncio
async def test_exporter_prefers_the_registered_logger(store: TernarySettingsStore, clean_callbacks: None):
    await store.save(SETTINGS)
    registered = TernaryLogger(api_key="other", connection_id="conn-999", base_url="https://api.ternary.app")
    litellm.logging_callback_manager.add_litellm_callback(registered)

    assert await ternary_exporter(store) is registered


@pytest.mark.asyncio
async def test_startup_registers_a_logger_from_stored_settings(store: TernarySettingsStore, clean_callbacks: None):
    await store.save(SETTINGS.model_copy(update={"frequency": "interval", "interval_seconds": 90}))

    await register_ternary_logger_from_db(store)

    [logger] = [cb for cb in litellm.callbacks if isinstance(cb, TernaryLogger)]
    assert (logger.frequency, logger.interval_seconds) == ("interval", 90)


@pytest.mark.asyncio
async def test_startup_leaves_config_file_setup_alone(store: TernarySettingsStore, clean_callbacks: None):
    await store.save(SETTINGS)
    litellm.callbacks.append("ternary")

    await register_ternary_logger_from_db(store)

    assert litellm.callbacks == ["ternary"]


@pytest.mark.asyncio
async def test_startup_survives_an_undecryptable_row(
    store: TernarySettingsStore, clean_callbacks: None, monkeypatch: pytest.MonkeyPatch
):
    await store.save(SETTINGS)
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-a-different-salt")

    await register_ternary_logger_from_db(store)

    assert litellm.callbacks == []
