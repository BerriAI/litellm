"""Behavior pins for ``proxy_server.py`` model-metrics routes.

Pins (PR2):
    - GET /model/streaming_metrics
    - GET /model/metrics
    - GET /model/metrics/slow_responses
    - GET /model/metrics/exceptions
    - GET /model/settings
    - GET /alerting/settings
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.config_resolvers.settings_rules import JsonValue
from litellm.proxy.config_resolvers.settings_store import SettingsStore

from .conftest import normalize  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prisma_with_query_raw(monkeypatch):
    pc = MagicMock()
    pc.db.query_raw = AsyncMock(return_value=[])
    monkeypatch.setattr(proxy_server, "prisma_client", pc)
    return pc


@pytest.fixture
def no_prisma(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    yield


# ---------------------------------------------------------------------------
# GET /model/streaming_metrics
# ---------------------------------------------------------------------------


def test_model_streaming_metrics_happy(client, auth_as, prisma_with_query_raw):
    """Pins ``GET /model/streaming_metrics`` (happy: empty data list).

    Drives the deterministic branch where ``query_raw`` returns an empty
    list; the handler should return the empty payload unchanged so the
    pin can rely on the exact response shape.
    """
    with auth_as():
        response = client.get(
            "/model/streaming_metrics", params={"_selected_model_group": "gpt-4"}
        )
    assert response.status_code == 200
    assert normalize(response.json()) == {"data": [], "all_api_bases": []}


def test_model_streaming_metrics_no_prisma_error(client, auth_as, no_prisma):
    """Pins ``GET /model/streaming_metrics`` (error: prisma not initialized)."""
    with auth_as():
        response = client.get("/model/streaming_metrics")
    assert response.status_code == 500
    assert response.content


# ---------------------------------------------------------------------------
# GET /model/metrics
# ---------------------------------------------------------------------------


def test_model_metrics_happy(client, auth_as, prisma_with_query_raw):
    """Pins ``GET /model/metrics`` (happy: empty result)."""
    with auth_as():
        response = client.get("/model/metrics")
    assert response.status_code == 200
    assert normalize(response.json()) == {"data": [], "all_api_bases": []}


def test_model_metrics_no_prisma_error(client, auth_as, no_prisma):
    """Pins ``GET /model/metrics`` (error: prisma not initialized)."""
    with auth_as():
        response = client.get("/model/metrics")
    assert response.status_code == 500
    assert response.content


# ---------------------------------------------------------------------------
# GET /model/metrics/slow_responses
# ---------------------------------------------------------------------------


def test_model_metrics_slow_responses_happy(
    client, auth_as, prisma_with_query_raw, monkeypatch
):
    """Pins ``GET /model/metrics/slow_responses`` (happy: empty list)."""
    logging_obj = MagicMock()
    logging_obj.slack_alerting_instance.alerting_threshold = 30
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging_obj)
    with auth_as():
        response = client.get("/model/metrics/slow_responses")
    assert response.status_code == 200
    assert normalize(response.json()) == []


def test_model_metrics_slow_responses_no_prisma(client, auth_as, no_prisma):
    """Pins ``GET /model/metrics/slow_responses`` (error: prisma not initialized)."""
    with auth_as():
        response = client.get("/model/metrics/slow_responses")
    assert response.status_code == 500
    assert response.content


# ---------------------------------------------------------------------------
# GET /model/metrics/exceptions
# ---------------------------------------------------------------------------


def test_model_metrics_exceptions_happy(client, auth_as, prisma_with_query_raw):
    """Pins ``GET /model/metrics/exceptions`` (happy: empty)."""
    with auth_as():
        response = client.get("/model/metrics/exceptions")
    assert response.status_code == 200
    assert normalize(response.json()) == {"data": [], "exception_types": []}


def test_model_metrics_exceptions_no_prisma(client, auth_as, no_prisma):
    """Pins ``GET /model/metrics/exceptions`` (error: prisma not initialized)."""
    with auth_as():
        response = client.get("/model/metrics/exceptions")
    assert response.status_code == 500
    assert response.content


# ---------------------------------------------------------------------------
# GET /model/settings
# ---------------------------------------------------------------------------


def test_model_settings_happy(client, auth_as, monkeypatch):
    """Pins ``GET /model/settings`` (happy)."""
    monkeypatch.setattr(litellm, "provider_list", ["openai"])
    monkeypatch.setattr(
        litellm,
        "get_provider_fields",
        lambda custom_llm_provider: [],
    )
    with auth_as():
        response = client.get("/model/settings")
    assert response.status_code == 200
    body = response.json()
    assert body == [{"name": "openai", "fields": []}]
    summary = {
        "status_code": response.status_code,
        "first_entry_name": body[0]["name"],
        "body_length": len(body),
    }
    assert summary == {
        "status_code": 200,
        "first_entry_name": "openai",
        "body_length": 1,
    }


def test_model_settings_method_not_allowed(client, auth_as):
    """Pins ``GET /model/settings`` (error: wrong method)."""
    with auth_as():
        response = client.post("/model/settings", json={})
    assert response.status_code == 405
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# GET /alerting/settings
# ---------------------------------------------------------------------------


def _alerting_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    yaml_values: Mapping[str, JsonValue],
    db_row: Mapping[str, JsonValue],
    live_args: Mapping[str, JsonValue],
) -> "SettingsStore":
    pc = MagicMock()
    row = MagicMock()
    row.param_value = db_row
    pc.db.litellm_config.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(proxy_server, "prisma_client", pc)

    logging_obj = MagicMock()
    args_model = MagicMock()
    args_model.model_dump = MagicMock(return_value=live_args)
    logging_obj.slack_alerting_instance.alerting_args = args_model
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging_obj)

    store = SettingsStore("general_settings")
    store.load_yaml(yaml_values)
    store.apply_db_row("general_settings", db_row)
    monkeypatch.setattr(proxy_server.proxy_config, "settings", store)
    monkeypatch.setattr(proxy_server, "general_settings", store)
    return store


def test_alerting_settings_reports_sources(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _alerting_client(
        monkeypatch,
        yaml_values={
            "alerting": ["slack"],
            "alerting_args": {"daily_report_frequency": 3, "report_check_interval": 300},
        },
        db_row={"alerting_args": {"daily_report_frequency": 7, "outage_alert_ttl": 4242}},
        live_args={"daily_report_frequency": 3},
    )

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")

    assert response.status_code == 200
    by_name = {entry["field_name"]: entry for entry in response.json()}

    assert by_name["slack_alerting"]["source"] == "config"
    assert by_name["daily_report_frequency"]["source"] == "config"
    assert by_name["report_check_interval"]["source"] == "config"
    assert by_name["outage_alert_ttl"]["source"] == "default"
    assert by_name["budget_alert_ttl"]["source"] == "default"


def test_alerting_settings_reports_db_source_when_the_file_omits_alerting_args(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _alerting_client(
        monkeypatch,
        yaml_values={"alerting": ["slack"]},
        db_row={
            "alerting_args": {
                "outage_alert_ttl": 4242,
                "region_outage_alert_ttl": [],
                "report_check_interval": None,
            }
        },
        live_args={"outage_alert_ttl": 4242},
    )

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")

    assert response.status_code == 200
    by_name = {entry["field_name"]: entry for entry in response.json()}

    assert store.source("alerting_args") == "db"
    assert by_name["outage_alert_ttl"]["source"] == "db"
    assert by_name["region_outage_alert_ttl"]["source"] == "db"
    assert by_name["report_check_interval"]["source"] == "db"
    assert by_name["budget_alert_ttl"]["source"] == "default"


def test_alerting_settings_reports_config_source_when_db_disagrees(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.config_resolvers import SettingsStore

    db_alerting_args = {"daily_report_frequency": 7}

    pc = MagicMock()
    row = MagicMock()
    row.param_value = {"alerting_args": db_alerting_args}
    pc.db.litellm_config.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(proxy_server, "prisma_client", pc)

    logging_obj = MagicMock()
    args_model = MagicMock()
    args_model.model_dump = MagicMock(return_value={"daily_report_frequency": 3})
    logging_obj.slack_alerting_instance.alerting_args = args_model
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging_obj)

    store = SettingsStore("general_settings")
    store.load_yaml({"alerting_args": {"daily_report_frequency": 3}})
    store.apply_db_row("general_settings", {"alerting_args": db_alerting_args})
    monkeypatch.setattr(proxy_server.proxy_config, "settings", store)
    monkeypatch.setattr(proxy_server, "general_settings", store)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")

    assert response.status_code == 200
    by_name = {entry["field_name"]: entry for entry in response.json()}
    assert store.source("alerting_args") == "config"
    assert by_name["daily_report_frequency"]["field_value"] == 3
    assert by_name["daily_report_frequency"]["source"] == "config"


@pytest.mark.parametrize("db_alerting_args", [None, []])
def test_alerting_settings_handles_empty_db_args(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    db_alerting_args: JsonValue,
) -> None:
    from litellm.proxy.config_resolvers import SettingsStore

    pc = MagicMock()
    row = MagicMock()
    row.param_value = {"alerting_args": db_alerting_args}
    pc.db.litellm_config.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(proxy_server, "prisma_client", pc)

    logging_obj = MagicMock()
    args_model = MagicMock()
    args_model.model_dump = MagicMock(return_value={})
    logging_obj.slack_alerting_instance.alerting_args = args_model
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging_obj)

    store = SettingsStore("general_settings")
    store.load_yaml({"alerting_args": {"report_check_interval": 300}})
    monkeypatch.setattr(proxy_server.proxy_config, "settings", store)
    monkeypatch.setattr(proxy_server, "general_settings", store)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")

    assert response.status_code == 200
    by_name = {entry["field_name"]: entry for entry in response.json()}
    assert by_name["report_check_interval"]["source"] == "config"
    assert by_name["budget_alert_ttl"]["source"] == "default"


@pytest.mark.parametrize(
    ("field_default", "expected"),
    [(43200, "default"), (None, "unset")],
)
def test_nested_setting_source_without_a_config_or_db_value(field_default: JsonValue, expected: str) -> None:
    store = SettingsStore("general_settings")
    store.load_yaml({})

    assert (
        proxy_server._nested_setting_source(store, {}, "alerting_args", "budget_alert_ttl", field_default) == expected
    )


def test_alerting_settings_no_db_error(client, auth_as, no_prisma):
    """Pins ``GET /alerting/settings`` (error: db not connected)."""
    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")
    assert response.status_code == 400
    assert "error" in response.text or "detail" in response.text


def test_alerting_settings_non_admin_error(client, auth_as, monkeypatch):
    """Pins ``GET /alerting/settings`` (error: non-admin forbidden)."""
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get("/alerting/settings")
    assert response.status_code == 400
    assert "internal_user" in response.text.lower() or "error" in response.text


def test_alerting_settings_happy(client, auth_as, monkeypatch):
    """Pins ``GET /alerting/settings`` (happy: returns list of ConfigList entries)."""
    pc = MagicMock()
    pc.db.litellm_config.find_first = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", pc)

    logging_obj = MagicMock()
    args_model = MagicMock()
    args_model.model_dump = MagicMock(return_value={})
    logging_obj.slack_alerting_instance.alerting_args = args_model
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging_obj)
    monkeypatch.setattr(proxy_server, "general_settings", {})

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["field_name"] == "slack_alerting"
    summary = {
        "status_code": response.status_code,
        "first_field_name": body[0]["field_name"],
        "first_field_value": body[0]["field_value"],
        "first_field_type": body[0]["field_type"],
    }
    assert summary == {
        "status_code": 200,
        "first_field_name": "slack_alerting",
        "first_field_value": False,
        "first_field_type": "Boolean",
    }

def test_alerting_settings_only_returns_explicitly_configured_values(client, auth_as, monkeypatch):
    pc = MagicMock()
    pc.db.litellm_config.find_first = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", pc)

    logging_obj = MagicMock()
    args_model = MagicMock()
    args_model.model_dump = MagicMock(return_value={"budget_alert_ttl": -30})
    logging_obj.slack_alerting_instance.alerting_args = args_model
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging_obj)
    settings = SettingsStore("general_settings")
    settings.load_yaml({})
    monkeypatch.setattr(proxy_server.proxy_config, "settings", settings)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")

    assert response.status_code == 200
    setting = next(item for item in response.json() if item["field_name"] == "budget_alert_ttl")
    assert setting["field_value"] is None
    assert setting["stored_in_db"] is None


def test_alerting_settings_returns_configured_values(client, auth_as, monkeypatch):
    pc = MagicMock()
    pc.db.litellm_config.find_first = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", pc)
    settings = SettingsStore("general_settings")
    settings.load_yaml({"alerting_args": {"budget_alert_ttl": 60}})
    monkeypatch.setattr(proxy_server.proxy_config, "settings", settings)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")

    assert response.status_code == 200
    setting = next(item for item in response.json() if item["field_name"] == "budget_alert_ttl")
    assert setting["field_value"] == 60
    assert setting["stored_in_db"] is False


def test_alerting_settings_returns_database_values(client, auth_as, monkeypatch):
    db_settings = MagicMock()
    db_settings.param_value = {"alerting_args": {"budget_alert_ttl": 60}}
    pc = MagicMock()
    pc.db.litellm_config.find_first = AsyncMock(return_value=db_settings)
    monkeypatch.setattr(proxy_server, "prisma_client", pc)
    settings = SettingsStore("general_settings")
    settings.load_yaml({})
    monkeypatch.setattr(proxy_server.proxy_config, "settings", settings)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/alerting/settings")

    assert response.status_code == 200
    setting = next(item for item in response.json() if item["field_name"] == "budget_alert_ttl")
    assert setting["field_value"] == 60
    assert setting["stored_in_db"] is True

@pytest.mark.parametrize(
    ("source", "expected_value", "expected_stored_in_db"),
    [
        ("config", 60, False),
        ("db", 90, True),
        ("unset", None, None),
        ("default", None, None),
    ],
)
def test_alerting_field_helpers_cover_each_source(
    source, expected_value, expected_stored_in_db
):
    config_values = {"budget_alert_ttl": 60}
    db_values = {"budget_alert_ttl": 90}

    assert (
        proxy_server._alerting_field_value(
            source,
            "budget_alert_ttl",
            config_values,
            db_values,
        )
        == expected_value
    )
    assert proxy_server._alerting_stored_in_db(source) is expected_stored_in_db


def test_alerting_field_response_uses_explicit_config_value():
    settings = SettingsStore("general_settings")
    settings.load_yaml({"alerting_args": {"budget_alert_ttl": 60}})
    field_info = proxy_server.SlackAlertingArgs.model_fields["budget_alert_ttl"]

    result = proxy_server._alerting_field_response(
        settings=settings,
        db_values={},
        config_values={"budget_alert_ttl": 60},
        allowed_args={"budget_alert_ttl": "Integer"},
        field_name="budget_alert_ttl",
        field_info=field_info,
    )

    assert result.field_name == "budget_alert_ttl"
    assert result.field_type == "Integer"
    assert result.field_value == 60
    assert result.stored_in_db is False
    assert result.source == "config"

