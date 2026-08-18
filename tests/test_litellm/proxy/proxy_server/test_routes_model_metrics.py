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

from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles

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
