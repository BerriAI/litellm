"""Pin tests for proxy_server.py Anthropic-beta-headers reload routes (PR3).

Routes covered:
- POST /reload/anthropic_beta_headers
- POST /schedule/anthropic_beta_headers_reload
- DELETE /schedule/anthropic_beta_headers_reload
- GET /schedule/anthropic_beta_headers_reload/status
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


from .conftest import VOLATILE_KEYS, normalize

# These routes return a "timestamp" ISO string that isn't in the default
# volatile-keys set — extend the set locally so dict-equality assertions
# can ignore it.
_VOLATILE = VOLATILE_KEYS | frozenset({"timestamp"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prisma_with_config(
    config_record=None,
):
    """Build a MagicMock prisma_client with a ``db.litellm_config`` namespace.

    The conftest's ``mock_prisma`` fixture stubs ``litellm_configtable`` but
    the anthropic-beta routes use ``prisma_client.db.litellm_config`` —
    a different attribute. Build one here so each test gets isolated state.
    """
    config = MagicMock()
    config.find_unique = AsyncMock(return_value=config_record)
    config.upsert = AsyncMock()
    config.delete = AsyncMock()

    db = MagicMock()
    db.litellm_config = config

    client = MagicMock()
    client.db = db
    return client


def _install_prisma(monkeypatch, prisma):
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", prisma)


def _stub_reload_beta_headers(monkeypatch, return_value=None):
    """Replace ``litellm.anthropic_beta_headers_manager.reload_beta_headers_config``
    with a deterministic stub so the route never hits the network."""
    if return_value is None:
        return_value = {
            "anthropic": {"beta_headers": ["foo"]},
            "openai": {"beta_headers": ["bar"]},
            "provider_aliases": {"a": "b"},
            "description": "test",
        }
    import litellm.anthropic_beta_headers_manager as mgr

    stub = MagicMock(return_value=return_value)
    monkeypatch.setattr(mgr, "reload_beta_headers_config", stub)
    return stub


# ---------------------------------------------------------------------------
# POST /reload/anthropic_beta_headers
# ---------------------------------------------------------------------------


def test_reload_anthropic_beta_headers_admin_success(client, auth_as, monkeypatch):
    """Admin can trigger immediate reload — handler returns providers count and
    a success status. Pins the response dict shape."""
    from litellm.proxy._types import LitellmUserRoles

    _stub_reload_beta_headers(monkeypatch)
    prisma = _make_prisma_with_config(config_record=None)
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/reload/anthropic_beta_headers")

    assert response.status_code == 200
    body = response.json()
    # Two non-alias keys: "anthropic", "openai"
    assert normalize(body, _VOLATILE) == {
        "message": "Anthropic beta headers configuration reloaded successfully! 2 providers updated.",
        "status": "success",
        "providers_count": 2,
        "timestamp": "<VOLATILE>",
    }
    prisma.db.litellm_config.upsert.assert_awaited_once()


def test_reload_anthropic_beta_headers_preserves_existing_interval(
    client, auth_as, monkeypatch
):
    """The manual-reload write only stamps the job-owned timestamp columns; param_value
    (the admin-owned interval) is not in the payload, so it cannot be clobbered."""
    from datetime import datetime

    from litellm.proxy._types import LitellmUserRoles

    _stub_reload_beta_headers(monkeypatch)
    existing = SimpleNamespace(
        param_name="anthropic_beta_headers_reload_config",
        param_value={"interval_hours": 12},
        reload_requested_at=None,
        last_run_at=None,
    )
    prisma = _make_prisma_with_config(config_record=existing)
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/reload/anthropic_beta_headers")

    assert response.status_code == 200
    call_kwargs = prisma.db.litellm_config.upsert.await_args.kwargs
    update_payload = call_kwargs["data"]["update"]
    assert set(update_payload) == {"last_run_at", "reload_requested_at"}
    assert isinstance(update_payload["last_run_at"], datetime)
    assert update_payload["reload_requested_at"] == update_payload["last_run_at"]
    assert "param_value" not in call_kwargs["data"]["create"]


def test_reload_anthropic_beta_headers_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post("/reload/anthropic_beta_headers")

    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")


def test_reload_anthropic_beta_headers_no_db_returns_500(client, auth_as, monkeypatch):
    """When prisma_client is None the handler raises 500 with a clear message."""
    from litellm.proxy._types import LitellmUserRoles

    _install_prisma(monkeypatch, None)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/reload/anthropic_beta_headers")

    assert response.status_code == 500
    assert "Database connection not available" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# POST /schedule/anthropic_beta_headers_reload
# ---------------------------------------------------------------------------


def test_schedule_anthropic_beta_headers_reload_admin_success(
    client, auth_as, monkeypatch
):
    """Happy path: admin schedules every N hours — response echoes interval."""
    from litellm.proxy._types import LitellmUserRoles

    prisma = _make_prisma_with_config()
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/schedule/anthropic_beta_headers_reload", params={"hours": 6}
        )

    assert response.status_code == 200
    assert normalize(response.json(), _VOLATILE) == {
        "message": "Anthropic beta headers reload scheduled for every 6 hours",
        "status": "success",
        "interval_hours": 6,
        "timestamp": "<VOLATILE>",
    }
    prisma.db.litellm_config.upsert.assert_awaited_once()


def test_schedule_anthropic_beta_headers_reload_zero_hours_400(
    client, auth_as, monkeypatch
):
    """``hours <= 0`` is rejected with 400 and a descriptive message."""
    from litellm.proxy._types import LitellmUserRoles

    prisma = _make_prisma_with_config()
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/schedule/anthropic_beta_headers_reload", params={"hours": 0}
        )

    assert response.status_code == 400
    assert "Hours must be greater than 0" in response.json().get("detail", "")


def test_schedule_anthropic_beta_headers_reload_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post(
            "/schedule/anthropic_beta_headers_reload", params={"hours": 6}
        )

    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")


def test_schedule_anthropic_beta_headers_reload_missing_hours_422(client, auth_as):
    """``hours`` is a required query param — omitting it is a FastAPI 422."""
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/schedule/anthropic_beta_headers_reload")

    assert response.status_code == 422
    assert "detail" in response.json()


# ---------------------------------------------------------------------------
# DELETE /schedule/anthropic_beta_headers_reload
# ---------------------------------------------------------------------------


def test_cancel_anthropic_beta_headers_reload_admin_success(
    client, auth_as, monkeypatch
):
    """Admin cancel: deletes the LiteLLM_Config row and returns success dict."""
    from litellm.proxy._types import LitellmUserRoles

    prisma = _make_prisma_with_config()
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.delete("/schedule/anthropic_beta_headers_reload")

    assert response.status_code == 200
    assert normalize(response.json(), _VOLATILE) == {
        "message": "Anthropic beta headers reload schedule cancelled",
        "status": "success",
        "timestamp": "<VOLATILE>",
    }
    prisma.db.litellm_config.delete.assert_awaited_once_with(
        where={"param_name": "anthropic_beta_headers_reload_config"}
    )


def test_cancel_anthropic_beta_headers_reload_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.delete("/schedule/anthropic_beta_headers_reload")

    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")


def test_cancel_anthropic_beta_headers_reload_no_db_returns_500(
    client, auth_as, monkeypatch
):
    from litellm.proxy._types import LitellmUserRoles

    _install_prisma(monkeypatch, None)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.delete("/schedule/anthropic_beta_headers_reload")

    assert response.status_code == 500
    assert "Database connection not available" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# GET /schedule/anthropic_beta_headers_reload/status
# ---------------------------------------------------------------------------


def test_get_anthropic_beta_headers_reload_status_scheduled(
    client, auth_as, monkeypatch
):
    """When a config row with ``interval_hours`` is present, ``scheduled`` is True
    and ``interval_hours`` echoes the DB value. Pins the full response shape."""
    from litellm.proxy._types import LitellmUserRoles

    record = SimpleNamespace(
        param_name="anthropic_beta_headers_reload_config",
        param_value={"interval_hours": 6},
        reload_requested_at=None,
        last_run_at=None,
    )
    prisma = _make_prisma_with_config(config_record=record)
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/anthropic_beta_headers_reload/status")

    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": True,
        "interval_hours": 6,
        "last_run": None,
        "next_run": None,
    }


def test_get_anthropic_beta_headers_reload_status_reports_persisted_last_run(
    client, auth_as, monkeypatch
):
    """last_run/next_run come from the DB row, so status survives pod restarts."""
    from datetime import datetime, timezone

    from litellm.proxy._types import LitellmUserRoles

    record = SimpleNamespace(
        param_name="anthropic_beta_headers_reload_config",
        param_value={"interval_hours": 6},
        reload_requested_at=None,
        last_run_at=datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc),
    )
    prisma = _make_prisma_with_config(config_record=record)
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/anthropic_beta_headers_reload/status")

    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": True,
        "interval_hours": 6,
        "last_run": "2024-01-01T06:00:00+00:00",
        "next_run": "2024-01-01T12:00:00+00:00",
    }


def test_get_anthropic_beta_headers_reload_status_not_scheduled_no_db(
    client, auth_as, monkeypatch
):
    """No DB connection: handler returns the unscheduled-status dict (not 500)."""
    from litellm.proxy._types import LitellmUserRoles

    _install_prisma(monkeypatch, None)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/anthropic_beta_headers_reload/status")

    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": False,
        "interval_hours": None,
        "last_run": None,
        "next_run": None,
    }


def test_get_anthropic_beta_headers_reload_status_no_interval_unscheduled(
    client, auth_as, monkeypatch
):
    """Config row present but ``interval_hours`` is None → unscheduled response."""
    from datetime import datetime, timezone

    from litellm.proxy._types import LitellmUserRoles

    record = SimpleNamespace(
        param_name="anthropic_beta_headers_reload_config",
        param_value={"interval_hours": None},
        reload_requested_at=datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc),
        last_run_at=None,
    )
    prisma = _make_prisma_with_config(config_record=record)
    _install_prisma(monkeypatch, prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/anthropic_beta_headers_reload/status")

    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": False,
        "interval_hours": None,
        "last_run": None,
        "next_run": None,
    }


def test_get_anthropic_beta_headers_reload_status_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get("/schedule/anthropic_beta_headers_reload/status")

    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")
