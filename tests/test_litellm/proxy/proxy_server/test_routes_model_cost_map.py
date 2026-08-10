"""Pin tests for proxy_server.py model cost map routes (PR3).

Routes covered:
- POST /reload/model_cost_map
- POST /schedule/model_cost_map_reload
- DELETE /schedule/model_cost_map_reload
- GET /schedule/model_cost_map_reload/status
- GET /model/cost_map/source
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock


from .conftest import VOLATILE_KEYS, normalize

# Some response bodies include a "timestamp" — extend the volatile set so
# dict-equality assertions remain stable.
_VOLATILE = VOLATILE_KEYS | frozenset({"timestamp"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attach_litellm_config(mock_prisma):
    """Attach a litellm_config table mock (not in conftest's _PRISMA_TABLES)."""
    table = MagicMock()
    table.find_unique = AsyncMock(return_value=None)
    table.find_first = AsyncMock(return_value=None)
    table.find_many = AsyncMock(return_value=[])
    table.upsert = AsyncMock()
    table.create = AsyncMock()
    table.update = AsyncMock()
    table.update_many = AsyncMock(return_value=1)
    table.delete = AsyncMock()
    table.delete_many = AsyncMock()
    mock_prisma.db.litellm_config = table
    return table


# ---------------------------------------------------------------------------
# POST /reload/model_cost_map
# ---------------------------------------------------------------------------


def test_reload_model_cost_map_happy(client, auth_as, monkeypatch, mock_prisma):
    """Admin can trigger a manual reload; handler returns model count + status."""
    from litellm.litellm_core_utils.get_model_cost_map import ModelCostMapReloaded
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _attach_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    fake_cost_map = {"gpt-4": {"input_cost": 0.03}, "gpt-3.5": {"input_cost": 0.002}}
    monkeypatch.setattr(
        "litellm.litellm_core_utils.get_model_cost_map.refetch_model_cost_map",
        AsyncMock(return_value=ModelCostMapReloaded(model_cost_map=fake_cost_map)),
    )
    monkeypatch.setattr("litellm.add_known_models", lambda model_cost_map=None: None)
    monkeypatch.setattr("litellm.model_cost", {}, raising=False)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server._invalidate_model_cost_lowercase_map",
        lambda: None,
        raising=False,
    )

    async def _fake_invalidate(name):
        return None

    monkeypatch.setattr(ps, "invalidate_config_param", _fake_invalidate)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/reload/model_cost_map")
    assert response.status_code == 200
    body = normalize(response.json(), volatile=_VOLATILE)
    assert body == {
        "message": "Price data reloaded successfully! 2 models updated.",
        "status": "success",
        "models_count": 2,
        "timestamp": "<VOLATILE>",
    }
    assert table.upsert.await_count == 1
    update_payload = table.upsert.await_args.kwargs["data"]["update"]
    assert set(update_payload) == {"last_run_at", "reload_revision"}
    assert update_payload["reload_revision"] == {"increment": 1}


def test_reload_model_cost_map_fetch_failure_502_keeps_map(
    client, auth_as, monkeypatch, mock_prisma
):
    """Fetch failure returns 502 with the reason; the pod's map and DB flag are untouched.

    Regression: the endpoint used to report success after silently swapping in
    the stale packaged backup.
    """
    from litellm.litellm_core_utils.get_model_cost_map import (
        ModelCostMapReloadUnavailable,
    )
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _attach_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    sentinel_map = {"existing-model": {"input_cost": 0.01}}
    monkeypatch.setattr("litellm.model_cost", sentinel_map, raising=False)
    monkeypatch.setattr(
        "litellm.litellm_core_utils.get_model_cost_map.refetch_model_cost_map",
        AsyncMock(
            return_value=ModelCostMapReloadUnavailable(
                reason="HTTP 429 from upstream (after 3 attempts)"
            )
        ),
    )

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/reload/model_cost_map")
    assert response.status_code == 502
    detail = response.json().get("detail", "")
    assert "HTTP 429 from upstream" in detail
    assert "Current pricing data was kept" in detail
    import litellm as litellm_module

    assert litellm_module.model_cost is sentinel_map
    assert table.upsert.await_count == 0


def test_reload_model_cost_map_not_admin_forbidden(client, auth_as):
    """Non-admin caller gets 403 with a role-specific detail."""
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post("/reload/model_cost_map")
    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")


def test_reload_model_cost_map_no_db_500(client, auth_as, monkeypatch):
    """Admin path but prisma_client is None — handler raises 500."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", None)
    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/reload/model_cost_map")
    assert response.status_code == 500
    assert "Database connection not available" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# POST /schedule/model_cost_map_reload
# ---------------------------------------------------------------------------


def test_schedule_model_cost_map_reload_happy(
    client, auth_as, monkeypatch, mock_prisma
):
    """Admin schedules a reload — handler upserts config and echoes interval."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _attach_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    async def _fake_invalidate(name):
        return None

    monkeypatch.setattr(ps, "invalidate_config_param", _fake_invalidate)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/schedule/model_cost_map_reload?hours=6")
    assert response.status_code == 200
    body = normalize(response.json(), volatile=_VOLATILE)
    assert body == {
        "message": "Model cost map reload scheduled for every 6 hours",
        "status": "success",
        "interval_hours": 6,
        "timestamp": "<VOLATILE>",
    }
    assert table.upsert.await_count == 1
    upsert_data = table.upsert.await_args.kwargs["data"]
    assert set(upsert_data["update"]) == {"param_value"}
    assert set(upsert_data["create"]) == {"param_name", "param_value"}


def test_schedule_model_cost_map_reload_invalid_hours(
    client, auth_as, monkeypatch, mock_prisma
):
    """hours <= 0 is rejected with 400."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _attach_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/schedule/model_cost_map_reload?hours=0")
    assert response.status_code == 400
    assert "Hours must be greater than 0" in response.json().get("detail", "")


def test_schedule_model_cost_map_reload_not_admin_forbidden(client, auth_as):
    """Non-admin caller blocked with 403."""
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post("/schedule/model_cost_map_reload?hours=6")
    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# DELETE /schedule/model_cost_map_reload
# ---------------------------------------------------------------------------


def test_cancel_model_cost_map_reload_happy(client, auth_as, monkeypatch, mock_prisma):
    """Admin cancellation clears the interval and returns success body. The row itself stays:
    it also holds the reload revision, and a counter restarted by a delete reissues a number
    pods already applied, silently skipping their next manual reload."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _attach_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.delete("/schedule/model_cost_map_reload")
    assert response.status_code == 200
    body = normalize(response.json(), volatile=_VOLATILE)
    assert body == {
        "message": "Model cost map reload schedule cancelled",
        "status": "success",
        "timestamp": "<VOLATILE>",
    }
    assert json.loads(table.update_many.await_args.kwargs["data"]["param_value"]) == {"interval_hours": None}
    assert table.delete.await_count == 0


def test_cancel_model_cost_map_reload_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.delete("/schedule/model_cost_map_reload")
    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")


def test_cancel_model_cost_map_reload_no_db_500(client, auth_as, monkeypatch):
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", None)
    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.delete("/schedule/model_cost_map_reload")
    assert response.status_code == 500
    assert "Database connection not available" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# GET /schedule/model_cost_map_reload/status
# ---------------------------------------------------------------------------


def test_get_model_cost_map_reload_status_no_db_not_scheduled(
    client, auth_as, monkeypatch
):
    """No prisma client → returns the not-scheduled shape (4 keys, all-null)."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", None)
    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/model_cost_map_reload/status")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": False,
        "interval_hours": None,
        "last_run": None,
        "next_run": None,
    }


def test_get_model_cost_map_reload_status_scheduled(
    client, auth_as, monkeypatch, mock_prisma
):
    """A valid config row → scheduled=True and the interval is echoed."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _attach_litellm_config(mock_prisma)
    config_row = MagicMock()
    config_row.param_value = {"interval_hours": 12}
    config_row.reload_revision = 0
    config_row.last_run_at = None
    table.find_unique = AsyncMock(return_value=config_row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/model_cost_map_reload/status")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": True,
        "interval_hours": 12,
        "last_run": None,
        "next_run": None,
    }


def test_get_model_cost_map_reload_status_reports_persisted_last_run(
    client, auth_as, monkeypatch, mock_prisma
):
    """last_run/next_run come from the DB row, so status survives pod restarts."""
    from datetime import datetime, timezone

    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _attach_litellm_config(mock_prisma)
    config_row = MagicMock()
    config_row.param_value = {"interval_hours": 6}
    config_row.reload_revision = 0
    config_row.last_run_at = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    table.find_unique = AsyncMock(return_value=config_row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/model_cost_map_reload/status")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": True,
        "interval_hours": 6,
        "last_run": "2024-01-01T06:00:00+00:00",
        "next_run": "2024-01-01T12:00:00+00:00",
    }


def test_get_model_cost_map_reload_status_no_config_not_scheduled(
    client, auth_as, monkeypatch, mock_prisma
):
    """A row left behind by a manual reload (interval_hours=None) → not scheduled."""
    from datetime import datetime, timezone

    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _attach_litellm_config(mock_prisma)
    config_row = MagicMock()
    config_row.param_value = {"interval_hours": None}
    config_row.reload_revision = 3
    config_row.last_run_at = None
    table.find_unique = AsyncMock(return_value=config_row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/schedule/model_cost_map_reload/status")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "scheduled": False,
        "interval_hours": None,
        "last_run": None,
        "next_run": None,
    }


def test_get_model_cost_map_reload_status_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get("/schedule/model_cost_map_reload/status")
    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# GET /model/cost_map/source
# ---------------------------------------------------------------------------


def test_get_model_cost_map_source_happy(client, auth_as, monkeypatch):
    """Admin gets the source-info dict, augmented with the current model_count."""
    from litellm.proxy._types import LitellmUserRoles

    fake_info = {
        "source": "remote",
        "url": "https://example.invalid/cost_map.json",
        "is_env_forced": False,
        "fallback_reason": None,
    }
    monkeypatch.setattr(
        "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map_source_info",
        lambda: fake_info,
    )
    monkeypatch.setattr("litellm.model_cost", {"a": 1, "b": 2, "c": 3}, raising=False)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/model/cost_map/source")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "source": "remote",
        "url": "https://example.invalid/cost_map.json",
        "is_env_forced": False,
        "fallback_reason": None,
        "model_count": 3,
    }


def test_get_model_cost_map_source_admin_view_only_allowed(
    client, auth_as, monkeypatch
):
    """PROXY_ADMIN_VIEW_ONLY can read source info — pins the read-only ACL."""
    from litellm.proxy._types import LitellmUserRoles

    fake_info = {
        "source": "local",
        "url": None,
        "is_env_forced": True,
        "fallback_reason": None,
    }
    monkeypatch.setattr(
        "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map_source_info",
        lambda: fake_info,
    )
    monkeypatch.setattr("litellm.model_cost", {"a": 1}, raising=False)

    with auth_as(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        response = client.get("/model/cost_map/source")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "source": "local",
        "url": None,
        "is_env_forced": True,
        "fallback_reason": None,
        "model_count": 1,
    }


def test_get_model_cost_map_source_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get("/model/cost_map/source")
    assert response.status_code == 403
    assert "Admin role required" in response.json().get("detail", "")
