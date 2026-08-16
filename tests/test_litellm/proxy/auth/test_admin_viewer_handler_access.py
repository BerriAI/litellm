"""
Handler-level admin viewer parity tests.

These tests assert that PROXY_ADMIN_VIEW_ONLY callers are NOT blocked at the
handler level for read-only admin endpoints. The route_checks layer is tested
separately in `test_route_checks.py`; here we verify each individual endpoint
function has been updated to use `_user_has_admin_view()` rather than a bare
`user_role != PROXY_ADMIN` check.

The principle (see Admin Viewer role doc): anything Proxy Admin can read,
Admin Viewer can read. No writes, no cost-incurring actions.
"""

import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../"))

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app


def _make_admin_viewer_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )


def _override_auth(role: LitellmUserRoles) -> None:
    fake_user = UserAPIKeyAuth(user_id="viewer_user", user_role=role)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_user


def _clear_overrides() -> None:
    app.dependency_overrides.clear()


@pytest.fixture
def admin_viewer_client(monkeypatch):
    """TestClient where auth always returns PROXY_ADMIN_VIEW_ONLY + a mocked Prisma."""
    mock_prisma = MagicMock()

    # Common DB tables touched by the read endpoints under test.
    mock_budget_table = MagicMock()
    mock_budget_table.find_many = AsyncMock(return_value=[])
    mock_budget_table.find_first = AsyncMock(return_value=None)

    mock_invitation_table = MagicMock()
    mock_invitation_table.find_unique = AsyncMock(return_value=None)

    mock_config_table = MagicMock()
    mock_config_table.find_first = AsyncMock(return_value=None)

    mock_prisma.db = types.SimpleNamespace(
        litellm_budgettable=mock_budget_table,
        litellm_invitationlink=mock_invitation_table,
        litellm_config=mock_config_table,
    )

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    _override_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

    yield TestClient(app)

    _clear_overrides()


def _assert_not_role_blocked(response) -> None:
    """The endpoint must not return a role-block error.

    Detects both the 400 ``not_allowed_access`` pattern (used by most
    management endpoints) and the 403 ``Admin role required`` pattern
    (used by model cost map endpoints).
    """
    if response.status_code in (400, 401, 403):
        body = response.json()
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            err = detail.get("error", "")
        else:
            err = str(detail)
        err_lower = err.lower()
        role_block_signals = (
            "your role=",
            "not allowed to access",
            "admin role required",
            "admin-only endpoint",
        )
        for signal in role_block_signals:
            assert (
                signal not in err_lower
            ), f"endpoint blocked PROXY_ADMIN_VIEW_ONLY at handler level: {err}"


def test_budget_list_allows_admin_viewer(admin_viewer_client):
    """`/budget/list` is read-only and must be accessible to Admin Viewer."""
    resp = admin_viewer_client.get("/budget/list")
    _assert_not_role_blocked(resp)
    assert resp.status_code == 200, resp.text


def test_budget_settings_allows_admin_viewer(admin_viewer_client):
    """`/budget/settings` describes a budget's fields; read-only."""
    resp = admin_viewer_client.get("/budget/settings", params={"budget_id": "b1"})
    _assert_not_role_blocked(resp)
    assert resp.status_code == 200, resp.text


def test_alerting_settings_allows_admin_viewer(admin_viewer_client):
    """`/alerting/settings` describes alerting params; read-only."""
    resp = admin_viewer_client.get("/alerting/settings")
    _assert_not_role_blocked(resp)
    # Endpoint may 400 for *config* reasons (no proxy config loaded), but it
    # must not 400 because of role.
    assert resp.status_code != 403, resp.text


def test_get_config_field_info_allows_admin_viewer(admin_viewer_client):
    """`/config/field/info` describes a single general-settings field; read-only."""
    resp = admin_viewer_client.get(
        "/config/field/info", params={"field_name": "alerting"}
    )
    _assert_not_role_blocked(resp)


def test_get_config_list_allows_admin_viewer(admin_viewer_client):
    """`/config/list` lists configurable params for a config_type; read-only."""
    resp = admin_viewer_client.get(
        "/config/list", params={"config_type": "general_settings"}
    )
    _assert_not_role_blocked(resp)


def test_get_config_callbacks_allows_admin_viewer(admin_viewer_client):
    """`/get/config/callbacks` lists current callbacks; read-only."""
    resp = admin_viewer_client.get("/get/config/callbacks")
    _assert_not_role_blocked(resp)


def test_invitation_info_allows_admin_viewer(admin_viewer_client):
    """`/invitation/info` reads a single invitation; read-only.

    The invitation lookup will return 400 because no invitation exists in our
    mock DB — that's fine. We only assert it doesn't hit the role-block path.
    """
    resp = admin_viewer_client.get(
        "/invitation/info", params={"invitation_id": "nonexistent"}
    )
    _assert_not_role_blocked(resp)


def test_model_cost_map_reload_status_allows_admin_viewer(admin_viewer_client):
    """`/schedule/model_cost_map_reload/status` is read-only operations status."""
    resp = admin_viewer_client.get("/schedule/model_cost_map_reload/status")
    _assert_not_role_blocked(resp)


def test_model_cost_map_source_allows_admin_viewer(admin_viewer_client):
    """`/model/cost_map/source` reads the configured cost map source URL."""
    resp = admin_viewer_client.get("/model/cost_map/source")
    _assert_not_role_blocked(resp)
