"""Pin tests for proxy_server.py control-plane config routes (PR3).

Routes covered:
- POST /config/update
- POST /config/field/update
- GET /config/field/info
- GET /config/list
- POST /config/field/delete
- POST /config/callback/delete
- GET /get/config/callbacks
- GET /config/yaml
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from .conftest import VOLATILE_KEYS, normalize


def _install_litellm_config(mock_prisma: MagicMock) -> MagicMock:
    """Ensure mock_prisma.db.litellm_config exists with async methods (the
    conftest only stubs ``litellm_configtable`` — this is a different table)."""
    table = MagicMock()
    table.find_unique = AsyncMock(return_value=None)
    table.find_first = AsyncMock(return_value=None)
    table.find_many = AsyncMock(return_value=[])
    table.create = AsyncMock()
    table.update = AsyncMock()
    table.upsert = AsyncMock(return_value=None)
    table.delete = AsyncMock()
    mock_prisma.db.litellm_config = table
    return table


# ---------------------------------------------------------------------------
# POST /config/update
# ---------------------------------------------------------------------------


def test_config_update_happy_admin(client, auth_as, mock_prisma, monkeypatch):
    """POST /config/update with admin role merges + upserts general_settings
    and returns the canonical success message."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    fake_proxy_config = MagicMock()
    fake_proxy_config.add_deployment = AsyncMock()
    monkeypatch.setattr(ps, "proxy_config", fake_proxy_config)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/update",
            json={"general_settings": {"alerting": ["slack"]}},
        )
    assert response.status_code == 200
    assert normalize(response.json()) == {"message": "Config updated successfully"}


def test_config_update_non_admin_forbidden(client, auth_as, mock_prisma, monkeypatch):
    """POST /config/update by a non-admin caller is rejected; the error
    surfaces as a ProxyException with the admin-only message."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post(
            "/config/update",
            json={"general_settings": {"alerting": ["slack"]}},
        )
    assert response.status_code != 200
    body = response.json()
    # ProxyException wraps the 403 detail string in its `message` field.
    assert "admin" in str(body).lower() or "auth" in str(body).lower()


def test_config_update_no_db_error(client, auth_as, monkeypatch):
    """POST /config/update with prisma_client=None returns a 'No DB Connected'
    style error (the route raises Exception which the handler maps to 400)."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", None)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/update",
            json={"general_settings": {"alerting": ["slack"]}},
        )
    assert response.status_code != 200
    assert (
        "db" in str(response.json()).lower()
        or "connect" in str(response.json()).lower()
    )


# ---------------------------------------------------------------------------
# POST /config/field/update
# ---------------------------------------------------------------------------


def test_config_field_update_happy_admin(client, auth_as, mock_prisma, monkeypatch):
    """POST /config/field/update for a known field upserts the DB row and
    returns the upsert response (we pin it to a specific shape)."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    table.find_first = AsyncMock(return_value=None)
    upsert_row = {
        "param_name": "general_settings",
        "param_value": {"max_parallel_requests": 5},
        "id": "row-1",
    }
    table.upsert = AsyncMock(return_value=upsert_row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/field/update",
            json={
                "field_name": "max_parallel_requests",
                "field_value": 5,
                "config_type": "general_settings",
            },
        )
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "param_name": "general_settings",
        "param_value": {"max_parallel_requests": 5},
        "id": "<VOLATILE>",
    }


def test_config_field_update_non_admin_rejected(
    client, auth_as, mock_prisma, monkeypatch
):
    """Non-admin cannot update config fields — returns 400 with not-allowed
    detail (handler uses 400 for the auth gate, not 403)."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post(
            "/config/field/update",
            json={
                "field_name": "max_parallel_requests",
                "field_value": 5,
                "config_type": "general_settings",
            },
        )
    assert response.status_code == 400
    assert "error" in response.json().get("detail", {})


def test_config_field_update_invalid_field(client, auth_as, mock_prisma, monkeypatch):
    """Unknown field_name is rejected with 400 + 'Invalid field=' detail."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/field/update",
            json={
                "field_name": "not_a_real_field_xyz",
                "field_value": 1,
                "config_type": "general_settings",
            },
        )
    assert response.status_code == 400
    assert "Invalid field" in response.json().get("detail", {}).get("error", "")


# ---------------------------------------------------------------------------
# GET /config/field/info
# ---------------------------------------------------------------------------


def test_config_field_info_happy_admin(client, auth_as, mock_prisma, monkeypatch):
    """Admin gets back ConfigFieldInfo with the stored value pulled from DB."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    row = MagicMock()
    row.param_value = {"max_parallel_requests": 7}
    table.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get(
            "/config/field/info", params={"field_name": "max_parallel_requests"}
        )
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "field_name": "max_parallel_requests",
        "field_value": 7,
    }


def test_config_field_info_non_admin_rejected(
    client, auth_as, mock_prisma, monkeypatch
):
    """Non-admin (INTERNAL_USER) is denied — admin-view gate fires."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get(
            "/config/field/info", params={"field_name": "max_parallel_requests"}
        )
    assert response.status_code == 400
    assert "error" in response.json().get("detail", {})


def test_config_field_info_field_not_in_db(client, auth_as, mock_prisma, monkeypatch):
    """When the field is missing from the DB row, returns 400 'not in DB'."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    row = MagicMock()
    row.param_value = {"some_other_field": "value"}
    table.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get(
            "/config/field/info", params={"field_name": "max_parallel_requests"}
        )
    assert response.status_code == 400
    assert "not in DB" in response.json().get("detail", {}).get("error", "")


# ---------------------------------------------------------------------------
# GET /config/list
# ---------------------------------------------------------------------------


def test_config_list_happy_admin(client, auth_as, mock_prisma, monkeypatch):
    """Admin gets a non-empty list of ConfigList rows for general_settings
    (one entry per known allowed_arg). Each row has the documented schema."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    row = MagicMock()
    row.param_value = {"max_parallel_requests": 3}
    table.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get(
            "/config/list", params={"config_type": "general_settings"}
        )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0
    sample = body[0]
    shape = {
        "has_field_name": "field_name" in sample,
        "has_field_type": "field_type" in sample,
        "has_field_value": "field_value" in sample,
        "has_stored_in_db": "stored_in_db" in sample,
    }
    assert shape == {
        "has_field_name": True,
        "has_field_type": True,
        "has_field_value": True,
        "has_stored_in_db": True,
    }


def test_config_list_non_admin_rejected(client, auth_as, mock_prisma, monkeypatch):
    """Non-admin gets a 400 with the role embedded in the error message."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get(
            "/config/list", params={"config_type": "general_settings"}
        )
    assert response.status_code == 400
    assert "role" in response.json().get("detail", {}).get("error", "").lower()


def test_config_list_no_db_error(client, auth_as, monkeypatch):
    """No DB → 400 with db_not_connected error."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", None)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get(
            "/config/list", params={"config_type": "general_settings"}
        )
    assert response.status_code == 400
    assert "error" in response.json().get("detail", {})


# ---------------------------------------------------------------------------
# POST /config/field/delete
# ---------------------------------------------------------------------------


def test_config_field_delete_happy_admin(client, auth_as, mock_prisma, monkeypatch):
    """Admin can delete a stored general_settings field — returns the upsert row."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    existing = MagicMock()
    existing.param_value = {"max_parallel_requests": 5, "other": "value"}
    table.find_first = AsyncMock(return_value=existing)
    table.upsert = AsyncMock(
        return_value={
            "param_name": "general_settings",
            "param_value": {"other": "value"},
            "id": "row-1",
        }
    )
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/field/delete",
            json={
                "config_type": "general_settings",
                "field_name": "max_parallel_requests",
            },
        )
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "param_name": "general_settings",
        "param_value": {"other": "value"},
        "id": "<VOLATILE>",
    }


def test_config_field_delete_non_admin_rejected(
    client, auth_as, mock_prisma, monkeypatch
):
    """Non-admin caller hits the 400 not-allowed branch with role in detail."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post(
            "/config/field/delete",
            json={
                "config_type": "general_settings",
                "field_name": "max_parallel_requests",
            },
        )
    assert response.status_code == 400
    assert "role" in response.json().get("detail", {}).get("error", "").lower()


def test_config_field_delete_field_not_in_config(
    client, auth_as, mock_prisma, monkeypatch
):
    """If there is no general_settings row at all, returns 400 'not in config'."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    table.find_first = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/field/delete",
            json={
                "config_type": "general_settings",
                "field_name": "max_parallel_requests",
            },
        )
    assert response.status_code == 400
    assert "not in config" in response.json().get("detail", {}).get("error", "")


# ---------------------------------------------------------------------------
# POST /config/callback/delete
# ---------------------------------------------------------------------------


def test_config_callback_delete_happy_admin(client, auth_as, mock_prisma, monkeypatch):
    """Admin deletes a configured success callback — handler returns the
    success message + remaining callbacks + a timestamp."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "store_model_in_db", True)

    fake_proxy_config = MagicMock()
    fake_proxy_config.get_config = AsyncMock(
        return_value={"litellm_settings": {"success_callback": ["langfuse", "slack"]}}
    )
    fake_proxy_config.save_config = AsyncMock()
    fake_proxy_config.add_deployment = AsyncMock()
    monkeypatch.setattr(ps, "proxy_config", fake_proxy_config)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/callback/delete", json={"callback_name": "langfuse"}
        )
    assert response.status_code == 200
    # `deleted_at` is an ISO timestamp generated at request time — extend
    # the volatile set just for this assertion so dict-equality still works.
    volatile = VOLATILE_KEYS | {"deleted_at"}
    assert normalize(response.json(), volatile) == {
        "message": "Successfully deleted callback: langfuse",
        "removed_callback": "langfuse",
        "remaining_callbacks": ["slack"],
        "deleted_at": "<VOLATILE>",
    }


def test_config_callback_delete_non_admin_rejected(
    client, auth_as, mock_prisma, monkeypatch
):
    """Non-admin caller is rejected with 400 not-allowed."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "store_model_in_db", True)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post(
            "/config/callback/delete", json={"callback_name": "langfuse"}
        )
    assert response.status_code == 400
    assert "role" in response.json().get("detail", {}).get("error", "").lower()


def test_config_callback_delete_not_found(client, auth_as, mock_prisma, monkeypatch):
    """Callback missing from current config returns 404."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "store_model_in_db", True)

    fake_proxy_config = MagicMock()
    fake_proxy_config.get_config = AsyncMock(
        return_value={"litellm_settings": {"success_callback": ["slack"]}}
    )
    monkeypatch.setattr(ps, "proxy_config", fake_proxy_config)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/config/callback/delete", json={"callback_name": "langfuse"}
        )
    # The handler re-raises HTTPException(404) verbatim (only generic
    # `Exception` becomes a 500 ProxyException), so pin 404 strictly.
    assert response.status_code == 404
    assert (
        "langfuse" in str(response.json()).lower()
        or "not found" in str(response.json()).lower()
    )


# ---------------------------------------------------------------------------
# GET /get/config/callbacks
# ---------------------------------------------------------------------------


def test_get_config_callbacks_happy(client, auth_as, mock_prisma, monkeypatch):
    """GET /get/config/callbacks returns the 5 pinned top-level keys:
    status, callbacks, alerts, router_settings, available_callbacks."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "llm_router", None)

    fake_proxy_config = MagicMock()
    fake_proxy_config.get_config = AsyncMock(
        return_value={
            "litellm_settings": {"success_callback": []},
            "general_settings": {},
            "environment_variables": {},
        }
    )
    monkeypatch.setattr(ps, "proxy_config", fake_proxy_config)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/get/config/callbacks")
    assert response.status_code == 200
    body = response.json()
    shape = {
        "status": body.get("status"),
        "has_callbacks": "callbacks" in body,
        "has_alerts": "alerts" in body,
        "has_router_settings": "router_settings" in body,
        "has_available_callbacks": "available_callbacks" in body,
    }
    assert shape == {
        "status": "success",
        "has_callbacks": True,
        "has_alerts": True,
        "has_router_settings": True,
        "has_available_callbacks": True,
    }


def test_get_config_callbacks_internal_error(client, auth_as, mock_prisma, monkeypatch):
    """If proxy_config.get_config() raises, the handler wraps the failure in
    a ProxyException → non-2xx response with an error body."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    fake_proxy_config = MagicMock()
    fake_proxy_config.get_config = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(ps, "proxy_config", fake_proxy_config)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/get/config/callbacks")
    assert response.status_code >= 400
    assert (
        "boom" in str(response.json()).lower()
        or "error" in str(response.json()).lower()
    )


# ---------------------------------------------------------------------------
# GET /config/yaml
# ---------------------------------------------------------------------------


def test_config_yaml_returns_demo_payload(client, auth_as):
    """GET /config/yaml is documented as a mock endpoint. It declares
    ConfigYAML as the body parameter, so a GET with an empty JSON body is
    accepted and returns the canonical demo dict."""
    with auth_as():
        response = client.request("GET", "/config/yaml", json={})
    shape = {
        "status": response.status_code,
        "media_type_yaml": response.headers.get("content-type", "").startswith(
            "application/json"
        ),
        "has_body": len(response.content) > 0,
    }
    assert shape == {
        "status": 200,
        "media_type_yaml": True,
        "has_body": True,
    }
    assert response.json() == {"hello": "world"}


def test_config_yaml_invalid_method(client):
    """POST against the GET-only /config/yaml is rejected (error path)."""
    response = client.post("/config/yaml", json={})
    assert response.status_code == 405
    # Method-not-allowed responses still return a JSON-ish body via the
    # FastAPI default handler — assert the body is not the success payload.
    assert response.content != b'{"hello":"world"}'
