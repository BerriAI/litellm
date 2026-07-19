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


def test_config_field_info_redacts_nested_secret_for_view_only_admin(
    client, auth_as, mock_prisma, monkeypatch
):
    """A view-only admin reading a structured field must not receive nested
    credentials. database_args carries aws_web_identity_token (a DynamoDB
    role-assumption credential); it must come back redacted while non-secret
    siblings like region_name stay visible."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    row = MagicMock()
    row.param_value = {
        "database_args": {
            "region_name": "us-east-1",
            "user_table_name": "LiteLLM_UserTable",
            "aws_web_identity_token": "sk-super-secret-token",
        }
    }
    table.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        response = client.get(
            "/config/field/info", params={"field_name": "database_args"}
        )
    assert response.status_code == 200
    value = response.json()["field_value"]
    assert value["aws_web_identity_token"] == "REDACTED"
    assert value["region_name"] == "us-east-1"
    assert value["user_table_name"] == "LiteLLM_UserTable"


def test_config_field_info_full_admin_sees_nested_secret(
    client, auth_as, mock_prisma, monkeypatch
):
    """The redaction must not over-redact for a full PROXY_ADMIN, who needs
    the real nested value to populate the edit form."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    row = MagicMock()
    row.param_value = {
        "database_args": {
            "region_name": "us-east-1",
            "aws_web_identity_token": "sk-super-secret-token",
        }
    }
    table.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get(
            "/config/field/info", params={"field_name": "database_args"}
        )
    assert response.status_code == 200
    value = response.json()["field_value"]
    assert value["aws_web_identity_token"] == "sk-super-secret-token"
    assert value["region_name"] == "us-east-1"


def test_config_field_info_redacts_top_level_scalar_for_view_only(
    client, auth_as, mock_prisma, monkeypatch
):
    """The top-level scalar branch must also redact for a view-only admin.
    database_url carries DB credentials and is not caught by the name masker,
    so it is in the explicit secret set."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    row = MagicMock()
    row.param_value = {"database_url": "postgresql://admin:p4ss@db:5432/litellm"}
    table.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        response = client.get(
            "/config/field/info", params={"field_name": "database_url"}
        )
    assert response.status_code == 200
    assert response.json()["field_value"] == "REDACTED"


def test_redact_general_setting_value_recurses_list_of_dicts():
    """The list branch of the recursor redacts secret leaves inside each dict
    while non-secret keys survive, and a full admin gets the value untouched."""
    from litellm.proxy import proxy_server as ps

    value = [
        {"path": "/foo", "headers": {"Authorization": "Bearer sk-x"}},
        {"path": "/bar", "client_secret": "sk-y"},
    ]
    redacted = ps._redact_general_setting_value(
        "some_list_field", value, is_full_admin=False
    )
    assert redacted[0]["headers"]["Authorization"] == "REDACTED"
    assert redacted[0]["path"] == "/foo"
    assert redacted[1]["client_secret"] == "REDACTED"
    assert redacted[1]["path"] == "/bar"
    assert (
        ps._redact_general_setting_value("some_list_field", value, is_full_admin=True)
        == value
    )


def test_redact_secret_values_in_obj_fails_closed_at_max_depth():
    """Past _REDACT_SECRET_MAX_DEPTH the whole subtree is replaced with
    "REDACTED" rather than returned verbatim, so a secret buried below the cap
    can never leak via depth-overrun. A future refactor that flips the cap
    branch to fail-open would surface here."""
    from litellm.proxy import proxy_server as ps

    # leaf and wrap keys are both NON-secret so neither the key-name
    # short-circuit nor the explicit-secret set catches the leak. The cap is
    # the only thing standing between the secret and the response — flip the
    # cap to fail-open and the secret comes back verbatim.
    nested: object = {"notes": "sk-leak-bottom"}
    for _ in range(ps._REDACT_SECRET_MAX_DEPTH + 2):
        nested = {"wrap": nested}

    out = ps._redact_general_setting_value(
        "some_struct_field", nested, is_full_admin=False
    )
    # the secret must not survive anywhere in the returned tree
    assert "sk-leak-bottom" not in repr(out)

    # full admin is unaffected by the cap — the value comes back untouched
    admin_out = ps._redact_general_setting_value(
        "some_struct_field", nested, is_full_admin=True
    )
    assert admin_out is nested


def test_config_list_redacts_pass_through_secret_for_view_only(
    client, auth_as, mock_prisma, monkeypatch
):
    """/config/list must not leak pass_through_endpoints upstream credentials
    to a view-only admin. pass_through_endpoints is a known secret-bearing
    field, so a non-admin gets it redacted; a full admin still sees it."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    table = _install_litellm_config(mock_prisma)
    row = MagicMock()
    row.param_value = {"max_parallel_requests": 3}
    table.find_first = AsyncMock(return_value=row)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(
        ps,
        "general_settings",
        {
            "pass_through_endpoints": [
                {
                    "path": "/foo",
                    "target": "https://upstream.example.com",
                    "headers": {"Authorization": "Bearer sk-UPSTREAM-SECRET"},
                }
            ]
        },
    )

    def _pass_through_value(body):
        return next(
            entry["field_value"]
            for entry in body
            if entry["field_name"] == "pass_through_endpoints"
        )

    with auth_as(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        view_resp = client.get(
            "/config/list", params={"config_type": "general_settings"}
        )
    assert view_resp.status_code == 200
    assert "sk-UPSTREAM-SECRET" not in view_resp.text
    assert _pass_through_value(view_resp.json()) == "REDACTED"

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        admin_resp = client.get(
            "/config/list", params={"config_type": "general_settings"}
        )
    assert admin_resp.status_code == 200
    admin_value = _pass_through_value(admin_resp.json())
    assert admin_value[0]["headers"]["Authorization"] == "Bearer sk-UPSTREAM-SECRET"


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


_CALLBACK_ENV_FIXTURE = {
    "LANGFUSE_PUBLIC_KEY": "pk-public-1234567890",
    "LANGFUSE_SECRET_KEY": "sk-langfuse-super-secret",
    "LANGFUSE_HOST": "https://cloud.langfuse.com",
    "DD_API_KEY": "dd-super-secret-api-key",
    "DD_SITE": "datadoghq.com",
    "OTEL_HEADERS": "Authorization=Bearer otel-super-secret",
    "OTEL_ENDPOINT": "https://otlp.example.com",
    "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T000/B000/SLACK-WEBHOOK-FIXTURE-SECRET",
}


def _install_callbacks_config(monkeypatch, mock_prisma):
    from litellm.proxy import proxy_server as ps

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "llm_router", None)

    fake_proxy_config = MagicMock()
    fake_proxy_config.get_config = AsyncMock(
        return_value={
            "litellm_settings": {"success_callback": ["langfuse", "datadog", "otel"]},
            "general_settings": {"alerting": ["slack"]},
            "environment_variables": dict(_CALLBACK_ENV_FIXTURE),
        }
    )
    monkeypatch.setattr(ps, "proxy_config", fake_proxy_config)


def _callback_variables(body: dict, name: str) -> dict:
    return next(
        cb["variables"] for cb in body["callbacks"] if cb["name"] == name
    )


def test_get_config_callbacks_redacts_secret_env_vars_for_view_only_admin(
    client, auth_as, mock_prisma, monkeypatch
):
    from litellm.proxy._types import LitellmUserRoles

    _install_callbacks_config(monkeypatch, mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        response = client.get("/get/config/callbacks")
    assert response.status_code == 200
    body = response.json()

    for secret in (
        _CALLBACK_ENV_FIXTURE["LANGFUSE_SECRET_KEY"],
        _CALLBACK_ENV_FIXTURE["DD_API_KEY"],
        _CALLBACK_ENV_FIXTURE["OTEL_HEADERS"],
        _CALLBACK_ENV_FIXTURE["LANGFUSE_PUBLIC_KEY"],
    ):
        assert secret not in response.text

    langfuse_vars = _callback_variables(body, "langfuse")
    assert langfuse_vars["LANGFUSE_PUBLIC_KEY"] == "REDACTED"
    assert langfuse_vars["LANGFUSE_SECRET_KEY"] == "REDACTED"
    assert langfuse_vars["LANGFUSE_HOST"] == _CALLBACK_ENV_FIXTURE["LANGFUSE_HOST"]

    datadog_vars = _callback_variables(body, "datadog")
    assert datadog_vars["DD_API_KEY"] == "REDACTED"
    assert datadog_vars["DD_SITE"] == _CALLBACK_ENV_FIXTURE["DD_SITE"]

    otel_vars = _callback_variables(body, "otel")
    assert otel_vars["OTEL_HEADERS"] == "REDACTED"
    assert otel_vars["OTEL_ENDPOINT"] == _CALLBACK_ENV_FIXTURE["OTEL_ENDPOINT"]


def test_get_config_callbacks_full_admin_still_sees_secret_env_vars(
    client, auth_as, mock_prisma, monkeypatch
):
    from litellm.proxy._types import LitellmUserRoles

    _install_callbacks_config(monkeypatch, mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/get/config/callbacks")
    assert response.status_code == 200
    body = response.json()

    langfuse_vars = _callback_variables(body, "langfuse")
    assert langfuse_vars["LANGFUSE_SECRET_KEY"] == _CALLBACK_ENV_FIXTURE["LANGFUSE_SECRET_KEY"]
    assert langfuse_vars["LANGFUSE_PUBLIC_KEY"] == _CALLBACK_ENV_FIXTURE["LANGFUSE_PUBLIC_KEY"]

    datadog_vars = _callback_variables(body, "datadog")
    assert datadog_vars["DD_API_KEY"] == _CALLBACK_ENV_FIXTURE["DD_API_KEY"]

    otel_vars = _callback_variables(body, "otel")
    assert otel_vars["OTEL_HEADERS"] == _CALLBACK_ENV_FIXTURE["OTEL_HEADERS"]


def test_get_config_callbacks_redacts_slack_webhook_urls_for_view_only_admin(
    client, auth_as, mock_prisma, monkeypatch
):
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_callbacks_config(monkeypatch, mock_prisma)

    webhooks = {
        "spend_reports": "https://hooks.slack.com/services/T000/B000/SPEND-WEBHOOK-SECRET",
        "budget_alerts": "https://hooks.slack.com/services/T000/B111/BUDGET-WEBHOOK-SECRET",
    }
    monkeypatch.setattr(
        ps.proxy_logging_obj.slack_alerting_instance,
        "alert_to_webhook_url",
        webhooks,
        raising=False,
    )

    def _slack_block(body):
        return next(a for a in body["alerts"] if a["name"] == "slack")

    with auth_as(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        view_resp = client.get("/get/config/callbacks")
    assert view_resp.status_code == 200
    for url in webhooks.values():
        assert url not in view_resp.text
    assert _CALLBACK_ENV_FIXTURE["SLACK_WEBHOOK_URL"] not in view_resp.text
    view_slack = _slack_block(view_resp.json())
    assert view_slack["alerts_to_webhook"] == {
        "spend_reports": "REDACTED",
        "budget_alerts": "REDACTED",
    }
    assert view_slack["variables"]["SLACK_WEBHOOK_URL"] == "REDACTED"

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        admin_resp = client.get("/get/config/callbacks")
    assert admin_resp.status_code == 200
    admin_slack = _slack_block(admin_resp.json())
    assert admin_slack["alerts_to_webhook"] == webhooks
    assert admin_slack["variables"]["SLACK_WEBHOOK_URL"] != "REDACTED"


def test_redact_callback_env_vars_helper_handles_none_and_non_secret_keys():
    from litellm.proxy import proxy_server as ps

    out = ps._redact_callback_env_vars(
        {
            "LANGFUSE_SECRET_KEY": "sk-leak",
            "LANGFUSE_HOST": "https://cloud.langfuse.com",
            "DD_API_KEY": None,
            "GALILEO_USERNAME": "galileo-user-1234",
            "GENERIC_LOGGER_HEADERS": "Authorization=Bearer x",
            "GCS_PATH_SERVICE_ACCOUNT": "/etc/secrets/gcs.json",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T/B/token",
            "SMTP_USERNAME": "smtp-user-1234",
        }
    )
    assert out == {
        "LANGFUSE_SECRET_KEY": "REDACTED",
        "LANGFUSE_HOST": "https://cloud.langfuse.com",
        "DD_API_KEY": None,
        "GALILEO_USERNAME": "REDACTED",
        "GENERIC_LOGGER_HEADERS": "REDACTED",
        "GCS_PATH_SERVICE_ACCOUNT": "REDACTED",
        "SLACK_WEBHOOK_URL": "REDACTED",
        "SMTP_USERNAME": "REDACTED",
    }


def test_get_config_callbacks_redacts_email_alerting_vars_for_view_only_admin(
    client, auth_as, mock_prisma, monkeypatch
):
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    _install_litellm_config(mock_prisma)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "llm_router", None)

    fake_proxy_config = MagicMock()
    fake_proxy_config.get_config = AsyncMock(
        return_value={
            "litellm_settings": {"success_callback": []},
            "general_settings": {"alerting": ["email"]},
            "environment_variables": {
                "SMTP_HOST": "smtp.resend.com",
                "SMTP_PORT": "587",
                "SMTP_USERNAME": "smtp-user-fixture-1234",
                "SMTP_PASSWORD": "smtp-password-fixture-1234",
                "SMTP_SENDER_EMAIL": "alerts@example.com",
                "TEST_EMAIL_ADDRESS": "admin@example.com",
                "EMAIL_LOGO_URL": "https://example.com/logo.png",
                "EMAIL_SUPPORT_CONTACT": "support@example.com",
            },
        }
    )
    monkeypatch.setattr(ps, "proxy_config", fake_proxy_config)

    def _email_block(body):
        return next(a for a in body["alerts"] if a["name"] == "email")

    with auth_as(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        view_resp = client.get("/get/config/callbacks")
    assert view_resp.status_code == 200
    for secret in ("smtp-user-fixture-1234", "smtp-password-fixture-1234"):
        assert secret not in view_resp.text
    view_email = _email_block(view_resp.json())["variables"]
    assert view_email["SMTP_PASSWORD"] == "REDACTED"
    assert view_email["SMTP_USERNAME"] == "REDACTED"
    assert view_email["SMTP_HOST"] == "smtp.resend.com"
    assert view_email["SMTP_PORT"] == "587"
    assert view_email["SMTP_SENDER_EMAIL"] == "alerts@example.com"

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        admin_resp = client.get("/get/config/callbacks")
    assert admin_resp.status_code == 200
    admin_email = _email_block(admin_resp.json())["variables"]
    assert admin_email["SMTP_USERNAME"] == "smtp-user-fixture-1234"
    assert admin_email["SMTP_PASSWORD"] != "REDACTED"
    assert admin_email["SMTP_HOST"] == "smtp.resend.com"


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
