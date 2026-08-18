"""Pin tests for proxy_server.py invitation routes (PR3).

Routes covered:
- POST /invitation/new
- GET /invitation/info
- POST /invitation/update
- POST /invitation/delete
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from .conftest import VOLATILE_KEYS, normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_invitation(
    invitation_id: str = "inv-abc",
    user_id: str = "user-target",
    created_by: str = "test-user-id",
    is_accepted: bool = False,
    accepted_at=None,
):
    """Build an invitation row with the fields ``InvitationModel`` requires.

    FastAPI serializes the returned object against ``response_model=InvitationModel``,
    so the object must expose ``id, user_id, is_accepted, accepted_at, expires_at,
    created_at, created_by, updated_at, updated_by`` either as attributes or
    dict keys.
    """
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=invitation_id,
        user_id=user_id,
        is_accepted=is_accepted,
        accepted_at=accepted_at,
        expires_at=now + timedelta(days=7),
        created_at=now,
        created_by=created_by,
        updated_at=now,
        updated_by=created_by,
    )


# ---------------------------------------------------------------------------
# POST /invitation/new
# ---------------------------------------------------------------------------


def test_invitation_new_admin_happy(client, auth_as, monkeypatch, mock_prisma):
    """Proxy admin → create_invitation_for_user returns invitation → 200."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_helpers import user_invitation

    invitation = _make_invitation(user_id="user-target")

    async def _fake_create_invitation(data, user_api_key_dict):
        return invitation

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(
        user_invitation, "create_invitation_for_user", _fake_create_invitation
    )

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/invitation/new", json={"user_id": "user-target"})

    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "user_id": "user-target",
        "is_accepted": False,
        "accepted_at": None,
        "expires_at": "<VOLATILE>",
        "created_at": "<VOLATILE>",
        "created_by": "test-user-id",
        "updated_at": "<VOLATILE>",
        "updated_by": "test-user-id",
    }


def test_invitation_new_non_admin_forbidden(client, auth_as, monkeypatch, mock_prisma):
    """Internal user without team/org admin privileges → 400 not-allowed."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints import common_utils

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    async def _no_privileges(**kwargs):
        return False

    # Patch at the proxy_server import site (used by the route).
    monkeypatch.setattr(ps, "_user_has_admin_privileges", _no_privileges)
    monkeypatch.setattr(common_utils, "_user_has_admin_privileges", _no_privileges)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post("/invitation/new", json={"user_id": "user-target"})

    assert response.status_code == 400
    err = response.json().get("error", response.json())
    err_text = str(err)
    assert "role=" in err_text or "not allowed" in err_text.lower()


def test_invitation_new_db_not_connected_400(client, auth_as, monkeypatch):
    """prisma_client is None → 400 db_not_connected_error."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", None)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/invitation/new", json={"user_id": "user-target"})

    assert response.status_code == 400
    body = response.json()
    err_text = str(body)
    # The handler wraps via handle_exception_on_proxy, so the error body
    # may take either the {"error": {...}} or {"detail": {...}} shape.
    assert "No connected db" in err_text or "db" in err_text.lower()


def test_invitation_new_missing_user_id_422(client, auth_as, monkeypatch, mock_prisma):
    """Body missing the required ``user_id`` field → FastAPI 422."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post("/invitation/new", json={})

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body.get("detail"), list)
    assert any("user_id" in str(item) for item in body["detail"])


# ---------------------------------------------------------------------------
# GET /invitation/info
# ---------------------------------------------------------------------------


def test_invitation_info_admin_happy(client, auth_as, monkeypatch, mock_prisma):
    """Admin requesting an existing invitation id → returns the invitation."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    invitation = _make_invitation(invitation_id="inv-xyz", user_id="user-target")
    mock_prisma.db.litellm_invitationlink.find_unique.return_value = invitation
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/invitation/info", params={"invitation_id": "inv-xyz"})

    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "user_id": "user-target",
        "is_accepted": False,
        "accepted_at": None,
        "expires_at": "<VOLATILE>",
        "created_at": "<VOLATILE>",
        "created_by": "test-user-id",
        "updated_at": "<VOLATILE>",
        "updated_by": "test-user-id",
    }


def test_invitation_info_not_admin_forbidden(client, auth_as, monkeypatch, mock_prisma):
    """Non-admin viewer (no admin-view privileges) → 400 not-allowed."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    # _user_has_admin_view is referenced from proxy_server's import.
    monkeypatch.setattr(ps, "_user_has_admin_view", lambda u: False)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get("/invitation/info", params={"invitation_id": "inv-xyz"})

    assert response.status_code == 400
    err_text = str(response.json())
    assert "role=" in err_text or "not allowed" in err_text.lower()


def test_invitation_info_not_found_400(client, auth_as, monkeypatch, mock_prisma):
    """Admin requesting an unknown invitation id → 400 does-not-exist."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    mock_prisma.db.litellm_invitationlink.find_unique.return_value = None
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get(
            "/invitation/info", params={"invitation_id": "does-not-exist"}
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {"error": "Invitation id does not exist in the database."}
    }


# ---------------------------------------------------------------------------
# POST /invitation/update
# ---------------------------------------------------------------------------


def test_invitation_update_happy(client, auth_as, monkeypatch, mock_prisma):
    """Authenticated user → invitation marked accepted → returns updated row."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    accepted = _make_invitation(
        invitation_id="inv-1",
        user_id="user-target",
        is_accepted=True,
        accepted_at=datetime.now(timezone.utc),
    )
    mock_prisma.db.litellm_invitationlink.update.return_value = accepted
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/invitation/update",
            json={"invitation_id": "inv-1", "is_accepted": True},
        )

    assert response.status_code == 200
    # ``accepted_at`` is a fresh timestamp on each run — extend volatile set.
    extended = VOLATILE_KEYS | {"accepted_at"}
    assert normalize(response.json(), extended) == {
        "id": "<VOLATILE>",
        "user_id": "user-target",
        "is_accepted": True,
        "accepted_at": "<VOLATILE>",
        "expires_at": "<VOLATILE>",
        "created_at": "<VOLATILE>",
        "created_by": "test-user-id",
        "updated_at": "<VOLATILE>",
        "updated_by": "test-user-id",
    }


def test_invitation_update_unknown_id_400(client, auth_as, monkeypatch, mock_prisma):
    """Update against an invitation id the DB returns None for → 400."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    mock_prisma.db.litellm_invitationlink.update.return_value = None
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/invitation/update",
            json={"invitation_id": "ghost", "is_accepted": True},
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {"error": "Invitation id does not exist in the database."}
    }


def test_invitation_update_no_user_id_500(client, auth_as, monkeypatch, mock_prisma):
    """If the auth principal lacks a user_id, handler returns 500."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN, user_id=None):
        response = client.post(
            "/invitation/update",
            json={"invitation_id": "inv-1", "is_accepted": True},
        )

    assert response.status_code == 500
    err_text = str(response.json())
    assert "Unable to identify user id" in err_text


# ---------------------------------------------------------------------------
# POST /invitation/delete
# ---------------------------------------------------------------------------


def test_invitation_delete_admin_happy(client, auth_as, monkeypatch, mock_prisma):
    """Proxy admin deletes by invitation_id → 200 with deleted row."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    deleted = _make_invitation(invitation_id="inv-del", user_id="user-target")
    mock_prisma.db.litellm_invitationlink.delete.return_value = deleted
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/invitation/delete", json={"invitation_id": "inv-del"}
        )

    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "user_id": "user-target",
        "is_accepted": False,
        "accepted_at": None,
        "expires_at": "<VOLATILE>",
        "created_at": "<VOLATILE>",
        "created_by": "test-user-id",
        "updated_at": "<VOLATILE>",
        "updated_by": "test-user-id",
    }


def test_invitation_delete_non_admin_forbidden(
    client, auth_as, monkeypatch, mock_prisma
):
    """Non-admin user without elevated privileges → 400 not-allowed."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    async def _no_privileges(**kwargs):
        return False

    monkeypatch.setattr(ps, "_user_has_admin_privileges", _no_privileges)

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.post(
            "/invitation/delete", json={"invitation_id": "inv-del"}
        )

    assert response.status_code == 400
    err_text = str(response.json())
    assert "role=" in err_text or "not allowed" in err_text.lower()


def test_invitation_delete_unknown_id_400(client, auth_as, monkeypatch, mock_prisma):
    """Delete returns None (no row) → 400 does-not-exist."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    mock_prisma.db.litellm_invitationlink.delete.return_value = None
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/invitation/delete", json={"invitation_id": "ghost"}
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {"error": "Invitation id does not exist in the database."}
    }


def test_invitation_delete_db_not_connected_400(client, auth_as, monkeypatch):
    """prisma_client is None → 400 db_not_connected_error."""
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(ps, "prisma_client", None)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.post(
            "/invitation/delete", json={"invitation_id": "inv-del"}
        )

    assert response.status_code == 400
    err_text = str(response.json())
    assert "No connected db" in err_text or "db" in err_text.lower()
