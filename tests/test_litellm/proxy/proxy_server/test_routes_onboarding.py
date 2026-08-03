"""Pin tests for proxy_server.py onboarding routes (PR3).

Routes covered:
- GET /onboarding/get_token
- POST /onboarding/claim_token
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest

from .conftest import normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_invite(
    invite_id: str = "inv-123",
    user_id: str = "user-abc",
    expires_at: datetime | None = None,
    is_accepted: bool = False,
    accepted_at=None,
):
    """Build a fake invitation object with the attributes the handler reads."""
    if expires_at is None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    return SimpleNamespace(
        id=invite_id,
        user_id=user_id,
        expires_at=expires_at,
        is_accepted=is_accepted,
        accepted_at=accepted_at,
    )


def _make_user_obj(
    user_id: str = "user-abc",
    user_email: str = "alice@example.com",
    user_role: str = "internal_user",
):
    return SimpleNamespace(
        user_id=user_id,
        user_email=user_email,
        user_role=user_role,
        password=None,
    )


def _install_tx_context(mock_prisma):
    """Wire ``async with prisma_client.db.tx() as tx`` to return ``mock_prisma.db``.

    The handler runs the update inside a transaction; have ``tx`` yield a
    namespace that exposes the same tables as the outer client so its
    ``update_many`` / ``update`` calls hit our mocks.
    """
    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=mock_prisma.db)
    tx_cm.__aexit__ = AsyncMock(return_value=None)
    mock_prisma.db.tx = MagicMock(return_value=tx_cm)


# ---------------------------------------------------------------------------
# GET /onboarding/get_token
# ---------------------------------------------------------------------------


def test_onboarding_get_token_happy(client, monkeypatch, mock_prisma):
    """Valid invite link → returns dict with login_url, token, user_email."""
    from litellm.proxy import proxy_server as ps

    invite = _make_invite()
    user_obj = _make_user_obj()
    mock_prisma.db.litellm_invitationlink.find_unique.return_value = invite
    mock_prisma.db.litellm_usertable.find_unique.return_value = user_obj

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps, "premium_user", False)

    response = client.get("/onboarding/get_token", params={"invite_link": "inv-123"})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"login_url", "token", "user_email"}
    assert body["user_email"] == "alice@example.com"
    assert "ui/onboarding" in body["login_url"]
    assert "token=" in body["login_url"]
    # The JWT in body["token"] must decode with the master_key.
    decoded = jwt.decode(body["token"], "sk-master-test", algorithms=["HS256"])
    assert normalize(
        {
            "user_id": decoded["user_id"],
            "user_email": decoded["user_email"],
            "login_method": decoded["login_method"],
            "premium_user": decoded["premium_user"],
        }
    ) == {
        "user_id": "user-abc",
        "user_email": "alice@example.com",
        "login_method": "username_password",
        "premium_user": False,
    }


def test_onboarding_get_token_master_key_missing_500(client, monkeypatch, mock_prisma):
    """No master_key configured → 500 with the master_key error payload."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", None)
    monkeypatch.setattr(ps, "general_settings", {})

    response = client.get("/onboarding/get_token", params={"invite_link": "inv-123"})
    assert response.status_code == 500
    body = response.json()
    # ProxyException serializes to {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}
    err_blob = body.get("error", body)
    assert "Master Key not set" in str(err_blob)


def test_onboarding_get_token_invalid_invite_link_401(
    client, monkeypatch, mock_prisma
):
    """Unknown invite link → 401 with the not-in-db error message."""
    from litellm.proxy import proxy_server as ps

    mock_prisma.db.litellm_invitationlink.find_unique.return_value = None
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})

    response = client.get(
        "/onboarding/get_token", params={"invite_link": "does-not-exist"}
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": {"error": "Invitation link does not exist in db."}
    }


def test_onboarding_get_token_expired_invite_401(client, monkeypatch, mock_prisma):
    """Invite whose expires_at is in the past → 401 expired."""
    from litellm.proxy import proxy_server as ps

    expired_invite = _make_invite(
        expires_at=datetime.now(timezone.utc) - timedelta(days=2)
    )
    mock_prisma.db.litellm_invitationlink.find_unique.return_value = expired_invite

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})

    response = client.get("/onboarding/get_token", params={"invite_link": "inv-123"})
    assert response.status_code == 401
    assert response.json().get("detail", {}).get("error") == "Invitation link has expired."


def test_onboarding_get_token_missing_query_param_422(client, monkeypatch, mock_prisma):
    """No ``invite_link`` query param → FastAPI 422 with a non-empty detail array."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})

    response = client.get("/onboarding/get_token")
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body.get("detail"), list)
    assert len(body["detail"]) >= 1


# ---------------------------------------------------------------------------
# POST /onboarding/claim_token
# ---------------------------------------------------------------------------


def _make_onboarding_jwt(
    master_key: str,
    invitation_link: str = "inv-123",
    user_id: str = "user-abc",
    token_type: str = "litellm_onboarding",
) -> str:
    return jwt.encode(
        {
            "token_type": token_type,
            "invitation_link": invitation_link,
            "user_id": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        },
        master_key,
        algorithm="HS256",
    )


def test_claim_onboarding_link_happy(client, monkeypatch, mock_prisma):
    """Valid claim → returns login_url, token, user_email, user."""
    from litellm.proxy import proxy_server as ps

    invite = _make_invite()
    user_obj = _make_user_obj()
    mock_prisma.db.litellm_invitationlink.find_unique.return_value = invite
    mock_prisma.db.litellm_invitationlink.update_many.return_value = 1
    mock_prisma.db.litellm_invitationlink.update.return_value = invite
    mock_prisma.db.litellm_usertable.update.return_value = user_obj
    _install_tx_context(mock_prisma)

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps, "premium_user", False)

    # Avoid hitting generate_key_helper_fn (touches DB / many globals); patch
    # the helper directly so we focus on the route's own behavior.
    async def _fake_session_token(user_obj):
        return "session-jwt-token"

    monkeypatch.setattr(
        ps, "_generate_onboarding_ui_session_token", _fake_session_token
    )

    onboarding_jwt = _make_onboarding_jwt("sk-master-test")
    response = client.post(
        "/onboarding/claim_token",
        json={
            "invitation_link": "inv-123",
            "user_id": "user-abc",
            "password": "hunter2",
        },
        headers={"Authorization": f"Bearer {onboarding_jwt}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"login_url", "token", "user_email", "user"}
    assert body["token"] == "session-jwt-token"
    assert body["user_email"] == "alice@example.com"
    assert body["login_url"].endswith("/ui/?login=success")


def test_claim_onboarding_link_invalid_invite_401(client, monkeypatch, mock_prisma):
    """Unknown invite link → 401 with not-in-db error."""
    from litellm.proxy import proxy_server as ps

    mock_prisma.db.litellm_invitationlink.find_unique.return_value = None
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})

    response = client.post(
        "/onboarding/claim_token",
        json={
            "invitation_link": "missing",
            "user_id": "user-abc",
            "password": "hunter2",
        },
        headers={"Authorization": "Bearer irrelevant"},
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": {"error": "Invitation link does not exist in db."}
    }


def test_claim_onboarding_link_user_id_mismatch_401(
    client, monkeypatch, mock_prisma
):
    """Invitation belongs to a different user_id → 401 with mismatch error."""
    from litellm.proxy import proxy_server as ps

    invite = _make_invite(user_id="user-real-owner")
    mock_prisma.db.litellm_invitationlink.find_unique.return_value = invite
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})

    response = client.post(
        "/onboarding/claim_token",
        json={
            "invitation_link": "inv-123",
            "user_id": "user-attacker",
            "password": "hunter2",
        },
        headers={"Authorization": "Bearer irrelevant"},
    )
    assert response.status_code == 401
    err = response.json().get("detail", {}).get("error", "")
    assert "Invalid invitation link" in err
    assert "user-attacker" in err


def test_claim_onboarding_link_missing_field_422(client, monkeypatch, mock_prisma):
    """Missing required body field → FastAPI 422 with detail listing the missing field."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})

    # Missing "password"
    response = client.post(
        "/onboarding/claim_token",
        json={"invitation_link": "inv-123", "user_id": "user-abc"},
    )
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body.get("detail"), list)
    # The missing field should be referenced in the validation error.
    assert any("password" in str(item) for item in body["detail"])


def test_claim_onboarding_link_bad_onboarding_jwt_401(
    client, monkeypatch, mock_prisma
):
    """Onboarding JWT decodes but token_type / invitation_link don't match → 401."""
    from litellm.proxy import proxy_server as ps

    invite = _make_invite()
    mock_prisma.db.litellm_invitationlink.find_unique.return_value = invite
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "master_key", "sk-master-test")
    monkeypatch.setattr(ps, "general_settings", {})

    # Wrong token_type — handler rejects.
    bogus_jwt = _make_onboarding_jwt(
        "sk-master-test",
        token_type="not_onboarding",
    )
    response = client.post(
        "/onboarding/claim_token",
        json={
            "invitation_link": "inv-123",
            "user_id": "user-abc",
            "password": "hunter2",
        },
        headers={"Authorization": f"Bearer {bogus_jwt}"},
    )
    assert response.status_code == 401
    assert (
        response.json().get("detail", {}).get("error")
        == "Invalid onboarding session for invitation link."
    )
