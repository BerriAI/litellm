"""
Tests for the invite-link onboarding endpoints.

Covers the security behavior of:
  GET  /onboarding/get_token   – rejects already-used links before showing any user data
  POST /onboarding/claim_token – rejects already-used links; marks is_accepted=True only
                                  after the password is successfully written
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import InvitationClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_invite(*, is_accepted: bool, expired: bool = False) -> MagicMock:
    now = litellm.utils.get_utc_datetime()
    invite = MagicMock()
    invite.id = "invite-abc"
    invite.user_id = "user-123"
    invite.is_accepted = is_accepted
    invite.expires_at = now - timedelta(days=1) if expired else now + timedelta(days=6)
    invite.accepted_at = None
    return invite


def _make_user() -> MagicMock:
    user = MagicMock()
    user.user_id = "user-123"
    user.user_email = "alice@example.com"
    user.user_role = "internal_user"
    return user


def _make_prisma(invite: MagicMock, user: MagicMock | None = None) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_invitationlink.find_unique = AsyncMock(return_value=invite)
    prisma.db.litellm_invitationlink.update = AsyncMock()
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user)
    prisma.db.litellm_usertable.update = AsyncMock(return_value=user)
    return prisma


# ---------------------------------------------------------------------------
# GET /onboarding/get_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_token_rejects_already_used_link():
    """
    If is_accepted is True the link was already claimed.
    The endpoint must raise 401 *before* returning any user data.
    """
    from litellm.proxy.proxy_server import onboarding

    invite = _make_invite(is_accepted=True)
    prisma = _make_prisma(invite)
    request = MagicMock()

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.master_key", "sk-test"):
        with pytest.raises(HTTPException) as exc_info:
            await onboarding(invite_link="invite-abc", request=request)

    assert exc_info.value.status_code == 401
    assert "already been used" in exc_info.value.detail["error"]
    # The user table must never have been queried
    prisma.db.litellm_usertable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_get_token_rejects_expired_link():
    """An expired link must raise 401 regardless of is_accepted."""
    from litellm.proxy.proxy_server import onboarding

    invite = _make_invite(is_accepted=False, expired=True)
    prisma = _make_prisma(invite)
    request = MagicMock()

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.master_key", "sk-test"):
        with pytest.raises(HTTPException) as exc_info:
            await onboarding(invite_link="invite-abc", request=request)

    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_get_token_rejects_missing_link():
    """A link that does not exist in the DB must raise 401."""
    from litellm.proxy.proxy_server import onboarding

    prisma = _make_prisma(invite=None)  # type: ignore[arg-type]
    request = MagicMock()

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.master_key", "sk-test"):
        with pytest.raises(HTTPException) as exc_info:
            await onboarding(invite_link="nonexistent", request=request)

    assert exc_info.value.status_code == 401
    assert "does not exist" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_get_token_does_not_set_is_accepted():
    """
    A valid, unused link should succeed and must NOT flip is_accepted to True.
    That flag is only written after the password is claimed.
    """
    from litellm.proxy.proxy_server import onboarding

    invite = _make_invite(is_accepted=False)
    user = _make_user()
    prisma = _make_prisma(invite, user)
    request = MagicMock()
    request.base_url = "http://localhost:4000/"

    mock_token_response = {"token": "sk-generated-key", "user_id": "user-123"}

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.master_key", "sk-test"), \
         patch("litellm.proxy.proxy_server.general_settings", {}), \
         patch("litellm.proxy.proxy_server.premium_user", False), \
         patch(
             "litellm.proxy.proxy_server.generate_key_helper_fn",
             new_callable=AsyncMock,
             return_value=mock_token_response,
         ), \
         patch("litellm.proxy.proxy_server.get_custom_url", return_value="http://localhost:4000/"), \
         patch("litellm.proxy.proxy_server.get_disabled_non_admin_personal_key_creation", return_value=False), \
         patch("litellm.proxy.proxy_server.get_server_root_path", return_value=""):
        result = await onboarding(invite_link="invite-abc", request=request)

    # Endpoint succeeded
    assert "token" in result
    assert "login_url" in result

    # is_accepted must NOT have been updated here
    prisma.db.litellm_invitationlink.update.assert_not_called()


# ---------------------------------------------------------------------------
# POST /onboarding/claim_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_token_rejects_already_used_link():
    """
    If is_accepted is True, the password has already been set.
    A second claim attempt must be rejected with 401.
    """
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=True)
    prisma = _make_prisma(invite)
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data)

    assert exc_info.value.status_code == 401
    assert "already been used" in exc_info.value.detail["error"]
    # Password must never have been written
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_claim_token_rejects_expired_link():
    """An expired link must be rejected even if is_accepted is False."""
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False, expired=True)
    prisma = _make_prisma(invite)
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data)

    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_claim_token_rejects_mismatched_user_id():
    """The user_id in the request must match the one on the invite."""
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    prisma = _make_prisma(invite)
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="wrong-user",
        password="NewP@ssw0rd",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data)

    assert exc_info.value.status_code == 401
    assert "does not match" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_claim_token_sets_is_accepted_after_password_written():
    """
    A valid first-time claim must:
      1. Write the hashed password to the user table.
      2. Flip is_accepted to True on the invitation link — and only after the
         password write succeeds.
    """
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    user = _make_user()
    prisma = _make_prisma(invite, user)

    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        result = await claim_onboarding_link(data=data)

    # Password was written
    prisma.db.litellm_usertable.update.assert_called_once()
    call_kwargs = prisma.db.litellm_usertable.update.call_args
    assert call_kwargs.kwargs["where"] == {"user_id": "user-123"}
    assert "password" in call_kwargs.kwargs["data"]

    # is_accepted was flipped to True on the invitation link
    prisma.db.litellm_invitationlink.update.assert_called_once()
    link_update_data = prisma.db.litellm_invitationlink.update.call_args.kwargs["data"]
    assert link_update_data["is_accepted"] is True
    assert link_update_data["accepted_at"] is not None
