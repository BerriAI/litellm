"""
Tests for the invite-link onboarding endpoints.

Covers the security behavior of:
  GET  /onboarding/get_token   – rejects already-used links and returns only a
                                  short-lived onboarding token, not a UI session key
  POST /onboarding/claim_token – requires that onboarding token; mints the UI
                                  session key only after the password is written
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import InvitationClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AsyncTx:
    def __init__(self, db: MagicMock):
        self.db = db

    async def __aenter__(self) -> MagicMock:
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_invite(
    *, is_accepted: bool, expired: bool = False, claimed: bool = False
) -> MagicMock:
    now = litellm.utils.get_utc_datetime()
    invite = MagicMock()
    invite.id = "invite-abc"
    invite.user_id = "user-123"
    invite.is_accepted = is_accepted
    invite.expires_at = now - timedelta(days=1) if expired else now + timedelta(days=6)
    invite.accepted_at = now if claimed else None
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
    prisma.db.litellm_invitationlink.update_many = AsyncMock(return_value=1)
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user)
    prisma.db.litellm_usertable.update = AsyncMock(return_value=user)
    prisma.db.tx = MagicMock(return_value=_AsyncTx(prisma.db))
    return prisma


def _make_onboarding_token(
    *,
    invitation_link: str = "invite-abc",
    user_id: str = "user-123",
    token_type: str = "litellm_onboarding",
    master_key: str = "sk-test",
) -> str:
    return jwt.encode(
        {
            "token_type": token_type,
            "invitation_link": invitation_link,
            "user_id": user_id,
            "exp": litellm.utils.get_utc_datetime() + timedelta(minutes=15),
        },
        master_key,
        algorithm="HS256",
    )


def _make_claim_request(token: str | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    request.base_url = "http://localhost:4000/"
    return request


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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
    ):
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
    ):
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await onboarding(invite_link="nonexistent", request=request)

    assert exc_info.value.status_code == 401
    assert "does not exist" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_get_token_returns_onboarding_token_without_minting_ui_key():
    """
    A valid, unused link should return a short-lived onboarding token, but
    must not reserve the invite or mint a usable UI/API key on GET.
    """
    from litellm.proxy.proxy_server import onboarding

    invite = _make_invite(is_accepted=False)
    user = _make_user()
    prisma = _make_prisma(invite, user)
    request = MagicMock()
    request.base_url = "http://localhost:4000/"

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.premium_user", False),
        patch(
            "litellm.proxy.proxy_server.generate_key_helper_fn",
            new_callable=AsyncMock,
        ) as mock_generate_key,
        patch(
            "litellm.proxy.proxy_server.get_custom_url",
            return_value="http://localhost:4000/",
        ),
        patch(
            "litellm.proxy.proxy_server.get_disabled_non_admin_personal_key_creation",
            return_value=False,
        ),
        patch("litellm.proxy.proxy_server.get_server_root_path", return_value=""),
    ):
        result = await onboarding(invite_link="invite-abc", request=request)

    # Endpoint succeeded
    assert "token" in result
    assert "login_url" in result

    outer_claims = jwt.decode(result["token"], "sk-test", algorithms=["HS256"])
    onboarding_token = outer_claims["key"]
    onboarding_claims = jwt.decode(onboarding_token, "sk-test", algorithms=["HS256"])
    assert onboarding_claims["token_type"] == "litellm_onboarding"
    assert onboarding_claims["invitation_link"] == "invite-abc"
    assert onboarding_claims["user_id"] == "user-123"
    assert not onboarding_token.startswith("sk-")

    mock_generate_key.assert_not_called()
    prisma.db.litellm_invitationlink.update_many.assert_not_called()
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

    invite = _make_invite(is_accepted=True, claimed=True)
    prisma = _make_prisma(invite)
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data, request=_make_claim_request())

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
            await claim_onboarding_link(data=data, request=_make_claim_request())

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
            await claim_onboarding_link(data=data, request=_make_claim_request())

    assert exc_info.value.status_code == 401
    assert "does not match" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_claim_token_rejects_missing_onboarding_token():
    """The password endpoint must require the onboarding token returned by get_token."""
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    prisma = _make_prisma(invite)
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data, request=_make_claim_request())

    assert exc_info.value.status_code == 401
    assert "Missing onboarding session" in exc_info.value.detail["error"]
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_claim_token_rejects_wrong_onboarding_session():
    """The onboarding token must be bound to the invite and user being claimed."""
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    prisma = _make_prisma(invite)
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )
    request = _make_claim_request(
        _make_onboarding_token(invitation_link="other-invite")
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data, request=request)

    assert exc_info.value.status_code == 401
    assert "Invalid onboarding session" in exc_info.value.detail["error"]
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_claim_token_rejects_invalid_bearer_token():
    """A regular API key must not be accepted as an onboarding token."""
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    prisma = _make_prisma(invite)
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )
    request = _make_claim_request("sk-regular-key")

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data, request=request)

    assert exc_info.value.status_code == 401
    assert "Invalid onboarding session" in exc_info.value.detail["error"]
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_claim_token_rejects_concurrent_reuse_before_password_write():
    """Only the first valid claim may reserve the invitation."""
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    prisma = _make_prisma(invite)
    prisma.db.litellm_invitationlink.update_many = AsyncMock(return_value=0)
    request = _make_claim_request(_make_onboarding_token())
    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch(
            "litellm.proxy.proxy_server.generate_key_helper_fn",
            new_callable=AsyncMock,
        ) as mock_generate_key,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data, request=request)

    assert exc_info.value.status_code == 401
    assert "already been used" in exc_info.value.detail["error"]
    prisma.db.litellm_usertable.update.assert_not_called()
    mock_generate_key.assert_not_called()


@pytest.mark.asyncio
async def test_claim_token_sets_accepted_at_after_password_written():
    """
    A valid first-time claim must:
      1. Write the hashed password to the user table.
      2. Set accepted_at on the invitation link after the password write succeeds.
    """
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    user = _make_user()
    prisma = _make_prisma(invite, user)
    request = _make_claim_request(_make_onboarding_token())

    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    mock_token_response = {"token": "sk-generated-key", "user_id": "user-123"}

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.premium_user", False),
        patch(
            "litellm.proxy.proxy_server.generate_key_helper_fn",
            new_callable=AsyncMock,
            return_value=mock_token_response,
        ),
        patch(
            "litellm.proxy.proxy_server.get_custom_url",
            return_value="http://localhost:4000/",
        ),
        patch(
            "litellm.proxy.proxy_server.get_disabled_non_admin_personal_key_creation",
            return_value=False,
        ),
        patch("litellm.proxy.proxy_server.get_server_root_path", return_value=""),
    ):
        result = await claim_onboarding_link(data=data, request=request)

    # Password was written
    prisma.db.litellm_invitationlink.update_many.assert_called_once()
    reserve_kwargs = prisma.db.litellm_invitationlink.update_many.call_args.kwargs
    assert reserve_kwargs["where"] == {"id": "invite-abc", "is_accepted": False}
    assert reserve_kwargs["data"]["is_accepted"] is True
    prisma.db.litellm_usertable.update.assert_called_once()
    call_kwargs = prisma.db.litellm_usertable.update.call_args
    assert call_kwargs.kwargs["where"] == {"user_id": "user-123"}
    assert "password" in call_kwargs.kwargs["data"]

    # is_accepted was flipped to True on the invitation link
    prisma.db.litellm_invitationlink.update.assert_called_once()
    link_update_data = prisma.db.litellm_invitationlink.update.call_args.kwargs["data"]
    assert "is_accepted" not in link_update_data
    assert link_update_data["accepted_at"] is not None
    outer_claims = jwt.decode(result["token"], "sk-test", algorithms=["HS256"])
    assert outer_claims["key"] == "sk-generated-key"


@pytest.mark.asyncio
async def test_claim_token_rolls_back_invite_when_session_key_mint_fails():
    """A session key failure must not leave the invite permanently consumed."""
    from litellm.proxy.proxy_server import claim_onboarding_link

    invite = _make_invite(is_accepted=False)
    user = _make_user()
    prisma = _make_prisma(invite, user)
    request = _make_claim_request(_make_onboarding_token())

    data = InvitationClaim(
        invitation_link="invite-abc",
        user_id="user-123",
        password="NewP@ssw0rd",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.master_key", "sk-test"),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch(
            "litellm.proxy.proxy_server.generate_key_helper_fn",
            new_callable=AsyncMock,
            side_effect=Exception("key mint failed"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await claim_onboarding_link(data=data, request=request)

    assert exc_info.value.status_code == 500
    assert "Failed to create onboarding session" in exc_info.value.detail["error"]
    assert prisma.db.litellm_invitationlink.update_many.call_count == 2
    rollback_kwargs = prisma.db.litellm_invitationlink.update_many.call_args_list[
        1
    ].kwargs
    assert rollback_kwargs["where"] == {
        "id": "invite-abc",
        "is_accepted": True,
    }
    assert rollback_kwargs["data"]["accepted_at"] is None
    assert rollback_kwargs["data"]["is_accepted"] is False
