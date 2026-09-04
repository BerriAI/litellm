"""
Tests for POST /user/password/change (litellm/proxy/management_endpoints/password_endpoints.py).

HIBP traffic is intercepted with respx; no test here touches the network.
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi import HTTPException

from litellm.proxy._types import LitellmTableNames, ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.management_endpoints.password_endpoints import change_password
from litellm.proxy.utils import hash_password, verify_password

CURRENT_PASSWORD = "OldP@ssw0rd-2026"
NEW_PASSWORD = "NewP@ssw0rd-2026"

_POLICY_NO_BREACH_CHECK = {"password_policy_check_breached_passwords": False}


def _make_user_row(password: str | None) -> MagicMock:
    user = MagicMock()
    user.user_id = "user-123"
    user.password = password
    return user


def _make_prisma(user: MagicMock | None) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_usertable.find_first = AsyncMock(return_value=user)
    prisma.db.litellm_usertable.update = AsyncMock(return_value=user)
    return prisma


def _caller(user_id: str | None = "user-123") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id)


def _hibp_url_for(password: str) -> str:
    sha1 = hashlib.sha1(password.encode(), usedforsecurity=False).hexdigest().upper()
    return f"https://api.pwnedpasswords.com/range/{sha1[:5]}"


def _hibp_suffix_for(password: str) -> str:
    return hashlib.sha1(password.encode(), usedforsecurity=False).hexdigest().upper()[5:]


@pytest.mark.asyncio
async def test_change_password_success_writes_new_scrypt_hash():
    from litellm.proxy._types import ChangePasswordRequest

    prisma = _make_prisma(_make_user_row(hash_password(CURRENT_PASSWORD)))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        response = await change_password(
            data=ChangePasswordRequest(current_password=CURRENT_PASSWORD, new_password=NEW_PASSWORD),
            user_api_key_dict=_caller(),
        )

    assert response.user_id == "user-123"
    update_kwargs = prisma.db.litellm_usertable.update.call_args.kwargs
    assert update_kwargs["where"] == {"user_id": "user-123"}
    stored = update_kwargs["data"]["password"]
    assert stored != NEW_PASSWORD
    assert verify_password(NEW_PASSWORD, stored)


@pytest.mark.asyncio
async def test_change_password_rejects_wrong_current_password():
    from litellm.proxy._types import ChangePasswordRequest

    prisma = _make_prisma(_make_user_row(hash_password(CURRENT_PASSWORD)))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        with pytest.raises(HTTPException) as exc_info:
            await change_password(
                data=ChangePasswordRequest(current_password="not-the-password", new_password=NEW_PASSWORD),
                user_api_key_dict=_caller(),
            )

    assert exc_info.value.status_code == 400
    assert "Current password is incorrect" in exc_info.value.detail["error"]
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_change_password_rejects_session_without_user():
    from litellm.proxy._types import ChangePasswordRequest

    prisma = _make_prisma(user=None)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        with pytest.raises(HTTPException) as exc_info:
            await change_password(
                data=ChangePasswordRequest(current_password=CURRENT_PASSWORD, new_password=NEW_PASSWORD),
                user_api_key_dict=_caller(user_id=None),
            )

    assert exc_info.value.status_code == 400
    prisma.db.litellm_usertable.find_first.assert_not_called()
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_change_password_rejects_account_without_password():
    """SSO users and the env-credential admin have no DB password row to change."""
    from litellm.proxy._types import ChangePasswordRequest

    prisma = _make_prisma(_make_user_row(password=None))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        with pytest.raises(HTTPException) as exc_info:
            await change_password(
                data=ChangePasswordRequest(current_password=CURRENT_PASSWORD, new_password=NEW_PASSWORD),
                user_api_key_dict=_caller(),
            )

    assert exc_info.value.status_code == 400
    assert "no password set" in exc_info.value.detail["error"]
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_change_password_enforces_min_length():
    from litellm.proxy._types import ChangePasswordRequest

    prisma = _make_prisma(_make_user_row(hash_password(CURRENT_PASSWORD)))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        with pytest.raises(ProxyException) as exc_info:
            await change_password(
                data=ChangePasswordRequest(current_password=CURRENT_PASSWORD, new_password="Short1!"),
                user_api_key_dict=_caller(),
            )

    assert exc_info.value.code == "400"
    assert exc_info.value.type == ProxyErrorTypes.validation_error
    assert exc_info.value.param == "password"
    assert "at least 12 characters" in exc_info.value.message
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
@respx.mock
async def test_change_password_rejects_breached_password():
    """With the default policy, the new password is screened against HIBP."""
    from litellm.proxy._types import ChangePasswordRequest

    breached_password = "Password123!"
    respx.get(_hibp_url_for(breached_password)).mock(
        return_value=httpx.Response(200, text=f"{_hibp_suffix_for(breached_password)}:1")
    )
    prisma = _make_prisma(_make_user_row(hash_password(CURRENT_PASSWORD)))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", {}),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        with pytest.raises(ProxyException) as exc_info:
            await change_password(
                data=ChangePasswordRequest(current_password=CURRENT_PASSWORD, new_password=breached_password),
                user_api_key_dict=_caller(),
            )

    assert exc_info.value.code == "400"
    assert exc_info.value.type == ProxyErrorTypes.validation_error
    assert exc_info.value.param == "password"
    assert "data breaches" in exc_info.value.message
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
@respx.mock
async def test_change_password_verifies_current_password_before_hibp_lookup():
    """A caller who fails current-password verification must not trigger any
    HIBP traffic. The HIBP check fails open on errors, so an unmocked lookup
    could not prove ordering; instead the route is registered and asserted
    uncalled."""
    from litellm.proxy._types import ChangePasswordRequest

    hibp_route = respx.get(_hibp_url_for(NEW_PASSWORD)).mock(return_value=httpx.Response(200, text=""))
    prisma = _make_prisma(_make_user_row(hash_password(CURRENT_PASSWORD)))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", {}),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        with pytest.raises(HTTPException) as exc_info:
            await change_password(
                data=ChangePasswordRequest(current_password="not-the-password", new_password=NEW_PASSWORD),
                user_api_key_dict=_caller(),
            )

    assert exc_info.value.status_code == 400
    assert "Current password is incorrect" in exc_info.value.detail["error"]
    assert not hibp_route.called
    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_change_password_success_emits_redacted_audit_log():
    """A successful change must land in the audit trail as field names only;
    the plaintext passwords must never reach the audit call."""
    from litellm.proxy._types import ChangePasswordRequest

    prisma = _make_prisma(_make_user_row(hash_password(CURRENT_PASSWORD)))
    audit_mock = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.management_endpoints.password_endpoints.create_object_audit_log", audit_mock),
    ):
        await change_password(
            data=ChangePasswordRequest(current_password=CURRENT_PASSWORD, new_password=NEW_PASSWORD),
            user_api_key_dict=_caller(),
        )

    audit_mock.assert_awaited_once()
    audit_kwargs = audit_mock.await_args.kwargs
    assert audit_kwargs["object_id"] == "user-123"
    assert audit_kwargs["action"] == "updated"
    assert audit_kwargs["table_name"] == LitellmTableNames.USER_TABLE_NAME
    assert audit_kwargs["after_value"] == '{"fields_changed": ["password"]}'
    assert CURRENT_PASSWORD not in str(audit_kwargs)
    assert NEW_PASSWORD not in str(audit_kwargs)


@pytest.mark.asyncio
async def test_change_password_failure_emits_no_audit_log():
    from litellm.proxy._types import ChangePasswordRequest

    prisma = _make_prisma(_make_user_row(hash_password(CURRENT_PASSWORD)))
    audit_mock = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.management_endpoints.password_endpoints.create_object_audit_log", audit_mock),
    ):
        with pytest.raises(HTTPException):
            await change_password(
                data=ChangePasswordRequest(current_password="not-the-password", new_password=NEW_PASSWORD),
                user_api_key_dict=_caller(),
            )

    audit_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_change_password_requires_db():
    from litellm.proxy._types import ChangePasswordRequest

    with (
        patch("litellm.proxy.proxy_server.prisma_client", None),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
        patch("litellm.proxy.proxy_server.general_settings", _POLICY_NO_BREACH_CHECK),  # test-quality-ok: change_password reads proxy_server module globals; no injection seam
    ):
        with pytest.raises(HTTPException) as exc_info:
            await change_password(
                data=ChangePasswordRequest(current_password=CURRENT_PASSWORD, new_password=NEW_PASSWORD),
                user_api_key_dict=_caller(),
            )

    assert exc_info.value.status_code == 500
