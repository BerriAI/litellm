"""
Unit tests for secure share endpoints
"""

import base64
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.secure_share.secure_share_endpoints import (
    SecureShareCreateRequest,
    SecureShareExpiry,
    create_secure_share,
    delete_secure_share,
    get_secure_share,
)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _valid_request(expiry: SecureShareExpiry = SecureShareExpiry.ONE_HOUR) -> SecureShareCreateRequest:
    return SecureShareCreateRequest(
        ciphertext=_b64(b"encrypted-secret-bytes"),
        salt=_b64(b"0123456789abcdef"),
        iv=_b64(b"0123456789ab"),
        expiry=expiry,
    )


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user")


def _fake_table() -> MagicMock:
    table = MagicMock()
    table.create = AsyncMock()
    table.find_unique = AsyncMock()
    table.delete = AsyncMock()
    return table


def _patched_prisma(table: MagicMock):
    prisma_client = MagicMock()
    prisma_client.db.litellm_securesharetable = table
    return patch.multiple(
        "litellm.proxy.proxy_server",
        prisma_client=prisma_client,
        litellm_proxy_admin_name="default_admin",
    )


@pytest.mark.parametrize(
    "expiry, expected",
    [
        (SecureShareExpiry.ONE_HOUR, timedelta(hours=1)),
        (SecureShareExpiry.SIX_HOURS, timedelta(hours=6)),
        (SecureShareExpiry.ONE_DAY, timedelta(days=1)),
        (SecureShareExpiry.SEVEN_DAYS, timedelta(days=7)),
    ],
)
def test_expiry_durations(expiry: SecureShareExpiry, expected: timedelta):
    assert expiry.duration == expected


def test_create_request_rejects_non_base64():
    with pytest.raises(ValidationError):
        SecureShareCreateRequest(ciphertext="not base64!!", salt=_b64(b"salt"), iv=_b64(b"iv"), expiry="1h")


def test_create_request_rejects_oversized_ciphertext():
    with pytest.raises(ValidationError):
        SecureShareCreateRequest(
            ciphertext=_b64(b"x" * (256 * 1024 + 1)),
            salt=_b64(b"salt"),
            iv=_b64(b"iv"),
            expiry="1h",
        )


def test_create_request_rejects_unknown_expiry():
    with pytest.raises(ValidationError):
        SecureShareCreateRequest(ciphertext=_b64(b"c"), salt=_b64(b"s"), iv=_b64(b"i"), expiry="30d")


def test_create_request_rejects_empty_ciphertext():
    with pytest.raises(ValidationError):
        SecureShareCreateRequest(ciphertext="", salt=_b64(b"s"), iv=_b64(b"i"), expiry="1h")


@pytest.mark.asyncio
async def test_create_returns_500_when_db_not_connected():
    with patch.multiple("litellm.proxy.proxy_server", prisma_client=None, litellm_proxy_admin_name="default_admin"):
        with pytest.raises(HTTPException) as exc:
            await create_secure_share(request=_valid_request(), user_api_key_dict=_admin())

    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_create_stores_ciphertext_and_computes_expiry():
    request = _valid_request(SecureShareExpiry.ONE_DAY)
    expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    table = _fake_table()
    table.create.return_value = SimpleNamespace(share_id="share-123", expires_at=expires_at)

    before = datetime.now(timezone.utc)
    with _patched_prisma(table):
        result = await create_secure_share(request=request, user_api_key_dict=_admin())
    after = datetime.now(timezone.utc)

    assert result.share_id == "share-123"
    stored = table.create.call_args.kwargs["data"]
    assert stored["ciphertext"] == request.ciphertext
    assert stored["salt"] == request.salt
    assert stored["iv"] == request.iv
    assert stored["created_by"] == "admin-user"
    assert before + timedelta(days=1) <= stored["expires_at"] <= after + timedelta(days=1)


@pytest.mark.asyncio
async def test_create_falls_back_to_admin_name_when_no_user_id():
    table = _fake_table()
    table.create.return_value = SimpleNamespace(
        share_id="s", expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id=None)

    with _patched_prisma(table):
        await create_secure_share(request=_valid_request(), user_api_key_dict=caller)

    assert table.create.call_args.kwargs["data"]["created_by"] == "default_admin"


@pytest.mark.asyncio
async def test_create_forbidden_for_non_admin():
    table = _fake_table()
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-user", user_id="u1")

    with _patched_prisma(table):
        with pytest.raises(HTTPException) as exc:
            await create_secure_share(request=_valid_request(), user_api_key_dict=caller)

    assert exc.value.status_code == 403
    table.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    ],
)
async def test_get_returns_unexpired_share_for_allowed_roles(role: LitellmUserRoles):
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    table = _fake_table()
    table.find_unique.return_value = SimpleNamespace(
        share_id="share-1",
        ciphertext=_b64(b"c"),
        salt=_b64(b"s"),
        iv=_b64(b"i"),
        expires_at=expires_at,
        created_by="admin-user",
    )
    caller = UserAPIKeyAuth(user_role=role, api_key="sk", user_id="u")

    with _patched_prisma(table):
        result = await get_secure_share(share_id="share-1", user_api_key_dict=caller)

    assert result.share_id == "share-1"
    assert result.ciphertext == _b64(b"c")
    table.delete.assert_awaited_once_with(where={"share_id": "share-1"})


@pytest.mark.asyncio
async def test_get_is_one_time_second_read_returns_404():
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    table = _fake_table()
    table.find_unique.side_effect = [
        SimpleNamespace(
            share_id="share-1",
            ciphertext=_b64(b"c"),
            salt=_b64(b"s"),
            iv=_b64(b"i"),
            expires_at=expires_at,
            created_by="admin-user",
        ),
        None,
    ]

    with _patched_prisma(table):
        first = await get_secure_share(share_id="share-1", user_api_key_dict=_admin())
        with pytest.raises(HTTPException) as exc:
            await get_secure_share(share_id="share-1", user_api_key_dict=_admin())

    assert first.share_id == "share-1"
    table.delete.assert_awaited_once_with(where={"share_id": "share-1"})
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_forbidden_for_customer_role():
    table = _fake_table()
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.CUSTOMER, api_key="sk", user_id="u")

    with _patched_prisma(table):
        with pytest.raises(HTTPException) as exc:
            await get_secure_share(share_id="share-1", user_api_key_dict=caller)

    assert exc.value.status_code == 403
    table.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_get_missing_share_returns_404():
    table = _fake_table()
    table.find_unique.return_value = None

    with _patched_prisma(table):
        with pytest.raises(HTTPException) as exc:
            await get_secure_share(share_id="missing", user_api_key_dict=_admin())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_expired_share_is_deleted_and_returns_410():
    expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    table = _fake_table()
    table.find_unique.return_value = SimpleNamespace(
        share_id="share-old",
        ciphertext=_b64(b"c"),
        salt=_b64(b"s"),
        iv=_b64(b"i"),
        expires_at=expires_at,
        created_by="admin-user",
    )

    with _patched_prisma(table):
        with pytest.raises(HTTPException) as exc:
            await get_secure_share(share_id="share-old", user_api_key_dict=_admin())

    assert exc.value.status_code == 410
    table.delete.assert_awaited_once_with(where={"share_id": "share-old"})


@pytest.mark.asyncio
async def test_get_treats_naive_expiry_as_utc():
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    table = _fake_table()
    table.find_unique.return_value = SimpleNamespace(
        share_id="share-naive",
        ciphertext=_b64(b"c"),
        salt=_b64(b"s"),
        iv=_b64(b"i"),
        expires_at=expires_at,
        created_by="admin-user",
    )

    with _patched_prisma(table):
        with pytest.raises(HTTPException) as exc:
            await get_secure_share(share_id="share-naive", user_api_key_dict=_admin())

    assert exc.value.status_code == 410


@pytest.mark.asyncio
async def test_delete_forbidden_for_non_admin():
    table = _fake_table()
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk", user_id="u")

    with _patched_prisma(table):
        with pytest.raises(HTTPException) as exc:
            await delete_secure_share(share_id="share-1", user_api_key_dict=caller)

    assert exc.value.status_code == 403
    table.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_removes_share_for_admin():
    table = _fake_table()
    table.delete.return_value = SimpleNamespace(share_id="share-1")

    with _patched_prisma(table):
        result = await delete_secure_share(share_id="share-1", user_api_key_dict=_admin())

    assert result == {"share_id": "share-1", "status": "deleted"}
    table.delete.assert_awaited_once_with(where={"share_id": "share-1"})


@pytest.mark.asyncio
async def test_delete_missing_share_returns_404():
    table = _fake_table()
    table.delete.return_value = None

    with _patched_prisma(table):
        with pytest.raises(HTTPException) as exc:
            await delete_secure_share(share_id="missing", user_api_key_dict=_admin())

    assert exc.value.status_code == 404
