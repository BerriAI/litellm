from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.repositories.password_reset_token_repository import (
    PasswordResetTokenRepository,
)


def _mock_prisma():
    prisma_client = MagicMock()
    prisma_client.db.litellm_passwordresettoken = MagicMock()
    return prisma_client


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_none_when_not_found():
    prisma_client = _mock_prisma()
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=None)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=datetime.now(timezone.utc))

    assert result is None


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_none_when_expired():
    prisma_client = _mock_prisma()
    expired_row = MagicMock()
    expired_row.dict.return_value = {
        "token_hash": "abc",
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
        "expires_at": datetime.now(timezone.utc) - timedelta(minutes=1),
        "used_at": None,
    }
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=expired_row)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=datetime.now(timezone.utc))

    assert result is None


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_none_when_already_used():
    prisma_client = _mock_prisma()
    used_row = MagicMock()
    now = datetime.now(timezone.utc)
    used_row.dict.return_value = {
        "token_hash": "abc",
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now - timedelta(minutes=5),
        "expires_at": now + timedelta(minutes=25),
        "used_at": now - timedelta(minutes=1),
    }
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=used_row)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=now)

    assert result is None


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_model_when_valid():
    prisma_client = _mock_prisma()
    now = datetime.now(timezone.utc)
    valid_row = MagicMock()
    valid_row.dict.return_value = {
        "token_hash": "abc",
        "user_id": "user-1",
        "requested_ip": "127.0.0.1",
        "created_at": now - timedelta(minutes=5),
        "expires_at": now + timedelta(minutes=25),
        "used_at": None,
    }
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=valid_row)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=now)

    assert result is not None
    assert result.user_id == "user-1"


@pytest.mark.asyncio
async def test_invalidate_unused_for_user_calls_update_many():
    prisma_client = _mock_prisma()
    prisma_client.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=2)
    now = datetime.now(timezone.utc)

    repo = PasswordResetTokenRepository(prisma_client)
    await repo.invalidate_unused_for_user(user_id="user-1", now=now)

    prisma_client.db.litellm_passwordresettoken.update_many.assert_awaited_once_with(
        where={"user_id": "user-1", "used_at": None},
        data={"used_at": now},
    )
