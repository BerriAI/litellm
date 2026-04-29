# Unit tests for litellm.proxy.auth.reject_invalid_tokens.InvalidVirtualKeyCache

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import hash_token
from litellm.proxy.auth.reject_invalid_tokens import InvalidVirtualKeyCache


def _sk_key() -> str:
    return "sk-test-invalid-virtual-key"

def _general_settings_positive_ttl() -> dict:
    """Force negative-cache path on (avoid relying only on default constant)."""
    return {"invalid_virtual_key_cache_ttl": 3600}


def _negative_cache_key_for(api_key: str) -> str:
    return InvalidVirtualKeyCache._cache_key(hash_token(token=api_key))


@pytest.mark.asyncio
async def test_check_invalid_token_empty_cache_db_miss_records_negative_entry():
    """
    Negative cache empty, Prisma finds no verification row → reject (True);
    miss is recorded for repeat traffic.
    """
    api_key = _sk_key()
    hashed = hash_token(token=api_key)
    neg_key = _negative_cache_key_for(api_key)

    user_api_key_cache = MagicMock()
    user_api_key_cache.async_get_cache = AsyncMock(return_value=None)
    user_api_key_cache.async_set_cache = AsyncMock()

    find_first = AsyncMock(return_value=None)
    prisma_client = MagicMock()
    prisma_client.db.litellm_verificationtoken.find_first = find_first

    result = await InvalidVirtualKeyCache.check_invalid_token(
        api_key=api_key,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        general_settings=_general_settings_positive_ttl(),
    )

    assert result is True
    find_first.assert_awaited_once_with(where={"token": hashed})
    user_api_key_cache.async_get_cache.assert_awaited_once_with(key=neg_key)
    user_api_key_cache.async_set_cache.assert_awaited_once_with(
        key=neg_key,
        value="",
        ttl=3600.0,
    )


@pytest.mark.asyncio
async def test_check_invalid_token_negative_cache_hit_short_circuits_even_if_db_has_row():
    """
    Hash already negative-cached → reject immediately without calling Prisma,
    even if a row exists in DB (stale negative cache after key creation is possible).
    """
    api_key = _sk_key()
    neg_key = _negative_cache_key_for(api_key)

    user_api_key_cache = MagicMock()
    user_api_key_cache.async_get_cache = AsyncMock(return_value="")
    user_api_key_cache.async_set_cache = AsyncMock()

    find_first = AsyncMock(return_value=MagicMock(token=hash_token(token=api_key)))
    prisma_client = MagicMock()
    prisma_client.db.litellm_verificationtoken.find_first = find_first

    result = await InvalidVirtualKeyCache.check_invalid_token(
        api_key=api_key,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        general_settings=_general_settings_positive_ttl(),
    )

    assert result is True
    find_first.assert_not_called()
    user_api_key_cache.async_set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_check_invalid_token_cache_miss_db_hit_allows_auth_flow():
    """Negative cache empty, Prisma finds a verification row → preflight passes (False)."""
    api_key = _sk_key()
    hashed = hash_token(token=api_key)
    neg_key = _negative_cache_key_for(api_key)

    user_api_key_cache = MagicMock()
    user_api_key_cache.async_get_cache = AsyncMock(return_value=None)
    user_api_key_cache.async_set_cache = AsyncMock()

    find_first = AsyncMock(return_value=MagicMock(token=hashed))
    prisma_client = MagicMock()
    prisma_client.db.litellm_verificationtoken.find_first = find_first

    result = await InvalidVirtualKeyCache.check_invalid_token(
        api_key=api_key,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        general_settings=_general_settings_positive_ttl(),
    )

    assert result is False
    find_first.assert_awaited_once_with(where={"token": hashed})
    user_api_key_cache.async_get_cache.assert_awaited_once_with(key=neg_key)
    user_api_key_cache.async_set_cache.assert_not_called()
