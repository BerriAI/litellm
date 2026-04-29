import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm.proxy.proxy_server
from litellm.proxy._types import hash_token
from litellm.proxy.auth.reject_invalid_tokens import InvalidVirtualKeyCache
from litellm.proxy.management_endpoints.key_management_endpoints import (
    generate_key_helper_fn,
)


@pytest.mark.asyncio
async def test_generate_key_helper_clears_stale_invalid_token_cache(monkeypatch):
    raw_key = "sk-user-supplied-key-that-was-previously-invalid"
    hashed_key = hash_token(token=raw_key)
    mock_prisma_client = MagicMock()
    mock_prisma_client.insert_data = AsyncMock(
        return_value=SimpleNamespace(
            token=hashed_key,
            litellm_budget_table=None,
            created_at=None,
            updated_at=None,
        )
    )
    mock_cache = MagicMock()
    mock_cache.async_delete_cache = AsyncMock()

    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", mock_cache)
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", False)

    await generate_key_helper_fn(
        request_type="key",
        key=raw_key,
        table_name="key",
    )

    mock_cache.async_delete_cache.assert_awaited_once_with(
        key=InvalidVirtualKeyCache._cache_key(hashed_key)
    )
