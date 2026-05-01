from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import get_key_object


@pytest.mark.asyncio
async def test_get_key_object_returns_copy_of_cached_user_api_key_auth():
    """
    In-memory cache returns object references. Auth callers must receive a copy so
    request-scoped mutations do not poison the cached key object.
    """
    cached_token = UserAPIKeyAuth(
        token="hashed-token",
        end_user_id=None,
        end_user_tpm_limit=None,
    )
    user_api_key_cache = MagicMock()
    user_api_key_cache.async_get_cache = AsyncMock(return_value=cached_token)

    result = await get_key_object(
        hashed_token="hashed-token",
        prisma_client=MagicMock(),
        user_api_key_cache=user_api_key_cache,
    )

    assert result is not cached_token
    result.end_user_id = "request-user"
    result.end_user_tpm_limit = 1
    assert cached_token.end_user_id is None
    assert cached_token.end_user_tpm_limit is None
