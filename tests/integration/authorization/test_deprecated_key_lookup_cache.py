import os
from datetime import datetime, timedelta, timezone
from typing import Final
from uuid import uuid4

import pytest

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    _deprecated_key_cache,
    _lookup_deprecated_key,
)


@pytest.mark.asyncio
async def test_deprecated_key_grace_period_cache_hit_path() -> None:
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    old_token_hash: Final = f"old-{uuid4().hex}"
    active_token_hash: Final = f"active-{uuid4().hex}"
    _deprecated_key_cache.clear()

    await client.connect()
    try:
        await client.db.litellm_verificationtoken.create(
            data={
                "token": active_token_hash,
                "models": [],
            }
        )
        await client.db.litellm_deprecatedverificationtoken.create(
            data={
                "token": old_token_hash,
                "active_token_id": active_token_hash,
                "revoke_at": datetime.now(timezone.utc) + timedelta(minutes=5),
            }
        )

        first: Final = await _lookup_deprecated_key(db=client.db, hashed_token=old_token_hash)
        assert first == active_token_hash

        await client.db.litellm_deprecatedverificationtoken.delete_many(where={"token": old_token_hash})

        second: Final = await _lookup_deprecated_key(db=client.db, hashed_token=old_token_hash)
        third: Final = await _lookup_deprecated_key(db=client.db, hashed_token=old_token_hash)

        assert second == active_token_hash
        assert third == active_token_hash

        cached: Final = _deprecated_key_cache.get(old_token_hash)
        assert isinstance(cached, tuple)
        assert len(cached) == 3
    finally:
        await client.db.litellm_deprecatedverificationtoken.delete_many(where={"token": old_token_hash})
        await client.db.litellm_verificationtoken.delete_many(where={"token": active_token_hash})
        _deprecated_key_cache.clear()
        await client.disconnect()
