from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import _deprecated_key_cache, _lookup_deprecated_key


@pytest.mark.asyncio
async def test_lookup_deprecated_key_handles_cached_entry():
    """
    The first lookup should populate the deprecated-key cache; the second lookup
    should read that cached entry without raising or hitting the database again.
    """
    hashed_token = "old-token-hash"
    active_token_id = "active-token-hash"
    revoke_at = datetime.now(timezone.utc) + timedelta(minutes=5)

    deprecated_row = MagicMock()
    deprecated_row.active_token_id = active_token_id
    deprecated_row.revoke_at = revoke_at

    db = MagicMock()
    db.litellm_deprecatedverificationtoken.find_first = AsyncMock(
        return_value=deprecated_row
    )

    _deprecated_key_cache.clear()
    try:
        assert await _lookup_deprecated_key(db=db, hashed_token=hashed_token) == active_token_id
        assert await _lookup_deprecated_key(db=db, hashed_token=hashed_token) == active_token_id
    finally:
        _deprecated_key_cache.clear()

    db.litellm_deprecatedverificationtoken.find_first.assert_awaited_once()
