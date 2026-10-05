"""
Tests for the grace-period key-rotation feature (MLI-6358).

Two bugs are confirmed in LiteLLM v1.83.7-stable (upstream BerriAI/litellm#27193).
Both live in _lookup_deprecated_key() (litellm/proxy/utils.py):

  Bug 1 — duplicate cache read (cosmetic, no functional impact on its own):
      The cache is fetched twice in a row with no state change between the calls.

  Bug 2 — cache stores a 2-tuple but unpacks as a 3-tuple:
      WRITE:  _deprecated_key_cache[hash] = (active_token_id, cache_expires_at_ts)
      READ:   active_token_id, cache_expires_at_ts, revoke_at_ts = cached   # ValueError!
      The ValueError is NOT inside the try/except, so it propagates up through
      PrismaClient.get_data() (which re-raises), killing the auth request.

The local demo script confirmed
that all three requests with the old key returned HTTP 401 immediately after
rotation even though the grace-period window was still open.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

HASHED_TOKEN = "165efe575c98fe7e65d98cb2de71b68842049e286afd33a92d3491c340216880"
ACTIVE_TOKEN_HASH = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab"


def _make_db(active_token_id: Optional[str]) -> MagicMock:
    """Prisma db mock whose deprecated-token find_first returns the given id."""
    row = MagicMock()
    row.active_token_id = active_token_id
    row.revoke_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db = MagicMock()
    db.litellm_deprecatedverificationtoken = MagicMock()
    db.litellm_deprecatedverificationtoken.find_first = AsyncMock(
        return_value=row if active_token_id else None
    )
    return db


# ── Bug 1: first call (DB path) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lookup_deprecated_key_db_miss_returns_none():
    """Token absent from deprecated table → returns None without error."""
    from litellm.proxy.utils import _lookup_deprecated_key, _deprecated_key_cache

    _deprecated_key_cache.clear()
    db = _make_db(active_token_id=None)

    result = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)

    assert result is None
    db.litellm_deprecatedverificationtoken.find_first.assert_called_once()


@pytest.mark.asyncio
async def test_lookup_deprecated_key_db_hit_returns_active_token_id():
    """
    First call (cold cache): DB row exists within grace window → returns
    active_token_id correctly.  The DB path itself works; the bug is on the
    second call when the result is read back from cache.
    """
    from litellm.proxy.utils import _lookup_deprecated_key, _deprecated_key_cache

    _deprecated_key_cache.clear()
    db = _make_db(active_token_id=ACTIVE_TOKEN_HASH)

    result = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)

    assert result == ACTIVE_TOKEN_HASH
    db.litellm_deprecatedverificationtoken.find_first.assert_called_once()


# ── Bug 2: second call (cache path) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_lookup_deprecated_key_cache_hit_returns_on_second_call():
    """
    Regression guard: after first call warms the cache with a 3-tuple,
    second call should return from cache without raising.
    """
    from litellm.proxy.utils import _lookup_deprecated_key, _deprecated_key_cache

    _deprecated_key_cache.clear()
    db = _make_db(active_token_id=ACTIVE_TOKEN_HASH)

    # First call: cold cache → DB hit → warms cache with 3-tuple → succeeds
    r1 = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)
    assert r1 == ACTIVE_TOKEN_HASH, "First call (DB path) must succeed"

    # Second call: cache hit path should succeed without DB access
    r2 = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)
    assert r2 == ACTIVE_TOKEN_HASH

    # DB is queried exactly once; the second call never reaches it
    assert db.litellm_deprecatedverificationtoken.find_first.call_count == 1


@pytest.mark.asyncio
async def test_lookup_deprecated_key_pre_warmed_cache_returns():
    """
    Pre-warmed 3-tuple cache entry should be served directly from cache.
    """
    from litellm.proxy.utils import _lookup_deprecated_key, _deprecated_key_cache

    _deprecated_key_cache.clear()
    now_ts = datetime.now(timezone.utc).timestamp()
    _deprecated_key_cache[HASHED_TOKEN] = (
        ACTIVE_TOKEN_HASH,
        now_ts + 60,
        now_ts + 300,
    )

    db = _make_db(active_token_id=ACTIVE_TOKEN_HASH)

    result = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)
    assert result == ACTIVE_TOKEN_HASH

    db.litellm_deprecatedverificationtoken.find_first.assert_not_called()


# ── End-to-end reproduction of the demo ──────────────────────────────────────


@pytest.mark.asyncio
async def test_grace_period_three_requests_mirrors_demo():
    """
    Reproduces Step 5 of the local demo script:

      Request 1 (cache miss  — DB lookup)  → succeeds
      Request 2 (cache hit)                → succeeds
      Request 3 (cache hit)                → succeeds
    """
    from litellm.proxy.utils import _lookup_deprecated_key, _deprecated_key_cache

    _deprecated_key_cache.clear()
    db = _make_db(active_token_id=ACTIVE_TOKEN_HASH)

    r1 = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)
    assert r1 == ACTIVE_TOKEN_HASH, "Request 1 (DB path) should succeed"

    r2 = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)
    r3 = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)
    assert r2 == ACTIVE_TOKEN_HASH
    assert r3 == ACTIVE_TOKEN_HASH

    # DB hit only once; requests 2 and 3 never reach it
    assert db.litellm_deprecatedverificationtoken.find_first.call_count == 1


@pytest.mark.asyncio
async def test_cache_hit_respects_revoke_at_timestamp():
    """Cache entries should not remain valid past revoke_at even if cache TTL is still live."""
    from litellm.proxy.utils import _lookup_deprecated_key, _deprecated_key_cache

    _deprecated_key_cache.clear()
    now_ts = datetime.now(timezone.utc).timestamp()
    # cache_expires_at is in the future, but revoke_at is already past.
    _deprecated_key_cache[HASHED_TOKEN] = (
        ACTIVE_TOKEN_HASH,
        now_ts + 60,
        now_ts - 1,
    )

    db = _make_db(active_token_id=None)
    result = await _lookup_deprecated_key(db=db, hashed_token=HASHED_TOKEN)
    assert result is None
    db.litellm_deprecatedverificationtoken.find_first.assert_called_once()
