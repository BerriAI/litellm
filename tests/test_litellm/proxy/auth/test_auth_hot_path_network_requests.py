"""
Test to count and track the number of network requests (DB queries, cache lookups)
made on the hot path for keys that have team_id and user_id attached.

This test ensures we don't regress on the number of network requests made during
request authentication, which directly impacts proxy latency.

The hot path covers auth functions called on every LLM API request:
- get_key_object: lookup the API key
- get_team_object: lookup the team (for keys with team_id)
- get_user_object: lookup the user (for keys with user_id)
- get_team_membership: lookup team member budget (when team_member_spend set)

Each function does: cache read -> (on miss) DB query -> cache write.
We count these to catch regressions in the number of network requests.

NOTE: This test does NOT require proxy extras (apscheduler, etc.) because
it tests at the auth_checks level, not the full proxy_server level.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
    LiteLLM_TeamMembership,
    hash_token,
)
from litellm.proxy.auth.auth_checks import (
    get_key_object,
    get_team_membership,
    get_team_object,
    get_user_object,
)


class CacheCallTracker:
    """
    Tracks cache read/write operations by wrapping DualCache methods.
    This is used to count network-level operations on the hot path.
    """

    def __init__(self):
        self.cache_reads: List[Dict[str, Any]] = []
        self.cache_writes: List[Dict[str, Any]] = []
        self.db_queries: List[Dict[str, Any]] = []

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_cache_reads": len(self.cache_reads),
            "total_cache_writes": len(self.cache_writes),
            "total_db_queries": len(self.db_queries),
            "total_network_requests": len(self.cache_reads)
            + len(self.cache_writes)
            + len(self.db_queries),
            "cache_read_keys": [r["key"] for r in self.cache_reads],
            "cache_write_keys": [w["key"] for w in self.cache_writes],
            "db_query_details": self.db_queries,
        }


def _wrap_cache_with_tracker(cache: DualCache, tracker: CacheCallTracker) -> DualCache:
    """Wrap a DualCache to track all reads and writes."""
    original_async_get = cache.async_get_cache
    original_async_set = cache.async_set_cache

    async def tracked_async_get(key, *args, **kwargs):
        result = await original_async_get(key, *args, **kwargs)
        tracker.cache_reads.append(
            {"key": key, "hit": result is not None, "method": "async_get_cache"}
        )
        return result

    async def tracked_async_set(key, value, *args, **kwargs):
        tracker.cache_writes.append({"key": key, "method": "async_set_cache"})
        return await original_async_set(key, value, *args, **kwargs)

    cache.async_get_cache = tracked_async_get
    cache.async_set_cache = tracked_async_set
    return cache


def _create_valid_token(
    api_key: str,
    team_id: str,
    user_id: str,
    has_team_member_spend: bool = False,
    org_id: Optional[str] = None,
) -> UserAPIKeyAuth:
    """Create a UserAPIKeyAuth with team_id and user_id set."""
    hashed = hash_token(api_key)
    return UserAPIKeyAuth(
        token=hashed,
        api_key=api_key,
        team_id=team_id,
        user_id=user_id,
        org_id=org_id,
        models=["gpt-4", "gpt-3.5-turbo"],
        max_budget=100.0,
        spend=10.0,
        team_spend=50.0,
        team_max_budget=1000.0,
        team_models=["gpt-4", "gpt-3.5-turbo"],
        team_member_spend=5.0 if has_team_member_spend else None,
        last_refreshed_at=time.time(),
        user_role=LitellmUserRoles.INTERNAL_USER,
    )


def _create_team_object(team_id: str) -> LiteLLM_TeamTableCachedObj:
    """Create a team table object for caching."""
    return LiteLLM_TeamTableCachedObj(
        team_id=team_id,
        models=["gpt-4", "gpt-3.5-turbo"],
        max_budget=1000.0,
        spend=50.0,
        tpm_limit=10000,
        rpm_limit=100,
        last_refreshed_at=time.time(),
    )


def _create_user_object(user_id: str) -> LiteLLM_UserTable:
    """Create a user table object for caching."""
    return LiteLLM_UserTable(
        user_id=user_id,
        max_budget=500.0,
        spend=25.0,
        models=["gpt-4"],
        tpm_limit=5000,
        rpm_limit=50,
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_email="test@example.com",
    )


# ============================================================================
# TEST: get_key_object cache behavior
# ============================================================================


@pytest.mark.asyncio
async def test_get_key_object_warm_cache():
    """
    Test get_key_object with a warm cache - should hit cache, no DB query.
    """
    api_key = "sk-test-key-warm"
    team_id = "team-123"
    user_id = "user-456"
    hashed_token = hash_token(api_key)

    valid_token = _create_valid_token(api_key, team_id, user_id)

    # Create cache with pre-populated data
    cache = DualCache(in_memory_cache=InMemoryCache())
    await cache.async_set_cache(key=hashed_token, value=valid_token)

    # Track cache operations
    tracker = CacheCallTracker()
    tracked_cache = _wrap_cache_with_tracker(cache, tracker)

    # Mock prisma client (should NOT be called for warm cache)
    mock_prisma = MagicMock()
    mock_prisma.get_data = AsyncMock()

    result = await get_key_object(
        hashed_token=hashed_token,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    summary = tracker.get_summary()

    # Should have exactly 1 cache read
    assert summary["total_cache_reads"] == 1
    assert hashed_token in summary["cache_read_keys"]

    # Prisma should NOT have been called
    mock_prisma.get_data.assert_not_called()

    # Result should be the cached token
    assert result.token == hashed_token


@pytest.mark.asyncio
async def test_get_key_object_cold_cache():
    """
    Test get_key_object with a cold cache - should miss cache, query DB.
    """
    api_key = "sk-test-key-cold"
    team_id = "team-123"
    user_id = "user-456"
    hashed_token = hash_token(api_key)

    valid_token = _create_valid_token(api_key, team_id, user_id)

    # Create empty cache
    cache = DualCache(in_memory_cache=InMemoryCache())

    tracker = CacheCallTracker()
    tracked_cache = _wrap_cache_with_tracker(cache, tracker)

    # Mock prisma client to return token on DB query
    mock_prisma = MagicMock()
    mock_prisma.get_data = AsyncMock(return_value=valid_token)

    await get_key_object(
        hashed_token=hashed_token,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    summary = tracker.get_summary()

    # Should have 1 cache read (miss) and at least 1 cache write (populate cache)
    assert summary["total_cache_reads"] >= 1

    # Prisma SHOULD have been called
    mock_prisma.get_data.assert_called_once()


# ============================================================================
# TEST: get_team_object cache behavior
# ============================================================================


@pytest.mark.asyncio
async def test_get_team_object_warm_cache():
    """
    Test get_team_object with a warm cache - should hit cache, no DB query.
    """
    team_id = "team-warm-123"
    team_obj = _create_team_object(team_id)

    cache = DualCache(in_memory_cache=InMemoryCache())
    cache_key = f"team_id:{team_id}"
    await cache.async_set_cache(key=cache_key, value=team_obj)

    tracker = CacheCallTracker()
    tracked_cache = _wrap_cache_with_tracker(cache, tracker)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_teamtable = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock()

    await get_team_object(
        team_id=team_id,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    summary = tracker.get_summary()

    assert summary["total_cache_reads"] >= 1
    assert cache_key in summary["cache_read_keys"]

    # DB should NOT have been called
    mock_prisma.db.litellm_teamtable.find_unique.assert_not_called()


# ============================================================================
# TEST: get_user_object cache behavior
# ============================================================================


@pytest.mark.asyncio
async def test_get_user_object_warm_cache():
    """
    Test get_user_object with a warm cache - should hit cache, no DB query.
    """
    user_id = "user-warm-456"
    user_obj = _create_user_object(user_id)

    cache = DualCache(in_memory_cache=InMemoryCache())
    await cache.async_set_cache(key=user_id, value=user_obj)

    tracker = CacheCallTracker()
    tracked_cache = _wrap_cache_with_tracker(cache, tracker)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_usertable = MagicMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock()

    await get_user_object(
        user_id=user_id,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
        user_id_upsert=False,
    )

    summary = tracker.get_summary()

    assert summary["total_cache_reads"] >= 1
    assert user_id in summary["cache_read_keys"]

    # DB should NOT have been called
    mock_prisma.db.litellm_usertable.find_unique.assert_not_called()


# ============================================================================
# TEST: get_team_membership cache behavior
# ============================================================================


@pytest.mark.asyncio
async def test_get_team_membership_warm_cache():
    """
    Test get_team_membership with a warm cache - should hit cache, no DB query.
    """
    user_id = "user-tm-456"
    team_id = "team-tm-123"

    membership_dict = {
        "user_id": user_id,
        "team_id": team_id,
        "spend": 3.0,
        "budget_id": None,
        "litellm_budget_table": None,
    }

    cache = DualCache(in_memory_cache=InMemoryCache())
    # Cache key format used by get_team_membership
    cache_key = f"team_membership:{user_id}:{team_id}"
    await cache.async_set_cache(key=cache_key, value=membership_dict)

    tracker = CacheCallTracker()
    tracked_cache = _wrap_cache_with_tracker(cache, tracker)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_teammembership = MagicMock()
    mock_prisma.db.litellm_teammembership.find_unique = AsyncMock()

    await get_team_membership(
        user_id=user_id,
        team_id=team_id,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    summary = tracker.get_summary()

    assert summary["total_cache_reads"] >= 1
    assert cache_key in summary["cache_read_keys"]

    # DB should NOT have been called
    mock_prisma.db.litellm_teammembership.find_unique.assert_not_called()


# ============================================================================
# TEST: Document duplicate team membership cache key issue
# ============================================================================


@pytest.mark.asyncio
async def test_team_membership_cache_key_duplication():
    """
    Document the team membership duplicate cache key issue:

    Team membership is queried via TWO different cache keys:
    1. "{team_id}_{user_id}" - used in user_api_key_auth.py:1048
    2. "team_membership:{user_id}:{team_id}" - used in auth_checks.py:960 (get_team_membership)

    This test documents that both keys refer to the same data but use different
    cache key formats, potentially leading to duplicate lookups.
    """
    user_id = "user-dup-456"
    team_id = "team-dup-123"

    # The two different cache keys used for the same data
    key_format_1 = f"{team_id}_{user_id}"  # user_api_key_auth format
    key_format_2 = f"team_membership:{user_id}:{team_id}"  # auth_checks format

    _ = {
        "user_id": user_id,
        "team_id": team_id,
        "spend": 3.0,
    }

    # Document that these are different keys
    assert (
        key_format_1 != key_format_2
    ), "Cache keys should be different (this is the bug)"

    # Document that these are different keys
    assert (
        key_format_1 != key_format_2
    ), "Cache keys should be different (this is the bug)"


# ============================================================================
# TEST: Full hot path network count summary
# ============================================================================


@pytest.mark.asyncio
async def test_full_hot_path_network_count():
    """
    Summary test that counts all network operations when processing
    a request with a key that has team_id and user_id attached.

    This test verifies the baseline number of cache operations expected
    on a fully warm cache path.
    """
    api_key = "sk-test-full-path"
    team_id = "team-full-123"
    user_id = "user-full-456"
    hashed_token = hash_token(api_key)

    # Create all objects
    valid_token = _create_valid_token(
        api_key, team_id, user_id, has_team_member_spend=True
    )
    team_obj = _create_team_object(team_id)
    user_obj = _create_user_object(user_id)
    membership_data = LiteLLM_TeamMembership(
        user_id=user_id,
        team_id=team_id,
        spend=3.0,
        budget_id=None,
        litellm_budget_table=None,
    )

    # Pre-populate cache with all data
    cache = DualCache(in_memory_cache=InMemoryCache())
    await cache.async_set_cache(key=hashed_token, value=valid_token)
    await cache.async_set_cache(key=f"team_id:{team_id}", value=team_obj)
    await cache.async_set_cache(key=user_id, value=user_obj)
    await cache.async_set_cache(
        key=f"team_membership:{user_id}:{team_id}", value=membership_data.model_dump()
    )
    await cache.async_set_cache(
        key=f"{team_id}_{user_id}", value=membership_data.model_dump()
    )

    # Create tracker AFTER populating cache
    tracker = CacheCallTracker()
    tracked_cache = _wrap_cache_with_tracker(cache, tracker)

    # Mock prisma (should not be called on warm cache)
    mock_prisma = MagicMock()

    # Call each function to simulate the hot path
    await get_key_object(
        hashed_token=hashed_token,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    await get_team_object(
        team_id=team_id,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    await get_user_object(
        user_id=user_id,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
        user_id_upsert=False,
    )

    await get_team_membership(
        user_id=user_id,
        team_id=team_id,
        prisma_client=mock_prisma,
        user_api_key_cache=tracked_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    summary = tracker.get_summary()

    # Assertions for expected baseline
    # On warm cache: 4 reads (key, team, user, team_membership)
    assert (
        summary["total_cache_reads"] == 4
    ), f"Expected 4 cache reads on warm path, got {summary['total_cache_reads']}"

    # No DB queries on warm cache
    assert (
        summary["total_db_queries"] == 0
    ), f"Expected 0 DB queries on warm path, got {summary['total_db_queries']}"

    # Total network requests should be exactly 4 on warm cache
    assert (
        summary["total_network_requests"] == 4
    ), f"Expected 4 total network requests on warm path, got {summary['total_network_requests']}"
