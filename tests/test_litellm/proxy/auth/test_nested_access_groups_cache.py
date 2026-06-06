"""
Cache-behavior tests for the nested-access-group membership map (#28032).

Hot-path callers go through get_cached_group_memberships() which TTL-caches
get_group_memberships_from_db() and is invalidated by every membership
write. These tests pin the cache hit/miss/invalidation semantics so the
optimization can't silently break later.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

import litellm.proxy.management_endpoints.model_access_group_management_endpoints as mgmt
from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
    delete_group_membership_edges,
    get_cached_group_memberships,
    invalidate_group_memberships_cache,
    upsert_group_memberships,
)


def _row(parent: str, child: str) -> SimpleNamespace:
    return SimpleNamespace(parent_group=parent, child_group=child)


def _make_prisma(membership_rows=None):
    membership_rows = membership_rows or []
    db = MagicMock()
    db.litellm_accessgroupmembership = MagicMock()
    db.litellm_accessgroupmembership.find_many = AsyncMock(return_value=membership_rows)
    db.litellm_accessgroupmembership.create_many = AsyncMock(return_value=0)
    db.litellm_accessgroupmembership.delete_many = AsyncMock(return_value=0)
    client = MagicMock()
    client.db = db
    return client


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    """Module-level cache state must not leak between tests."""
    invalidate_group_memberships_cache()
    yield
    invalidate_group_memberships_cache()


@pytest.mark.asyncio
async def test_cache_miss_then_hit_avoids_second_db_query():
    prisma = _make_prisma(membership_rows=[_row("project-x", "image")])

    first = await get_cached_group_memberships(prisma_client=prisma)
    second = await get_cached_group_memberships(prisma_client=prisma)

    assert first == second == {"project-x": ["image"]}
    # Only the first call should hit the DB
    prisma.db.litellm_accessgroupmembership.find_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_invalidation_forces_db_refetch():
    prisma = _make_prisma(membership_rows=[_row("project-x", "image")])

    await get_cached_group_memberships(prisma_client=prisma)
    invalidate_group_memberships_cache()
    await get_cached_group_memberships(prisma_client=prisma)

    assert prisma.db.litellm_accessgroupmembership.find_many.await_count == 2


@pytest.mark.asyncio
async def test_upsert_invalidates_cache():
    """Writing edges must drop the cache so the next read sees the change."""
    prisma = _make_prisma(membership_rows=[_row("project-x", "image")])
    prisma.db.litellm_accessgroupmembership.create_many = AsyncMock(return_value=1)

    await get_cached_group_memberships(prisma_client=prisma)  # populates cache
    await upsert_group_memberships(
        parent_group="project-x",
        child_groups=["reasoning"],
        prisma_client=prisma,
    )
    await get_cached_group_memberships(prisma_client=prisma)  # must re-fetch

    assert prisma.db.litellm_accessgroupmembership.find_many.await_count == 2


@pytest.mark.asyncio
async def test_delete_edges_invalidates_cache():
    """Deleting edges must drop the cache too."""
    prisma = _make_prisma(membership_rows=[_row("project-x", "image")])
    prisma.db.litellm_accessgroupmembership.delete_many = AsyncMock(return_value=1)

    await get_cached_group_memberships(prisma_client=prisma)
    await delete_group_membership_edges(access_group="project-x", prisma_client=prisma)
    await get_cached_group_memberships(prisma_client=prisma)

    assert prisma.db.litellm_accessgroupmembership.find_many.await_count == 2


@pytest.mark.asyncio
async def test_cache_expires_after_ttl(monkeypatch):
    """When monotonic time advances past the TTL, the next read re-fetches."""
    prisma = _make_prisma(membership_rows=[_row("project-x", "image")])

    # Freeze time; advance past TTL between calls
    now = [1000.0]
    monkeypatch.setattr(mgmt.time, "monotonic", lambda: now[0])

    await get_cached_group_memberships(prisma_client=prisma)
    now[0] += mgmt._MEMBERSHIPS_CACHE_TTL_SECONDS + 1
    await get_cached_group_memberships(prisma_client=prisma)

    assert prisma.db.litellm_accessgroupmembership.find_many.await_count == 2


@pytest.mark.asyncio
async def test_cache_within_ttl_does_not_refetch(monkeypatch):
    """Reads inside the TTL window stay served from cache."""
    prisma = _make_prisma(membership_rows=[_row("project-x", "image")])

    now = [1000.0]
    monkeypatch.setattr(mgmt.time, "monotonic", lambda: now[0])

    await get_cached_group_memberships(prisma_client=prisma)
    now[0] += mgmt._MEMBERSHIPS_CACHE_TTL_SECONDS - 1
    await get_cached_group_memberships(prisma_client=prisma)

    prisma.db.litellm_accessgroupmembership.find_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_falls_through_empty_dict_on_error_path():
    """When the underlying helper returns {} due to a DB error, the cache
    still stores it - we don't want to retry on every single request."""
    prisma = _make_prisma()
    prisma.db.litellm_accessgroupmembership.find_many = AsyncMock(
        side_effect=ConnectionError("postgres unreachable")
    )

    first = await get_cached_group_memberships(prisma_client=prisma)
    second = await get_cached_group_memberships(prisma_client=prisma)

    assert first == second == {}
    # Only one DB attempt; subsequent calls served from the cached {}
    prisma.db.litellm_accessgroupmembership.find_many.assert_awaited_once()
