import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.models.model import LiteLLM_ProxyModelTable
from litellm.proxy.model_list_cache import (
    MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS,
    MODEL_LIST_CACHE_KEY,
    MODEL_LIST_CACHE_REDIS_TTL_SECONDS,
    get_cached_model_rows,
    invalidate_model_list_cache,
    model_rows_fingerprint,
    parse_model_rows,
    refresh_model_list_cache,
    set_cached_model_rows,
)

UPDATED_AT = datetime(2026, 1, 1, 12, 0, 0)


def make_model(model_id: str, updated_at: datetime = UPDATED_AT) -> LiteLLM_ProxyModelTable:
    return LiteLLM_ProxyModelTable(
        model_id=model_id,
        model_name=f"group-{model_id}",
        litellm_params={"model": "openai/gpt-4.1-nano", "api_key": "encrypted-value"},
        model_info={"id": model_id},
        updated_at=updated_at,
    )


def pod_cache(redis_client: fakeredis.aioredis.FakeRedis) -> DualCache:
    """A pod's own DualCache, sharing the cluster's Redis with the other pods."""
    redis_cache = RedisCache(host="localhost", port="6379")
    redis_cache.init_async_client = lambda **kwargs: redis_client
    return DualCache(
        redis_cache=redis_cache,
        default_in_memory_ttl=MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS,
    )


@pytest.fixture
def shared_redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis()


class TestModelListCacheSharing:
    @pytest.mark.asyncio
    async def test_write_on_one_pod_is_readable_on_another(self, shared_redis):
        pod_a, pod_b = pod_cache(shared_redis), pod_cache(shared_redis)

        await set_cached_model_rows([make_model("m-1")], cache=pod_a)

        rows = await get_cached_model_rows(cache=pod_b)
        assert rows is not None
        assert [row.model_id for row in rows] == ["m-1"]
        assert rows[0].litellm_params["api_key"] == "encrypted-value"

    @pytest.mark.asyncio
    async def test_invalidation_on_one_pod_forces_a_db_read_on_another(self, shared_redis):
        """The fix: without this eviction the sibling pod keeps serving the pre-write list."""
        pod_a, pod_b = pod_cache(shared_redis), pod_cache(shared_redis)

        await set_cached_model_rows([make_model("m-1"), make_model("m-2")], cache=pod_a)
        assert await get_cached_model_rows(cache=pod_b) is not None

        await invalidate_model_list_cache(cache=pod_a)
        assert await shared_redis.get(MODEL_LIST_CACHE_KEY) is None

        await asyncio.sleep(MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS + 0.1)
        assert await get_cached_model_rows(cache=pod_b) is None

    @pytest.mark.asyncio
    async def test_shared_entry_outlives_the_local_copy(self, shared_redis):
        """A local TTL on the shared entry would send every pod to the DB on every sync."""
        await set_cached_model_rows([make_model("m-1")], cache=pod_cache(shared_redis))

        assert await shared_redis.ttl(MODEL_LIST_CACHE_KEY) > MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS
        assert MODEL_LIST_CACHE_REDIS_TTL_SECONDS > MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_reads_return_fresh_copies_so_callers_cannot_corrupt_the_cache(self, shared_redis):
        """`_add_deployment` decrypts `litellm_params` in place on whatever it is handed."""
        cache = pod_cache(shared_redis)
        await set_cached_model_rows([make_model("m-1")], cache=cache)

        first = await get_cached_model_rows(cache=cache)
        assert first is not None
        first[0].litellm_params["api_key"] = "decrypted-value"

        second = await get_cached_model_rows(cache=cache)
        assert second is not None
        assert second[0].litellm_params["api_key"] == "encrypted-value"

    @pytest.mark.asyncio
    async def test_unreadable_entry_falls_back_to_the_db(self, shared_redis):
        cache = pod_cache(shared_redis)
        await cache.async_set_cache(MODEL_LIST_CACHE_KEY, [{"not": "a model row"}], local_only=True)

        assert await get_cached_model_rows(cache=cache) is None

    @pytest.mark.asyncio
    async def test_empty_model_list_is_cached_as_empty_not_as_a_miss(self, shared_redis):
        """An operator who deleted every model must not send every pod back to the DB."""
        cache = pod_cache(shared_redis)
        await set_cached_model_rows([], cache=cache)

        assert await get_cached_model_rows(cache=cache) == ()

    @pytest.mark.asyncio
    async def test_read_fill_cannot_overwrite_a_racing_write(self, shared_redis):
        """A read that began pre-write must not republish its stale snapshot over the write.

        The write path publishes fresh rows; a slower read-fill holding the pre-write list
        then writes with `overwrite=False`, so the shared Redis copy keeps the fresh rows.
        """
        writer, reader = pod_cache(shared_redis), pod_cache(shared_redis)

        await set_cached_model_rows([make_model("m-new")], cache=writer)
        await set_cached_model_rows([make_model("m-stale")], cache=reader, overwrite=False)

        rows = await get_cached_model_rows(cache=pod_cache(shared_redis))
        assert rows is not None
        assert [row.model_id for row in rows] == ["m-new"]

    @pytest.mark.asyncio
    async def test_read_fill_still_populates_an_empty_shared_key(self, shared_redis):
        """`NX` must only decline when a value already exists, not disable read-through fills."""
        await set_cached_model_rows([make_model("m-1")], cache=pod_cache(shared_redis), overwrite=False)

        rows = await get_cached_model_rows(cache=pod_cache(shared_redis))
        assert rows is not None
        assert [row.model_id for row in rows] == ["m-1"]

    @pytest.mark.asyncio
    async def test_read_fill_does_not_populate_in_memory_when_write_races(self, shared_redis):
        """Read-fill must not replace a pod's in-memory cache even if the Redis NX succeeds.

        A write path publishes fresh rows; if a stale read-fill replaces the in-memory
        copy on the same pod, the pod will use obsolete rows until the 1s TTL lapses.
        So read-fill skips the local cache entirely, forcing the next sync to hit Redis.
        """
        cache = pod_cache(shared_redis)

        await set_cached_model_rows([make_model("m-fresh")], cache=cache, overwrite=True)
        assert await get_cached_model_rows(cache=cache) is not None
        assert (await get_cached_model_rows(cache=cache))[0].model_id == "m-fresh"

        await set_cached_model_rows([make_model("m-stale")], cache=cache, overwrite=False)

        rows = await get_cached_model_rows(cache=cache)
        assert rows is not None
        assert [row.model_id for row in rows] == ["m-fresh"], "in-memory must keep fresh rows"


class TestModelRowsFingerprint:
    def test_ignores_row_order(self):
        rows = [make_model("m-1"), make_model("m-2")]
        assert model_rows_fingerprint(rows) == model_rows_fingerprint(list(reversed(rows)))

    def test_changes_when_a_row_is_deleted(self):
        rows = [make_model("m-1"), make_model("m-2")]
        assert model_rows_fingerprint(rows) != model_rows_fingerprint(rows[:1])

    def test_changes_when_a_row_is_edited(self):
        edited = make_model("m-1", updated_at=UPDATED_AT + timedelta(seconds=1))
        assert model_rows_fingerprint([make_model("m-1")]) != model_rows_fingerprint([edited])


class TestParseModelRows:
    def test_accepts_orm_rows_and_mappings(self):
        row = make_model("m-1")
        assert parse_model_rows([row]) == parse_model_rows([row.model_dump()])

    def test_rejects_rows_it_cannot_read(self):
        with pytest.raises(TypeError):
            parse_model_rows(["not-a-row"])


class TestRefreshModelListCache:
    """Test the write-through refresh logic that fixes the stale-snapshot race."""

    @pytest.mark.asyncio
    async def test_refresh_writes_fresh_rows_to_both_cache_tiers(self, shared_redis):
        """The fix: refresh writes fresh data, not delete-only."""
        cache = pod_cache(shared_redis)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[make_model("m-1")])

        await refresh_model_list_cache(prisma_client=mock_prisma, cache=cache)

        rows = await get_cached_model_rows(cache=cache)
        assert rows is not None
        assert [row.model_id for row in rows] == ["m-1"]

    @pytest.mark.asyncio
    async def test_refresh_handles_empty_model_list(self, shared_redis):
        """A deleted model returns an empty list; refresh must cache that, not evict."""
        cache = pod_cache(shared_redis)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        await refresh_model_list_cache(prisma_client=mock_prisma, cache=cache)

        rows = await get_cached_model_rows(cache=cache)
        assert rows == ()  # Empty, not None

    @pytest.mark.asyncio
    async def test_refresh_evicts_on_db_error(self, shared_redis):
        """If the refresh fails, evict so the next read hits the DB fresh."""
        cache = pod_cache(shared_redis)
        await set_cached_model_rows([make_model("m-1")], cache=cache)

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(side_effect=RuntimeError("DB down"))

        await refresh_model_list_cache(prisma_client=mock_prisma, cache=cache)

        rows = await get_cached_model_rows(cache=cache)
        assert rows is None  # Evicted on error

    @pytest.mark.asyncio
    async def test_refresh_handles_none_prisma_client(self, shared_redis):
        """Guard against None prisma_client to avoid spurious errors."""
        cache = pod_cache(shared_redis)
        await set_cached_model_rows([make_model("m-1")], cache=cache)

        await refresh_model_list_cache(prisma_client=None, cache=cache)

        rows = await get_cached_model_rows(cache=cache)
        assert rows is None  # Evicted when prisma is None
