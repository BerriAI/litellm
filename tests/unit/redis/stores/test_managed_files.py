import json
import time
from collections.abc import Mapping
from typing import Final, cast
from unittest.mock import patch

import fakeredis
import pytest
from litellm_enterprise.proxy.hooks.managed_files import PROXY_LiteLLMManagedFiles
from pydantic import ValidationError
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.dual_cache import DualCache
from litellm.models.managed_files import LiteLLM_ManagedFileTable
from litellm.proxy.utils import InternalUsageCache, PrismaClient
from litellm.types.llms.openai import OpenAIFileObject
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from tests._support.fake_redis import fake_redis_cache
from tests.unit.enterprise.proxy.hooks.test_managed_files import (
    _InMemoryManagedFilesPrismaClient,
    _InMemoryManagedFileTable,
)


def _row(file_id: str = "unified-é") -> LiteLLM_ManagedFileTable:
    return LiteLLM_ManagedFileTable(
        unified_file_id=file_id,
        file_object=OpenAIFileObject(
            id=file_id, object="file", bytes=123, created_at=456, filename="文.jsonl", purpose="batch"
        ),
        model_mappings={"model": "provider-file"},
        flat_model_file_ids=["provider-file"],
        created_by="owner",
        updated_by="owner",
    )


def _hook(cache: DualCache, table: _InMemoryManagedFileTable) -> PROXY_LiteLLMManagedFiles:
    return PROXY_LiteLLMManagedFiles(
        InternalUsageCache(cache), cast(PrismaClient, _InMemoryManagedFilesPrismaClient(table))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("file_id", ["unified-é", "tenant:unified-é"])
@pytest.mark.parametrize("ttl", [1, 17.9, None, 0.5])
async def test_file_metadata_matches_legacy_wire_and_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, file_id: str, ttl: float | None
) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=ttl, default_redis_ttl=999)
    hook: Final = _hook(dual, _InMemoryManagedFileTable())
    row: Final = _row(file_id)
    key: Final = file_id if file_id.startswith("tenant:") else f"tenant:{file_id}"
    with patch("time.time", return_value=time.time()):
        await hook.store_unified_file_id(
            file_id, row.file_object, None, row.model_mappings, UserAPIKeyAuth(user_id="owner")
        )
        encoded: Final = await raw.get(key)
        expiry: Final = await raw.pttl(key)
        await cache.async_set_cache("legacy", row.model_dump(), **({"ttl": ttl} if ttl is not None else {}))
        assert encoded == await raw.get("tenant:legacy")
        assert expiry == await raw.pttl("tenant:legacy")
        assert encoded == (None if ttl == 0.5 else json.dumps(row.model_dump()).encode())
        assert await cache.async_get_cache(file_id) == (None if ttl == 0.5 else row.model_dump())
        assert await raw.get(f"tenant:{key}") is None


@pytest.mark.asyncio
async def test_old_writer_new_reader_backfills_only_redis_hits(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=1)
    row: Final = _row()
    table: Final = _InMemoryManagedFileTable(row)
    hook: Final = _hook(dual, table)
    await cache.async_set_cache(row.unified_file_id, row.model_dump(), ttl=60)
    assert await hook.get_unified_file_id(row.unified_file_id) == row
    assert table.find_first_calls == []
    server.connected = False
    assert await hook.get_unified_file_id(row.unified_file_id) == row
    server.connected = True
    dual.in_memory_cache.flush_cache()
    await raw.delete(f"tenant:{row.unified_file_id}")
    assert await hook.get_unified_file_id(row.unified_file_id) == row
    assert await raw.get(f"tenant:{row.unified_file_id}") is None
    assert len(table.find_first_calls) == 1


@pytest.mark.asyncio
async def test_file_expiry_and_legacy_literal_record(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    row: Final = _row()
    with patch("time.time", return_value=1_800_000_000):
        cache: Final = fake_redis_cache(server)
        table: Final = _InMemoryManagedFileTable(row)
        hook: Final = _hook(DualCache(redis_cache=cache, default_in_memory_ttl=1), table)
        await raw.set(row.unified_file_id, repr(row.model_dump()), ex=1)
        assert await hook.get_unified_file_id(row.unified_file_id) == row
        assert table.find_first_calls == []
    with patch("time.time", return_value=1_800_000_002):
        assert await hook.get_unified_file_id(row.unified_file_id) == row
        assert len(table.find_first_calls) == 1
        assert await raw.get(row.unified_file_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [b"null", b"{}", b"broken", b"\xff"])
async def test_file_misses_and_malformed_bytes_fall_back_to_database(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, encoded: bytes
) -> None:
    row: Final = _row()
    table: Final = _InMemoryManagedFileTable(row)
    hook: Final = _hook(DualCache(redis_cache=fake_redis_cache(server)), table)
    await raw.set(row.unified_file_id, encoded)
    assert await hook.get_unified_file_id(row.unified_file_id) == row
    assert len(table.find_first_calls) == 1
    assert await raw.get(row.unified_file_id) == encoded


@pytest.mark.asyncio
async def test_truthy_invalid_file_record_still_raises_validation_error(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    table: Final = _InMemoryManagedFileTable(_row())
    hook: Final = _hook(DualCache(redis_cache=fake_redis_cache(server)), table)
    await raw.set("unified-é", b'{"invalid": true}')
    with pytest.raises(ValidationError):
        await hook.get_unified_file_id("unified-é")
    assert table.find_first_calls == []


class _DeleteObservingTable(_InMemoryManagedFileTable):
    def __init__(self, raw: fakeredis.FakeAsyncRedis, row: LiteLLM_ManagedFileTable) -> None:
        super().__init__(row)
        self.raw = raw

    async def delete(self, where: Mapping[str, object]) -> LiteLLM_ManagedFileTable | None:
        assert await self.raw.get(f"tenant:{where['unified_file_id']}") == b"null"
        assert await self.raw.pttl(f"tenant:{where['unified_file_id']}") == 1000
        return await super().delete(where)


@pytest.mark.asyncio
async def test_file_delete_writes_null_before_database_delete(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    row: Final = _row()
    table: Final = _DeleteObservingTable(raw, row)
    hook: Final = _hook(
        DualCache(redis_cache=fake_redis_cache(server, namespace="tenant"), default_in_memory_ttl=1), table
    )
    with patch("time.time", return_value=time.time()):
        assert await hook.delete_unified_file_id(row.unified_file_id) == row.file_object
        assert table.rows == {}
        assert await hook.get_unified_file_id(row.unified_file_id) is None
        assert len(table.find_first_calls) == 2


@pytest.mark.asyncio
async def test_file_outage_keeps_local_write_and_database_policy(server: fakeredis.FakeServer) -> None:
    row: Final = _row()
    cache: Final = fake_redis_cache(server)
    table: Final = _InMemoryManagedFileTable()
    dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=1)
    hook: Final = _hook(dual, table)
    server.connected = False
    await hook.store_unified_file_id(
        row.unified_file_id, row.file_object, None, row.model_mappings, UserAPIKeyAuth(user_id="owner")
    )
    assert await hook.get_unified_file_id(row.unified_file_id) == row
    assert table.rows[row.unified_file_id] == row
    dual.in_memory_cache.flush_cache()
    for _ in range(8):
        assert await hook.get_unified_file_id(row.unified_file_id) == row
    assert cache._circuit_breaker.state == "open"


@pytest.mark.asyncio
async def test_unknown_applied_file_write_is_not_rolled_back(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    row: Final = _row()
    table: Final = _InMemoryManagedFileTable()
    hook: Final = _hook(DualCache(redis_cache=cache, default_in_memory_ttl=1), table)

    async def applied_then_timeout(name: str, value: str, *, px: int | None, nx: bool) -> bool:
        await raw.set(name, value, px=px, nx=nx)
        raise RedisTimeoutError("reply lost after apply")

    with patch.object(cache.init_async_client(), "set", side_effect=applied_then_timeout):
        await hook.store_unified_file_id(
            row.unified_file_id, row.file_object, None, row.model_mappings, UserAPIKeyAuth(user_id="owner")
        )
    assert await raw.get(f"tenant:{row.unified_file_id}") == json.dumps(row.model_dump()).encode()
    assert table.rows[row.unified_file_id] == row


@pytest.mark.asyncio
async def test_file_refresh_and_lost_compare_and_swap_preserve_cache_order(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    row: Final = _row()
    table: Final = _InMemoryManagedFileTable(row)
    hook: Final = _hook(DualCache(redis_cache=fake_redis_cache(server), default_in_memory_ttl=1), table)
    updated: Final = row.file_object.model_copy(update={"bytes": 987})
    await hook._save_refreshed_file_object(row, updated)
    assert await raw.get(row.unified_file_id) == json.dumps(table.rows[row.unified_file_id].model_dump()).encode()
    assert await hook._save_xai_refreshed_file_object(row, updated) is None
    assert await raw.get(row.unified_file_id) == b"null"
    assert table.rows[row.unified_file_id].file_object == updated
    assert await hook._save_xai_refreshed_file_object(table.rows[row.unified_file_id], updated) == updated
    assert await raw.get(row.unified_file_id) == json.dumps(table.rows[row.unified_file_id].model_dump()).encode()
