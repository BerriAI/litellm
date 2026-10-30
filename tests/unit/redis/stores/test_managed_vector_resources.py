import base64
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import patch

import fakeredis
import pytest
from litellm_enterprise.proxy.hooks.managed_vector_stores import PROXY_LiteLLMManagedVectorStores
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.dual_cache import DualCache
from litellm.proxy.utils import InternalUsageCache, PrismaClient
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.vector_stores import VectorStoreCreateResponse
from tests._support.fake_redis import fake_redis_cache


@dataclass(frozen=True)
class _Row:
    data: Mapping[str, object]

    @property
    def resource_object(self) -> object:
        return self.data.get("resource_object")

    def model_dump(self) -> dict[str, object]:
        return dict(self.data)


class _Table:
    def __init__(self, row: _Row | None = None, raw: fakeredis.FakeAsyncRedis | None = None) -> None:
        self.row = row
        self.reads = 0
        self.raw = raw

    async def create(self, data: Mapping[str, object]) -> _Row:
        self.row = _Row(data)
        return self.row

    async def find_first(self, where: Mapping[str, object]) -> _Row | None:
        self.reads += 1
        return self.row if self.row and self.row.data["unified_resource_id"] == where["unified_resource_id"] else None

    async def delete(self, where: Mapping[str, object]) -> None:
        if self.raw is not None:
            assert await self.raw.get(f"tenant:{where['unified_resource_id']}") == b"null"
        self.row = None


def _resource() -> VectorStoreCreateResponse:
    return cast(VectorStoreCreateResponse, {"id": "vs-é", "object": "vector_store", "name": "文", "usage_bytes": 2**70})


def _hook(cache: DualCache, table: _Table) -> PROXY_LiteLLMManagedVectorStores:
    return PROXY_LiteLLMManagedVectorStores(
        InternalUsageCache(cache),
        cast(PrismaClient, SimpleNamespace(db=SimpleNamespace(litellm_managedvectorstoretable=table))),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [1, 12.9, None, 0.5, 0, -1])
async def test_vector_wire_matches_legacy_with_namespace_and_effective_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, ttl: float | None
) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=ttl, default_redis_ttl=999)
    dual.update_cache_ttl(ttl, None)
    table: Final = _Table()
    hook: Final = _hook(dual, table)
    resource: Final = _resource()
    unified_id: Final = hook.generate_unified_resource_id([resource], ["model-a"])
    assert (
        base64.urlsafe_b64decode(unified_id + "=" * (-len(unified_id) % 4))
        .decode()
        .startswith("litellm_proxy:vector_store;unified_id,")
    )
    record: Final = {
        "unified_resource_id": unified_id,
        "resource_object": resource,
        "model_mappings": {"model-a": "vs-é"},
        "flat_model_resource_ids": ["vs-é"],
        "created_by": "owner",
        "team_id": "team-a",
        "updated_by": "owner",
        "extra": "文",
    }
    with patch("time.time", return_value=time.time()):
        await hook.store_unified_resource_id(
            unified_id,
            resource,
            None,
            {"model-a": "vs-é"},
            UserAPIKeyAuth(user_id="owner", team_id="team-a"),
            {"extra": "文"},
        )
        await cache.async_set_cache("legacy", record, **({"ttl": ttl} if ttl is not None else {}))
        assert await raw.get(f"tenant:{unified_id}") == await raw.get("tenant:legacy")
        assert await raw.pttl(f"tenant:{unified_id}") == await raw.pttl("tenant:legacy")
        assert await raw.get(f"tenant:{unified_id}") == (
            None if ttl is not None and ttl < 1 else json.dumps(record).encode()
        )
        assert await cache.async_get_cache(unified_id) == (None if ttl is not None and ttl < 1 else record)
        assert await raw.get(f"tenant:tenant:{unified_id}") is None
        assert table.row is not None
        assert table.row.data == {
            **record,
            "resource_object": json.dumps(resource),
            "model_mappings": json.dumps(record["model_mappings"]),
        }


@pytest.mark.asyncio
async def test_old_vector_writer_new_reader_preserves_access_mapping_and_expiry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    resource_id: Final = (
        base64.urlsafe_b64encode(
            b"litellm_proxy:vector_store;unified_id,uuid;target_model_names,model;resource_id,vs-1;model_id,model"
        )
        .decode()
        .rstrip("=")
    )
    record: Final = {
        "unified_resource_id": resource_id,
        "resource_object": _resource(),
        "model_mappings": {"model": "vs-é"},
        "created_by": "owner",
        "team_id": "team-a",
    }
    with patch("time.time", return_value=1_800_000_000):
        cache: Final = fake_redis_cache(server, namespace="tenant")
        dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=1)
        table: Final = _Table(_Row({**record, "model_mappings": json.dumps(record["model_mappings"])}))
        hook: Final = _hook(dual, table)
        await cache.async_set_cache(resource_id, record, ttl=1)
        assert await hook.get_unified_resource_id(resource_id) == record
        assert await hook.can_user_access_unified_resource_id(resource_id, UserAPIKeyAuth(user_id="owner")) is True
        assert await hook.can_user_access_unified_resource_id(resource_id, UserAPIKeyAuth(team_id="team-a")) is True
        assert await hook.can_user_access_unified_resource_id(resource_id, UserAPIKeyAuth(team_id="team-b")) is False
        assert await hook.get_model_resource_id_mapping([resource_id], None) == {resource_id: {"model": "vs-é"}}
        assert table.reads == 0
    with patch("time.time", return_value=1_800_000_002):
        assert await hook.get_model_resource_id_mapping([resource_id], None) == {resource_id: {"model": "vs-é"}}
        assert table.reads == 1
        assert await raw.get(f"tenant:{resource_id}") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [b"null", b"{}", b"broken", b"\xff"])
async def test_vector_null_or_malformed_value_falls_back_without_backfill(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, encoded: bytes
) -> None:
    row: Final = _Row({"unified_resource_id": "rid", "resource_object": _resource()})
    table: Final = _Table(row)
    hook: Final = _hook(DualCache(redis_cache=fake_redis_cache(server)), table)
    await raw.set("rid", encoded)
    assert await hook.get_unified_resource_id("rid") == row.model_dump()
    assert await raw.get("rid") == encoded
    assert table.reads == 1


@pytest.mark.asyncio
async def test_recursive_vector_metadata_retains_legacy_database_fallback(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    row: Final = _Row({"unified_resource_id": "rid", "resource_object": _resource()})
    table: Final = _Table(row)
    cache: Final = fake_redis_cache(server)
    hook: Final = _hook(DualCache(redis_cache=cache), table)
    await raw.set("rid", b"[" * 10000 + b"0" + b"]" * 10000)
    limit: Final = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(1000)
        assert await cache.async_get_cache("rid") is None
        assert await hook.get_unified_resource_id("rid") == row.model_dump()
        assert table.reads == 1
    finally:
        sys.setrecursionlimit(limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [b"(1, 2)", b"{'item'}", b"{1: 'value'}"])
async def test_python_literal_vector_value_retains_legacy_read_outcome(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, encoded: bytes
) -> None:
    cache: Final = fake_redis_cache(server)
    table: Final = _Table(_Row({"unified_resource_id": "rid", "resource_object": _resource()}))
    hook: Final = _hook(DualCache(redis_cache=cache), table)
    await raw.set("rid", encoded)
    assert await hook.get_unified_resource_id("rid") == await cache.async_get_cache("rid")
    assert table.reads == 0


@pytest.mark.asyncio
async def test_vector_invalidation_precedes_database_delete_and_keeps_null_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    row: Final = _Row({"unified_resource_id": "rid", "resource_object": _resource()})
    table: Final = _Table(row, raw)
    hook: Final = _hook(
        DualCache(redis_cache=fake_redis_cache(server, namespace="tenant"), default_in_memory_ttl=1), table
    )
    with patch("time.time", return_value=time.time()):
        assert await hook.delete_unified_resource_id("rid") == _resource()
        assert await raw.get("tenant:rid") == b"null"
        assert await raw.pttl("tenant:rid") == 1000
        assert await hook.get_unified_resource_id("rid") is None
        assert table.reads == 2


@pytest.mark.asyncio
async def test_vector_outage_keeps_local_and_database_writes(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    dual: Final = DualCache(redis_cache=cache)
    table: Final = _Table()
    hook: Final = _hook(dual, table)
    server.connected = False
    await hook.store_unified_resource_id("rid", _resource(), None, {"model": "vs-é"}, UserAPIKeyAuth(user_id="owner"))
    cached: Final = await hook.get_unified_resource_id("rid")
    assert cached is not None
    assert cached["resource_object"] == _resource()
    assert table.reads == 0
    dual.in_memory_cache.flush_cache()
    assert table.row is not None
    for _ in range(8):
        assert await hook.get_unified_resource_id("rid") == table.row.model_dump()
    assert cache._circuit_breaker.state == "open"  # pyright: ignore[reportPrivateUsage]  # asserts the breaker opened after the outage


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), float("-inf")])
async def test_invalid_vector_cache_ttl_does_not_block_database_write(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, ttl: float
) -> None:
    table: Final = _Table()
    dual: Final = DualCache(redis_cache=fake_redis_cache(server), default_in_memory_ttl=ttl)
    hook: Final = _hook(dual, table)
    with patch("time.time", return_value=1_800_000_000):
        await dual.async_set_cache("legacy", _resource())
        await hook.store_unified_resource_id(
            "rid", _resource(), None, {"model": "vs-é"}, UserAPIKeyAuth(user_id="owner")
        )
        assert table.row is not None
        assert table.row.data["unified_resource_id"] == "rid"
        assert (await raw.get("rid") is None) == (await raw.get("legacy") is None)
        assert await raw.pttl("rid") == await raw.pttl("legacy")


@pytest.mark.asyncio
async def test_vector_unknown_applied_write_does_not_undo_cache_or_database(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server)
    table: Final = _Table()
    hook: Final = _hook(DualCache(redis_cache=cache, default_in_memory_ttl=1), table)

    async def applied_then_timeout(name: str, value: str, *, ex: timedelta | None, nx: bool) -> bool:
        await raw.set(name, value, ex=ex, nx=nx)
        raise RedisTimeoutError("reply lost after apply")

    with patch.object(cache.init_async_client(), "set", side_effect=applied_then_timeout):
        await hook.store_unified_resource_id(
            "rid", _resource(), None, {"model": "vs-é"}, UserAPIKeyAuth(user_id="owner")
        )
    assert await raw.get("rid") == json.dumps(await hook.get_unified_resource_id("rid")).encode()
    assert table.row is not None


@pytest.mark.asyncio
async def test_absent_vector_object_does_not_overwrite_cache(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    table: Final = _Table()
    hook: Final = _hook(DualCache(redis_cache=fake_redis_cache(server)), table)
    await raw.set("rid", b"previous")
    await hook.store_unified_resource_id("rid", None, None, {"model": "vs-é"}, UserAPIKeyAuth())
    assert await raw.get("rid") == b"previous"
    assert table.row is not None
    assert "resource_object" not in table.row.data
