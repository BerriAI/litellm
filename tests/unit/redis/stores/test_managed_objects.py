import json
import time
from collections.abc import Mapping
from datetime import timedelta
from types import SimpleNamespace
from typing import Final, Literal, cast
from unittest.mock import patch

import fakeredis
import pytest
from litellm_enterprise.proxy.hooks.managed_files import PROXY_LiteLLMManagedFiles
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.dual_cache import DualCache
from litellm.models.managed_files import LiteLLM_ManagedObjectTable
from litellm.proxy.utils import InternalUsageCache, PrismaClient
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.utils import LiteLLMBatch, LiteLLMFineTuningJob
from tests._support.fake_redis import fake_redis_cache


class _Table:
    def __init__(self, raw: fakeredis.FakeAsyncRedis | None = None, expected: bytes | None = None) -> None:
        self.raw = raw
        self.expected = expected
        self.created: Mapping[str, object] | None = None
        self.updated: Mapping[str, object] | None = None

    async def upsert(self, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        if self.raw is not None:
            assert await self.raw.get(f"tenant:{where['unified_object_id']}") == self.expected
        self.created = data

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int:
        if self.raw is not None:
            assert await self.raw.get(f"tenant:{where['unified_object_id']}") == self.expected
        self.updated = data
        return 0


def _hook(cache: DualCache, table: _Table) -> PROXY_LiteLLMManagedFiles:
    db: Final = SimpleNamespace(litellm_managedobjecttable=table)
    return PROXY_LiteLLMManagedFiles(
        InternalUsageCache(cache), cast(PrismaClient, SimpleNamespace(db=db, writer_db=db))
    )


def _object(
    purpose: Literal["batch", "fine-tune", "response"],
) -> LiteLLMBatch | LiteLLMFineTuningJob | ResponsesAPIResponse:
    if purpose == "batch":
        return LiteLLMBatch(
            id="batch-é",
            completion_window="24h",
            created_at=123,
            endpoint="/v1/chat/completions",
            input_file_id="input-文",
            object="batch",
            status="completed",
        )
    if purpose == "fine-tune":
        return LiteLLMFineTuningJob(
            id="ft-é",
            object="fine_tuning.job",
            model="model",
            created_at=123,
            organization_id="org",
            result_files=[],
            status="validating_files",
            training_file="file-文",
            hyperparameters={"n_epochs": 8},
            seed=0,
        )
    return ResponsesAPIResponse(
        id="resp-é",
        created_at=123,
        status="completed",
        output=[
            {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "private generated output", "annotations": []}],
            }
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["batch", "fine-tune", "response"])
@pytest.mark.parametrize("ttl", [1, 21.9, None, 0.5, 0, -1])
async def test_object_wire_and_ttl_match_old_reader_and_write_precedes_database(
    server: fakeredis.FakeServer,
    raw: fakeredis.FakeAsyncRedis,
    purpose: Literal["batch", "fine-tune", "response"],
    ttl: float | None,
) -> None:
    value: Final = _object(purpose)
    sanitized: Final = value.model_copy(update={"output": []}) if isinstance(value, ResponsesAPIResponse) else value
    record: Final = LiteLLM_ManagedObjectTable(
        unified_object_id="object-é",
        model_object_id=value.id,
        file_purpose=purpose,
        file_object=sanitized,
    ).model_dump()
    expected: Final = None if ttl is not None and ttl < 1 else json.dumps(record).encode()
    cache: Final = fake_redis_cache(server, namespace="tenant")
    table: Final = _Table(raw, expected)
    dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=ttl, default_redis_ttl=999)
    dual.update_cache_ttl(ttl, None)
    hook: Final = _hook(dual, table)
    with patch("time.time", return_value=time.time()):
        await hook.store_unified_object_id("object-é", value, None, value.id, purpose, UserAPIKeyAuth(user_id="owner"))
        await cache.async_set_cache("legacy", record, **({"ttl": ttl} if ttl is not None else {}))
        assert await raw.get("tenant:object-é") == await raw.get("tenant:legacy") == expected
        assert await raw.pttl("tenant:object-é") == await raw.pttl("tenant:legacy")
        assert await cache.async_get_cache("object-é") == (None if ttl is not None and ttl < 1 else record)
        assert await raw.get("tenant:tenant:object-é") is None
        assert await dual.async_get_cache("object-é", local_only=True) == (
            None if ttl is not None and ttl < 0 else record
        )
    assert table.created is not None
    assert "private generated output" not in json.dumps(table.created, default=str)
    if isinstance(value, ResponsesAPIResponse):
        assert len(value.output) == 1  # pyright: ignore[reportUnknownArgumentType]  # output carries a bare dict member


@pytest.mark.asyncio
async def test_object_update_only_still_writes_and_expires_cache(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    value: Final = _object("batch")
    with patch("time.time", return_value=1_800_000_000):
        cache: Final = fake_redis_cache(server)
        dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=1)
        table: Final = _Table()
        hook: Final = _hook(dual, table)
        await hook.store_unified_object_id(
            "object", value, None, value.id, "batch", UserAPIKeyAuth(user_id="poller"), create_if_missing=False
        )
        assert await raw.pttl("object") == 1000
        assert table.created is None
        assert table.updated is not None
        assert "created_by" not in table.updated
    with patch("time.time", return_value=1_800_000_002):
        assert await raw.get("object") is None
        assert await dual.async_get_cache("object", local_only=True) is None


@pytest.mark.asyncio
async def test_object_outage_keeps_local_and_database_write(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    dual: Final = DualCache(redis_cache=cache)
    table: Final = _Table()
    hook: Final = _hook(dual, table)
    value: Final = _object("batch")
    server.connected = False
    for _ in range(8):
        await hook.store_unified_object_id("object", value, None, value.id, "batch", UserAPIKeyAuth(user_id="owner"))
    assert table.created is not None
    assert (
        await dual.async_get_cache("object", local_only=True)
        == LiteLLM_ManagedObjectTable(
            unified_object_id="object",
            model_object_id=value.id,
            file_purpose="batch",
            file_object=value,
        ).model_dump()
    )
    assert cache._circuit_breaker.state == "open"  # pyright: ignore[reportPrivateUsage]  # asserts the breaker opened after the outage


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), float("-inf")])
async def test_invalid_object_cache_ttl_does_not_block_database_write(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, ttl: float
) -> None:
    table: Final = _Table()
    dual: Final = DualCache(redis_cache=fake_redis_cache(server), default_in_memory_ttl=ttl)
    hook: Final = _hook(dual, table)
    value: Final = _object("batch")
    with patch("time.time", return_value=1_800_000_000):
        await dual.async_set_cache("legacy", value.model_dump())
        await hook.store_unified_object_id("object", value, None, value.id, "batch", UserAPIKeyAuth(user_id="owner"))
        assert table.created is not None
        assert (await raw.get("object") is None) == (await raw.get("legacy") is None)
        assert await raw.pttl("object") == await raw.pttl("legacy")


@pytest.mark.asyncio
async def test_object_unknown_applied_write_is_not_undone(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server)
    dual: Final = DualCache(redis_cache=cache, default_in_memory_ttl=1)
    table: Final = _Table()
    hook: Final = _hook(dual, table)
    value: Final = _object("batch")

    async def applied_then_timeout(name: str, value: str, *, ex: timedelta | None, nx: bool) -> bool:
        await raw.set(name, value, ex=ex, nx=nx)
        raise RedisTimeoutError("reply lost after apply")

    with patch.object(cache.init_async_client(), "set", side_effect=applied_then_timeout):
        await hook.store_unified_object_id("object", value, None, value.id, "batch", UserAPIKeyAuth(user_id="owner"))
    assert await raw.get("object") == json.dumps(await dual.async_get_cache("object", local_only=True)).encode()
    assert table.created is not None
