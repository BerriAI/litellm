import json
import time
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.moyai_endpoints import _persist_moyai_url
from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
    cache_ui_settings,
    get_ui_settings,
    get_ui_settings_cached,
    update_ui_settings,
)
from litellm.redis.stores.ui_settings import TransportUISettingsStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)


@pytest.mark.asyncio
async def test_old_writer_new_reader_preserves_values_and_local_backfill_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server, namespace="ns")
    cache: Final = DualCache(redis_cache=redis, default_in_memory_ttl=17)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    settings: Final = {"moyai_url": "https://example.com/é", "unknown": [False, None, 1, 1.25]}
    await redis.async_set_cache("ui_settings:settings_dict", settings, ttl=600)

    assert await get_ui_settings_cached() == settings
    assert await raw.get("ns:ui_settings:settings_dict") == json.dumps(settings).encode()
    assert await raw.pttl("ns:ui_settings:settings_dict") == 600_000
    assert cache.in_memory_cache.ttl_dict["ui_settings:settings_dict"] == 1_800_000_017.0
    server.connected = False
    assert await get_ui_settings_cached() == settings


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["fill", "get", "update", "moyai"])
async def test_all_writers_preserve_bytes_sanitization_and_ttl_for_old_readers(
    writer: Literal["fill", "get", "update", "moyai"],
    server: fakeredis.FakeServer,
    raw: fakeredis.FakeAsyncRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis: Final = fake_redis_cache(server, namespace="ns")
    cache: Final = DualCache(redis_cache=redis)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    prisma: Final = MagicMock()
    prisma.db.litellm_uisettings.find_unique = AsyncMock(
        return_value=SimpleNamespace(ui_settings={"moyai_url": "https://example.com/é", "unknown": [1, 1.5, None]})
    )
    prisma.db.litellm_uisettings.upsert = AsyncMock()
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    match writer:
        case "fill":
            assert await get_ui_settings_cached() == {"moyai_url": "https://example.com/é"}
        case "get":
            await get_ui_settings()
        case "update":
            await update_ui_settings({}, UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN))
        case "moyai":
            await _persist_moyai_url(prisma, "https://example.com/é")
    expected: Final = (
        {"moyai_url": "https://example.com/é", "unknown": [1, 1.5, None]}
        if writer == "moyai"
        else {"moyai_url": "https://example.com/é"}
    )

    assert await redis.async_get_cache("ui_settings:settings_dict") == expected
    assert await raw.get("ns:ui_settings:settings_dict") == json.dumps(expected).encode()
    assert await raw.keys("*") == [b"ns:ui_settings:settings_dict"]
    assert await raw.pttl("ns:ui_settings:settings_dict") == 600_000
    assert cache.in_memory_cache.ttl_dict["ui_settings:settings_dict"] == 1_800_000_600.0


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"{", b"[]", b"null", b"\xff"])
async def test_malformed_settings_fall_back_without_changing_the_stored_bytes(
    payload: bytes, server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache(redis_cache=redis))
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    await raw.set("ui_settings:settings_dict", payload)

    failure: Final = await TransportUISettingsStore(redis.transport()).get()
    assert isinstance(failure, RedisFailure)
    assert failure.kind == "decode"
    assert await get_ui_settings_cached() == {}
    assert await raw.get("ui_settings:settings_dict") == payload


@pytest.mark.asyncio
async def test_legacy_literal_and_empty_object_remain_cache_hits(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    for payload, expected in ((b"{'disable_custom_api_keys': True}", {"disable_custom_api_keys": True}), (b"{}", {})):
        monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache(redis_cache=redis))
        await raw.set("ui_settings:settings_dict", payload)
        assert await get_ui_settings_cached() == expected


@pytest.mark.asyncio
async def test_outage_and_open_breaker_keep_the_local_write_and_db_fallback(
    server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    server.connected = False

    assert await get_ui_settings_cached() == {}
    await cache_ui_settings(cache, {"disable_custom_api_keys": True})
    assert await get_ui_settings_cached() == {"disable_custom_api_keys": True}
    for _ in range(10):
        await TransportUISettingsStore(redis.transport()).get()
    server.connected = True
    refused: Final = await TransportUISettingsStore(redis.transport()).get()
    assert isinstance(refused, RedisFailure)
    assert refused.kind == "breaker_open"
    assert await get_ui_settings_cached() == {"disable_custom_api_keys": True}


@pytest.mark.asyncio
async def test_applied_write_timeout_keeps_local_value_without_replaying(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server, namespace="ns")
    cache: Final = DualCache(redis_cache=redis)
    client: Final = redis.init_async_client()

    async def applied_then_timeout(*args: object, **kwargs: object) -> None:
        await raw.execute_command("SET", "ns:ui_settings:settings_dict", '{"moyai_url": "first"}', "PX", 600_000)
        raise RedisTimeoutError("reply lost after SET")

    with patch.object(client, "set", side_effect=applied_then_timeout) as write:
        await cache_ui_settings(cache, {"moyai_url": "first"})
        assert write.call_count == 1

    assert await raw.get("ns:ui_settings:settings_dict") == b'{"moyai_url": "first"}'
    assert await cache.async_get_cache("ui_settings:settings_dict", local_only=True) == {"moyai_url": "first"}
    assert await raw.pttl("ns:ui_settings:settings_dict") == 600_000


@pytest.mark.asyncio
async def test_request_local_miss_stays_a_miss(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    with request_redis_batch_scope():
        deleted: Final = await cache.async_delete_cache_pre_call("ui_settings:settings_dict")
        assert deleted is not None
        await deleted
        await raw.set("ui_settings:settings_dict", '{"disable_custom_api_keys": true}')
        assert await get_ui_settings_cached() == {}
    assert await get_ui_settings_cached() == {"disable_custom_api_keys": True}
