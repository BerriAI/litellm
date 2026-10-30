import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.constants import LITELLM_CONFIG_CACHE_TTL_SECONDS
from litellm.proxy import utils
from litellm.redis.stores.config_params import CONFIG_PARAM_MISS, TransportConfigParamsStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


def config_database(row: object) -> utils.PrismaClient:
    client = utils.PrismaClient.__new__(utils.PrismaClient)
    client.db = MagicMock()
    client.db.litellm_config.find_first = AsyncMock(return_value=row)
    client.db.litellm_config.find_many = AsyncMock(return_value=[] if row is None else [row])
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", [None, "", "tenant"])
async def test_old_writer_new_reader_and_new_writer_old_reader(server, raw, monkeypatch, namespace):
    redis = fake_redis_cache(server, namespace=namespace)
    cache = DualCache(redis_cache=redis, default_in_memory_ttl=60)
    monkeypatch.setattr(utils, "litellm_config_cache", cache)
    key = "litellm_config:param:settings"
    wire_key = f"{namespace}:{key}" if namespace else key
    value = {"param_name": "settings", "param_value": {"é": [0, 1.25, False]}, "reload_revision": 42}
    database = config_database(SimpleNamespace(**value))

    with patch("time.time", return_value=1700000000):
        await redis.async_set_cache(key, value, ttl=60)
        row = await utils.get_config_param(database, "settings")
        assert (row.param_name, row.param_value, row.reload_revision) == ("settings", value["param_value"], 42)
        await utils.evict_config_param("settings")
        assert await raw.get(wire_key) is None
        assert cache.in_memory_cache.get_cache(key) is None
        await utils.get_config_param(database, "settings")
        assert await raw.get(wire_key) == json.dumps(value).encode()
        assert await raw.pttl(wire_key) == LITELLM_CONFIG_CACHE_TTL_SECONDS * 1000
        assert await redis.async_get_cache(key) == value


@pytest.mark.asyncio
async def test_prefetch_caches_rows_and_negative_lookups(server, raw, monkeypatch):
    redis = fake_redis_cache(server, namespace="tenant")
    cache = DualCache(redis_cache=redis)
    monkeypatch.setattr(utils, "litellm_config_cache", cache)
    row = SimpleNamespace(param_name="present", param_value='{"é": 0}', reload_revision=8)
    database = config_database(row)
    await utils.prefetch_config_params(database, ["present", "missing"])
    cache.in_memory_cache.flush_cache()
    assert await utils.get_config_param(database, "missing") is None
    loaded = await utils.get_config_param(database, "present")
    assert (loaded.param_value, loaded.reload_revision) == (row.param_value, 8)
    assert await raw.get("tenant:litellm_config:param:missing") == json.dumps(CONFIG_PARAM_MISS).encode()
    assert database.db.litellm_config.find_first.await_count == 0


@pytest.mark.asyncio
async def test_outage_falls_back_to_database_and_keeps_local_value(server, monkeypatch):
    redis = fake_redis_cache(server, namespace="tenant")
    cache = DualCache(redis_cache=redis)
    monkeypatch.setattr(utils, "litellm_config_cache", cache)
    database = config_database(SimpleNamespace(param_name="settings", param_value=0))
    server.connected = False
    row = await utils.get_config_param(database, "settings")
    assert row.param_value == 0
    assert (await utils.get_config_param(database, "settings")).param_value == 0
    assert database.db.litellm_config.find_first.await_count == 1
    await utils.evict_config_param("settings")
    assert cache.in_memory_cache.get_cache("litellm_config:param:settings") is None


@pytest.mark.asyncio
async def test_ambiguous_fill_keeps_applied_value_without_replay(server, raw, monkeypatch):
    redis = fake_redis_cache(server, namespace="tenant")
    monkeypatch.setattr(utils, "litellm_config_cache", DualCache(redis_cache=redis))
    database = config_database(SimpleNamespace(param_name="settings", param_value=False))
    client = redis.init_async_client()
    original_set = client.set

    async def apply_then_disconnect(*args, **kwargs):
        await original_set(*args, **kwargs)
        raise RedisConnectionError("reply lost")

    with patch.object(client, "set", side_effect=apply_then_disconnect) as write:
        assert (await utils.get_config_param(database, "settings")).param_value is False
        assert write.call_count == 1
    assert json.loads(await raw.get("tenant:litellm_config:param:settings")) == {
        "param_name": "settings",
        "param_value": False,
        "reload_revision": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"not-json", b"\xff"])
async def test_malformed_payload_is_decode_failure_and_caller_refills(server, raw, monkeypatch, payload):
    redis = fake_redis_cache(server)
    monkeypatch.setattr(utils, "litellm_config_cache", DualCache(redis_cache=redis))
    await raw.set("litellm_config:param:settings", payload)
    failure = await TransportConfigParamsStore(redis.transport()).load("settings")
    assert isinstance(failure, RedisFailure) and failure.kind == "decode"
    row = await utils.get_config_param(
        config_database(SimpleNamespace(param_name="settings", param_value=7)), "settings"
    )
    assert row.param_value == 7


@pytest.mark.asyncio
async def test_breaker_open_keeps_database_fallback(server, monkeypatch):
    redis = fake_redis_cache(server)
    monkeypatch.setattr(utils, "litellm_config_cache", DualCache(redis_cache=redis))
    store = TransportConfigParamsStore(redis.transport())
    server.connected = False
    for _ in range(6):
        await store.load("settings")
    failure = await store.fill("settings", None, ttl=timedelta(seconds=60))
    assert isinstance(failure, RedisFailure) and failure.kind == "breaker_open"
    assert await utils.get_config_param(config_database(None), "settings") is None


@pytest.mark.asyncio
async def test_legacy_python_literal_retains_value_types(server, raw, monkeypatch):
    redis = fake_redis_cache(server)
    monkeypatch.setattr(utils, "litellm_config_cache", DualCache(redis_cache=redis))
    value = {"param_name": "settings", "param_value": (True, {1: "legacy"}), "reload_revision": True}
    await raw.set("litellm_config:param:settings", repr(value))
    row = await utils.get_config_param(config_database(None), "settings")
    assert row.param_value == value["param_value"]
    assert row.reload_revision is None


@pytest.mark.asyncio
async def test_invalid_expiry_and_unserializable_rows_leave_existing_value(server, raw):
    store = TransportConfigParamsStore(fake_redis_cache(server).transport())
    await raw.set("litellm_config:param:settings", "existing")
    for row, ttl, kind in [(None, timedelta(0), "server"), ({"value": object()}, timedelta(seconds=60), "decode")]:
        failure = await store.fill("settings", row, ttl=ttl)
        assert isinstance(failure, RedisFailure) and failure.kind == kind
        assert await raw.get("litellm_config:param:settings") == b"existing"


@pytest.mark.asyncio
async def test_request_known_miss_keeps_database_read_through(server, raw, monkeypatch):
    redis = fake_redis_cache(server)
    monkeypatch.setattr(utils, "litellm_config_cache", DualCache(redis_cache=redis))
    with request_redis_batch_scope() as scope:
        assert await scope.batch(redis).mget(["litellm_config:param:settings"]) == {
            "litellm_config:param:settings": None
        }
        await raw.set("litellm_config:param:settings", json.dumps({"param_name": "settings", "param_value": "old"}))
        row = await utils.get_config_param(
            config_database(SimpleNamespace(param_name="settings", param_value="db")), "settings"
        )
        assert row.param_value == "db"
        assert (await redis.async_get_cache("litellm_config:param:settings"))["param_value"] == "db"
