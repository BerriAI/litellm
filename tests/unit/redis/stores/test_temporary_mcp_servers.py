import json
import time
from types import SimpleNamespace
from typing import Final, NoReturn
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import ConnectionError

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.management_endpoints.mcp_management_endpoints import (
    _cache_temporary_mcp_server_in_redis,
    _get_temporary_mcp_server_from_redis,
)
from litellm.redis.stores.temporary_mcp_servers import TransportTemporaryMCPServersStore
from litellm.redis.transport.contract import RedisFailure
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from tests._support.fake_redis import fake_redis_cache


def lost_write_reply(response: object, **kwargs: object) -> NoReturn:
    raise ConnectionError("write applied but reply lost")


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "temporary-server-unit-test-key")
    monkeypatch.setattr(time, "time", lambda: 2_000_000_000)


@pytest.mark.asyncio
async def test_old_and_new_envelopes_share_response_cache(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server, namespace="mcp-test")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    entry: Final = MCPServer(server_id="temporary-雪", name="snow", url="https://example.com/mcp", transport="http")
    key: Final = f"litellm:mcp:temporary_server:{entry.server_id}"
    document: Final = json.dumps(entry.model_dump(mode="json"))
    old_envelope: Final = encrypt_value_helper(document)
    with patch.object(litellm, "cache", SimpleNamespace(cache=cache)):
        await cache.async_set_cache(key=key, value=old_envelope, ttl=300)
        assert await _get_temporary_mcp_server_from_redis(entry.server_id) == entry
        await _cache_temporary_mcp_server_in_redis(entry, ttl_seconds=123)
        envelope: Final = await cache.async_get_cache(key)
        assert isinstance(envelope, str)
        assert await raw.get(f"mcp-test:{key}") == json.dumps(envelope).encode()
        assert decrypt_value_helper(envelope, key="temporary_mcp_server") == document
        assert await raw.pttl(f"mcp-test:{key}") == 123_000
        assert await raw.get(key) is None
        assert await _get_temporary_mcp_server_from_redis(entry.server_id) == entry


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"{}", b'"not encrypted"', b"null", b"invalid", b'""'])
async def test_malformed_envelopes_remain_a_caller_miss(server: fakeredis.FakeServer, payload: bytes) -> None:
    cache: Final = fake_redis_cache(server, namespace="mcp-test")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set("mcp-test:litellm:mcp:temporary_server:bad", payload)
    failure: Final = await TransportTemporaryMCPServersStore(cache.transport()).get("bad")
    assert isinstance(failure, RedisFailure) and failure.kind == "decode"
    with patch.object(litellm, "cache", SimpleNamespace(cache=cache)):
        assert await _get_temporary_mcp_server_from_redis("bad") is None


@pytest.mark.asyncio
async def test_outage_and_open_breaker_are_best_effort(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server, namespace="mcp-test")
    entry: Final = MCPServer(server_id="outage", name="outage", transport="http")
    server.connected = False
    with patch.object(litellm, "cache", SimpleNamespace(cache=cache)):
        await _cache_temporary_mcp_server_in_redis(entry, ttl_seconds=300)
        assert await _get_temporary_mcp_server_from_redis("outage") is None
        for _ in range(10):
            await _get_temporary_mcp_server_from_redis("outage")
        failure: Final = await TransportTemporaryMCPServersStore(cache.transport()).get("outage")
        assert isinstance(failure, RedisFailure) and failure.kind == "breaker_open"


@pytest.mark.asyncio
async def test_absent_and_non_redis_backends_keep_the_same_fallback() -> None:
    entry: Final = MCPServer(server_id="memory", name="memory", transport="http")
    with patch.object(litellm, "cache", None):
        await _cache_temporary_mcp_server_in_redis(entry, ttl_seconds=300)
        assert await _get_temporary_mcp_server_from_redis("memory") is None
    with patch.object(litellm, "cache", SimpleNamespace(cache=InMemoryCache())):
        await _cache_temporary_mcp_server_in_redis(entry, ttl_seconds=0)
        assert await _get_temporary_mcp_server_from_redis("memory") == entry


@pytest.mark.asyncio
async def test_unknown_applied_write_keeps_envelope_and_ttl(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server, namespace="mcp-test")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    cache.init_async_client().set_response_callback("SET", lost_write_reply)
    entry: Final = MCPServer(server_id="lost-reply", name="lost reply", transport="http")
    with patch.object(litellm, "cache", SimpleNamespace(cache=cache)):
        await _cache_temporary_mcp_server_in_redis(entry, ttl_seconds=0)
        assert await raw.pttl("mcp-test:litellm:mcp:temporary_server:lost-reply") == 1000
        assert await _get_temporary_mcp_server_from_redis("lost-reply") == entry
