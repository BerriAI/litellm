import json
from datetime import timedelta
from typing import Final

import fakeredis
import pytest

from litellm.redis.stores.mcp_per_user_token import TransportMcpPerUserTokenStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache

KEY: Final = "ns:mcp:per_user_token:alice:srv"


@pytest.mark.asyncio
async def test_blob_bytes_ttl_and_namespace_match_the_legacy_dual_cache_writer(server: fakeredis.FakeServer) -> None:
    store: Final = TransportMcpPerUserTokenStore(fake_redis_cache(server, namespace="ns").transport())
    raw: Final = fakeredis.FakeAsyncRedis(server=server)

    assert await store.put("alice", "srv", 'enc "\u96ea"', ttl=timedelta(seconds=90.9)) is True
    assert await raw.get(KEY) == json.dumps('enc "\u96ea"').encode()
    assert await raw.ttl(KEY) == 90
    assert await store.get("alice", "srv") == 'enc "\u96ea"'
    assert await store.delete("alice", "srv") == 1
    assert await store.get("alice", "srv") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("stored", "expected"), [(b"'legacy-literal'", "legacy-literal"), (b"12345", None)])
async def test_legacy_literal_and_non_string_values_decode_like_the_dual_cache(
    server: fakeredis.FakeServer, stored: bytes, expected: str | None
) -> None:
    await fakeredis.FakeAsyncRedis(server=server).set(KEY, stored)
    store: Final = TransportMcpPerUserTokenStore(fake_redis_cache(server, namespace="ns").transport())
    assert await store.get("alice", "srv") == expected


@pytest.mark.asyncio
async def test_malformed_values_are_decode_failures_and_outages_are_transport_failures(
    server: fakeredis.FakeServer,
) -> None:
    await fakeredis.FakeAsyncRedis(server=server).set(KEY, b"{not json")
    store: Final = TransportMcpPerUserTokenStore(fake_redis_cache(server, namespace="ns").transport())
    malformed: Final = await store.get("alice", "srv")
    server.connected = False
    outage: Final = await store.get("alice", "srv")

    assert isinstance(malformed, RedisFailure) and malformed.kind == "decode"
    assert isinstance(outage, RedisFailure) and outage.kind == "transport"
