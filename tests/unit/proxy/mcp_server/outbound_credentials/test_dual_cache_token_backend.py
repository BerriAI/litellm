"""The shared per-user token backend: encrypted round trip, legacy wire bytes, TTL, misses and outages."""

import json

import fakeredis
import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.outbound_credentials.dual_cache_token_backend import DualCacheTokenCacheBackend
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.proxy.mcp_server.outbound_credentials.token_cache_codec import OAuthTokenCacheCodec
from tests._support.fake_redis import fake_redis_cache

KEY = "mcp:per_user_token:alice:srv"
CODEC = OAuthTokenCacheCodec(encrypt=lambda s: f"enc:{s}", decrypt=lambda b: b[4:] if b.startswith("enc:") else None)


def _worker(server: fakeredis.FakeServer, namespace: str | None = None) -> UserApiKeyCache:
    return UserApiKeyCache(in_memory_cache=InMemoryCache(), redis_cache=fake_redis_cache(server, namespace=namespace))


def _backend(cache: UserApiKeyCache, codec: OAuthTokenCacheCodec = CODEC) -> DualCacheTokenCacheBackend:
    return DualCacheTokenCacheBackend(cache, codec)


@pytest.mark.asyncio
async def test_a_token_one_worker_writes_reaches_another_with_the_legacy_wire_bytes() -> None:
    server = fakeredis.FakeServer()
    raw = fakeredis.FakeAsyncRedis(server=server)
    await _backend(_worker(server, "ns")).set("alice", "srv", OAuthToken(access_token="at"), 120.9)

    assert await raw.get(f"ns:{KEY}") == json.dumps("enc:at").encode()
    assert await raw.ttl(f"ns:{KEY}") == 120
    got = await _backend(_worker(server, "ns")).get("alice", "srv")
    assert got is not None and got.access_token == "at"


@pytest.mark.asyncio
async def test_a_token_an_old_pod_wrote_is_read_and_cached_locally() -> None:
    server = fakeredis.FakeServer()
    raw = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(KEY, json.dumps("enc:old"), ex=60)
    reader = _worker(server)

    got = await _backend(reader).get("alice", "srv")
    await raw.delete(KEY)
    cached = await _backend(reader).get("alice", "srv")

    assert got is not None and got.access_token == "old"
    assert cached is not None and cached.access_token == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [b"12345", b'"not-our-ciphertext"', b"{not json"])
async def test_wrong_type_undecryptable_or_malformed_entries_are_misses(stored: bytes) -> None:
    server = fakeredis.FakeServer()
    await fakeredis.FakeAsyncRedis(server=server).set(KEY, stored)
    assert await _backend(_worker(server)).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_non_positive_ttl_is_not_written() -> None:
    server = fakeredis.FakeServer()
    await _backend(_worker(server)).set("alice", "srv", OAuthToken(access_token="at"), 0.0)
    assert await fakeredis.FakeAsyncRedis(server=server).keys("*") == []


@pytest.mark.asyncio
async def test_delete_removes_the_entry_here_and_in_redis() -> None:
    server = fakeredis.FakeServer()
    backend = _backend(_worker(server))
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)
    await backend.delete("alice", "srv")
    assert await backend.get("alice", "srv") is None
    assert await fakeredis.FakeAsyncRedis(server=server).exists(KEY) == 0


@pytest.mark.asyncio
async def test_keys_isolate_users_and_servers() -> None:
    server = fakeredis.FakeServer()
    backend = _backend(_worker(server))
    await backend.set("alice", "srv", OAuthToken(access_token="a"), 60.0)
    await backend.set("bob", "srv", OAuthToken(access_token="b"), 60.0)
    alice = await _backend(_worker(server)).get("alice", "srv")
    bob = await _backend(_worker(server)).get("bob", "srv")
    assert alice is not None and alice.access_token == "a"
    assert bob is not None and bob.access_token == "b"


@pytest.mark.asyncio
async def test_a_redis_outage_degrades_every_operation_to_its_safe_value() -> None:
    server = fakeredis.FakeServer()
    backend = _backend(_worker(server))
    server.connected = False
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)
    await backend.delete("alice", "srv")
    assert await _backend(_worker(server)).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_codec_failures_degrade_to_a_miss_or_a_no_op() -> None:
    def _boom(_: str) -> str:
        raise ValueError("codec unavailable")

    server = fakeredis.FakeServer()
    await _backend(_worker(server), OAuthTokenCacheCodec(encrypt=_boom, decrypt=_boom)).set(
        "alice", "srv", OAuthToken(access_token="at"), 60.0
    )
    await fakeredis.FakeAsyncRedis(server=server).set(KEY, json.dumps("whatever"))
    assert (
        await _backend(_worker(server), OAuthTokenCacheCodec(encrypt=_boom, decrypt=_boom)).get("alice", "srv") is None
    )
