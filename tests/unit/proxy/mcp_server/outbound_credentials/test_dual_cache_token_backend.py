"""Tests for the DualCache-backed token cache backend: encrypted round-trip, key, TTL, miss."""

from typing import Final

import fakeredis
import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.mcp_server.outbound_credentials.dual_cache_token_backend import (
    DualCacheTokenCacheBackend,
)
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)
from tests._support.fake_redis import fake_redis_cache


def _cache() -> DualCache:
    return DualCache(redis_cache=fake_redis_cache(fakeredis.FakeServer()))


def _raw(cache: DualCache) -> fakeredis.FakeAsyncRedis:
    assert cache.redis_cache is not None
    client: Final = cache.redis_cache.init_async_client()
    assert isinstance(client, fakeredis.FakeAsyncRedis)
    return client


def _backend(cache: DualCache) -> DualCacheTokenCacheBackend:
    codec = OAuthTokenCacheCodec(
        encrypt=lambda s: f"enc:{s}",
        decrypt=lambda b: b[4:] if b.startswith("enc:") else None,
    )
    return DualCacheTokenCacheBackend(cache, codec)


@pytest.mark.asyncio
async def test_set_then_get_round_trips_encrypted_under_the_per_user_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.time", lambda: 1_800_000_000.0)
    cache = _cache()
    backend = _backend(cache)
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 120.0)

    key = "mcp:per_user_token:alice:srv"
    assert await _raw(cache).pttl(key) == 120000
    # Stored via the codec (encrypted), not the bare token; the codec's own test proves real
    # NaCl output hides the secret - here the fake encrypt just wraps, so we check it was applied.
    assert await _raw(cache).get(key) == b'"enc:at"'

    got = await backend.get("alice", "srv")
    assert got is not None and got.access_token == "at"


@pytest.mark.asyncio
async def test_get_missing_key_is_none():
    assert await _backend(_cache()).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_non_str_cache_value_is_a_miss():
    cache = _cache()
    cache.in_memory_cache.set_cache("mcp:per_user_token:alice:srv", 12345)
    assert await _backend(cache).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_undecryptable_blob_is_a_miss():
    # e.g. a master-key rotation leaves an entry the codec can't decrypt; it must read as a miss so
    # the store re-reads the DB, not raise (the decrypt helper here returns None for unknown blobs).
    cache = _cache()
    cache.in_memory_cache.set_cache("mcp:per_user_token:alice:srv", "not-our-ciphertext")
    assert await _backend(cache).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_non_positive_ttl_is_not_written():
    cache = _cache()
    await _backend(cache).set("alice", "srv", OAuthToken(access_token="at"), 0.0)
    assert await cache.async_get_cache("mcp:per_user_token:alice:srv") is None


@pytest.mark.asyncio
async def test_delete_removes_the_entry():
    cache = _cache()
    backend = _backend(cache)
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)
    await backend.delete("alice", "srv")
    assert await backend.get("alice", "srv") is None


@pytest.mark.asyncio
async def test_keys_isolate_users_and_servers():
    cache = _cache()
    backend = _backend(cache)
    await backend.set("alice", "srv", OAuthToken(access_token="a"), 60.0)
    await backend.set("bob", "srv", OAuthToken(access_token="b"), 60.0)
    alice = await backend.get("alice", "srv")
    bob = await backend.get("bob", "srv")
    assert alice is not None and alice.access_token == "a"
    assert bob is not None and bob.access_token == "b"


def _offline_cache():
    server = fakeredis.FakeServer()
    server.connected = False
    return DualCache(redis_cache=fake_redis_cache(server))


@pytest.mark.asyncio
async def test_get_is_a_miss_when_the_cache_raises():
    assert await _backend(_offline_cache()).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_set_is_swallowed_when_the_cache_raises():
    backend = _backend(_offline_cache())
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)
    assert await backend.get("alice", "srv") == OAuthToken(access_token="at")


@pytest.mark.asyncio
async def test_delete_is_swallowed_when_the_cache_raises():
    backend = _backend(_offline_cache())
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)
    await backend.delete("alice", "srv")
    assert await backend.get("alice", "srv") is None


@pytest.mark.asyncio
async def test_set_is_swallowed_when_the_codec_raises():
    def _boom(_: str) -> str:
        raise ValueError("encrypt unavailable")

    codec = OAuthTokenCacheCodec(encrypt=_boom, decrypt=lambda b: None)
    backend = DualCacheTokenCacheBackend(_cache(), codec)
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)
    assert await backend.get("alice", "srv") is None


@pytest.mark.asyncio
async def test_get_is_a_miss_when_decrypt_raises():
    # A blob the decrypt rejects with an exception (e.g. bad ciphertext after key rotation) must read
    # as a miss, not propagate the error out of get() and abort fetch().
    def _boom(_: str) -> str | None:
        raise ValueError("bad ciphertext")

    codec = OAuthTokenCacheCodec(encrypt=lambda s: f"enc:{s}", decrypt=_boom)
    cache = _cache()
    cache.in_memory_cache.set_cache("mcp:per_user_token:alice:srv", "whatever")
    assert await DualCacheTokenCacheBackend(cache, codec).get("alice", "srv") is None
