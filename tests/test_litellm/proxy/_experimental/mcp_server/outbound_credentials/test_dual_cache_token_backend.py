"""Tests for the DualCache-backed token cache backend: encrypted round-trip, key, TTL, miss."""

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.dual_cache_token_backend import (
    DualCacheTokenCacheBackend,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)


class _FakeCache:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    async def async_get_cache(self, key):
        return self.values.get(key)

    async def async_set_cache(self, key, value, ttl=None):
        self.values[key] = value
        self.ttls[key] = ttl

    async def async_delete_cache(self, key):
        self.values.pop(key, None)


def _backend(cache):
    codec = OAuthTokenCacheCodec(
        encrypt=lambda s: f"enc:{s}",
        decrypt=lambda b: b[4:] if b.startswith("enc:") else None,
    )
    return DualCacheTokenCacheBackend(cache, codec)


@pytest.mark.asyncio
async def test_set_then_get_round_trips_encrypted_under_the_per_user_key():
    cache = _FakeCache()
    backend = _backend(cache)
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 120.0)

    key = "mcp:per_user_token:alice:srv"
    assert cache.ttls[key] == 120.0
    # Stored via the codec (encrypted), not the bare token; the codec's own test proves real
    # NaCl output hides the secret - here the fake encrypt just wraps, so we check it was applied.
    assert cache.values[key] == "enc:at"

    got = await backend.get("alice", "srv")
    assert got is not None and got.access_token == "at"


@pytest.mark.asyncio
async def test_get_missing_key_is_none():
    assert await _backend(_FakeCache()).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_non_str_cache_value_is_a_miss():
    cache = _FakeCache()
    cache.values["mcp:per_user_token:alice:srv"] = 12345  # corrupt / wrong type
    assert await _backend(cache).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_undecryptable_blob_is_a_miss():
    # e.g. a master-key rotation leaves an entry the codec can't decrypt; it must read as a miss so
    # the store re-reads the DB, not raise (the decrypt helper here returns None for unknown blobs).
    cache = _FakeCache()
    cache.values["mcp:per_user_token:alice:srv"] = "not-our-ciphertext"
    assert await _backend(cache).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_non_positive_ttl_is_not_written():
    cache = _FakeCache()
    await _backend(cache).set("alice", "srv", OAuthToken(access_token="at"), 0.0)
    assert cache.values == {}  # an already-expired token is not cached


@pytest.mark.asyncio
async def test_delete_removes_the_entry():
    cache = _FakeCache()
    backend = _backend(cache)
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)
    await backend.delete("alice", "srv")
    assert await backend.get("alice", "srv") is None


@pytest.mark.asyncio
async def test_keys_isolate_users_and_servers():
    cache = _FakeCache()
    backend = _backend(cache)
    await backend.set("alice", "srv", OAuthToken(access_token="a"), 60.0)
    await backend.set("bob", "srv", OAuthToken(access_token="b"), 60.0)
    alice = await backend.get("alice", "srv")
    bob = await backend.get("bob", "srv")
    assert alice is not None and alice.access_token == "a"
    assert bob is not None and bob.access_token == "b"


class _RaisingCache(_FakeCache):
    """A cache whose every op raises, e.g. a Redis outage tripping the circuit breaker."""

    async def async_get_cache(self, key):
        raise ConnectionError("redis down")

    async def async_set_cache(self, key, value, ttl=None):
        raise ConnectionError("redis down")

    async def async_delete_cache(self, key):
        raise ConnectionError("redis down")


# A cache or codec failure must degrade to the safe value (miss / no-op), never propagate: otherwise a
# Redis outage turns CachedOAuthTokenStore.fetch() into a 500 instead of a cache miss that re-reads the
# DB (get/set) or issues the OAuth challenge (the unauthorized branch deletes before returning None).
@pytest.mark.asyncio
async def test_get_is_a_miss_when_the_cache_raises():
    assert await _backend(_RaisingCache()).get("alice", "srv") is None


@pytest.mark.asyncio
async def test_set_is_swallowed_when_the_cache_raises():
    await _backend(_RaisingCache()).set("alice", "srv", OAuthToken(access_token="at"), 60.0)


@pytest.mark.asyncio
async def test_delete_is_swallowed_when_the_cache_raises():
    await _backend(_RaisingCache()).delete("alice", "srv")


@pytest.mark.asyncio
async def test_set_is_swallowed_when_the_codec_raises():
    def _boom(_: str) -> str:
        raise ValueError("encrypt unavailable")

    codec = OAuthTokenCacheCodec(encrypt=_boom, decrypt=lambda b: None)
    backend = DualCacheTokenCacheBackend(_FakeCache(), codec)
    await backend.set("alice", "srv", OAuthToken(access_token="at"), 60.0)  # must not raise


@pytest.mark.asyncio
async def test_get_is_a_miss_when_decrypt_raises():
    # A blob the decrypt rejects with an exception (e.g. bad ciphertext after key rotation) must read
    # as a miss, not propagate the error out of get() and abort fetch().
    def _boom(_: str) -> str | None:
        raise ValueError("bad ciphertext")

    codec = OAuthTokenCacheCodec(encrypt=lambda s: f"enc:{s}", decrypt=_boom)
    cache = _FakeCache()
    cache.values["mcp:per_user_token:alice:srv"] = "whatever"
    assert await DualCacheTokenCacheBackend(cache, codec).get("alice", "srv") is None
