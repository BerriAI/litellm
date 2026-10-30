import json
from contextlib import nullcontext
from datetime import timedelta
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
from pydantic import TypeAdapter
from redis.exceptions import ConnectionError

from litellm.caching.redis_batch import active_request_redis_batch, request_redis_batch_scope
from litellm.proxy import proxy_server
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    AuthCacheInvalidationSubscriber,
    auth_cache_invalidation_channel,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.oauth2_token_cache import MCPPerUserTokenCache
from litellm.proxy.mcp_server.outbound_credentials.dual_cache_token_backend import DualCacheTokenCacheBackend
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.proxy.mcp_server.outbound_credentials.token_cache_codec import OAuthTokenCacheCodec
from litellm.redis.stores.mcp_per_user_tokens import TransportMcpPerUserTokenStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache, trip_breaker


@pytest.fixture(autouse=True)
def encryption_and_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master")
    monkeypatch.setattr("time.time", lambda: 1_800_000_000.0)


_MESSAGE: Final = TypeAdapter(dict[str, object])


def _legacy_codec(cache: MCPPerUserTokenCache) -> OAuthTokenCacheCodec:
    return cache._codec()  # pyright: ignore[reportPrivateUsage]  # the legacy cache owns the codec the new backend must match


@pytest.mark.asyncio
@pytest.mark.parametrize("proof", [None, "bound-é"])
async def test_legacy_wire_is_shared_by_both_callers_with_exact_codec_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
    proof: str | None,
) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="oauth")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    cache: Final = UserApiKeyCache(redis_cache=redis)
    legacy: Final = MCPPerUserTokenCache()
    backend: Final = DualCacheTokenCacheBackend(cache, _legacy_codec(legacy))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    user: Final = 'a:é"'
    key: Final = f"mcp:per_user_token:{user}:東京"
    token: Final = OAuthToken("bearer-é", identity_binding_proof=proof)
    plaintext: Final = (
        token.access_token
        if proof is None
        else "litellm-bound-oauth-v1:"
        + json.dumps({"access_token": token.access_token, "identity_binding_proof": proof})
    )
    ciphertext: Final = encrypt_value_helper(plaintext)
    await redis.async_set_cache(key, ciphertext, ttl=90)
    assert await backend.get(user, "東京") == token
    assert await legacy.get_token(user, "東京") == token
    await backend.set(user, "東京", token, 120.9)
    written: Final = await redis.async_get_cache(key)
    assert isinstance(written, str)
    assert decrypt_value_helper(written, "test") == plaintext
    assert await raw.get("oauth:" + key) == json.dumps(written).encode()
    assert await raw.pttl("oauth:" + key) == 120000
    assert await raw.keys("*") == [("oauth:" + key).encode()]
    cache.in_memory_cache.delete_cache(key)
    await legacy.set(user, "東京", "updated", 73, proof)
    cache.in_memory_cache.delete_cache(key)
    assert await backend.get(user, "東京") == OAuthToken("updated", identity_binding_proof=proof)
    assert await raw.pttl("oauth:" + key) == 73000
    await backend.delete(user, "東京")
    assert await legacy.get(user, "東京") is None
    assert await raw.get("oauth:" + key) is None


@pytest.mark.asyncio
async def test_auth_cache_deferred_write_joins_request_batch_and_retains_memory_policy() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="oauth")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    cache: Final = UserApiKeyCache(redis_cache=redis)
    backend: Final = DualCacheTokenCacheBackend(cache, _legacy_codec(MCPPerUserTokenCache()))
    with request_redis_batch_scope():
        await backend.set("u", "s", OAuthToken("pending"), 20.9)
        assert await backend.get("u", "s") == OAuthToken("pending")
        assert await raw.get("oauth:mcp:per_user_token:u:s") is None
        batch: Final = active_request_redis_batch(redis)
        assert batch is not None
        await batch.flush()
        assert await raw.pttl("oauth:mcp:per_user_token:u:s") == 20000
        cache.in_memory_cache.delete_cache("mcp:per_user_token:u:s")
        assert await backend.get("u", "s") == OAuthToken("pending")
        batch.delete("mcp:per_user_token:u:s")
        cache.in_memory_cache.delete_cache("mcp:per_user_token:u:s")
        assert await backend.get("u", "s") is None
        await batch.flush()
    assert await raw.get("oauth:mcp:per_user_token:u:s") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b"{}", b"null", b"42", b"broken", b"\xff", b'"litellm-bound-oauth-v1:{}"'])
async def test_malformed_redis_tokens_are_safe_misses(monkeypatch: pytest.MonkeyPatch, wire: bytes) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    cache: Final = UserApiKeyCache(redis_cache=redis)
    legacy: Final = MCPPerUserTokenCache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    await fakeredis.FakeAsyncRedis(server=server).set("mcp:per_user_token:u:s", wire)
    assert await legacy.get("u", "s") is None
    assert await DualCacheTokenCacheBackend(cache, _legacy_codec(legacy)).get("u", "s") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [False, True])
async def test_ambiguous_write_retains_local_value_without_retry_and_peer_sees_only_applied_value(
    applied: bool,
) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="oauth")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    local: Final = DualCacheTokenCacheBackend(UserApiKeyCache(redis_cache=redis), _legacy_codec(MCPPerUserTokenCache()))
    peer: Final = DualCacheTokenCacheBackend(UserApiKeyCache(redis_cache=redis), _legacy_codec(MCPPerUserTokenCache()))

    async def lost_reply(*args: object, **kwargs: object) -> None:
        if applied:
            assert isinstance(args[1], (str, bytes))
            await raw.set("oauth:mcp:per_user_token:u:s", args[1], ex=20)
        raise ConnectionError("lost reply")

    with patch.object(redis.init_async_client(), "set", lost_reply):
        await local.set("u", "s", OAuthToken("test-token"), 20)
    assert await local.get("u", "s") == OAuthToken("test-token")
    assert await peer.get("u", "s") == (OAuthToken("test-token") if applied else None)


@pytest.mark.asyncio
async def test_expiry_and_outage_preserve_local_cache_and_safe_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    cache: Final = UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=5)
    backend: Final = DualCacheTokenCacheBackend(cache, _legacy_codec(MCPPerUserTokenCache()))
    await backend.set("u", "s", OAuthToken("expires"), 2)
    monkeypatch.setattr("time.time", lambda: 1_800_000_003.0)
    assert await backend.get("u", "s") is None
    server.connected = False
    await backend.set("u", "s", OAuthToken("local"), 20)
    assert await backend.get("u", "s") == OAuthToken("local")
    await backend.delete("u", "s")
    assert await backend.get("u", "s") is None
    store: Final = TransportMcpPerUserTokenStore(redis.transport())
    assert isinstance(await store.read("u", "s"), RedisFailure)
    assert isinstance(await store.write("u", "s", "encrypted", ttl=timedelta(seconds=2)), RedisFailure)


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("ttl", [0.9, 0.99999999])
async def test_subsecond_ttl_keeps_local_value_but_does_not_replace_shared_entry(deferred: bool, ttl: float) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    cache: Final = UserApiKeyCache(redis_cache=redis)
    backend: Final = DualCacheTokenCacheBackend(cache, _legacy_codec(MCPPerUserTokenCache()))
    await backend.set("u", "s", OAuthToken("old"), 30)
    with request_redis_batch_scope() if deferred else nullcontext():
        await backend.set("u", "s", OAuthToken("local"), ttl)
        assert await backend.get("u", "s") == OAuthToken("local")
        batch: Final = active_request_redis_batch(redis)
        if batch is not None:
            await batch.flush()
        cache.in_memory_cache.delete_cache("mcp:per_user_token:u:s")
        assert await backend.get("u", "s") == OAuthToken("old")


@pytest.mark.asyncio
async def test_open_breaker_preserves_existing_auth_memory_fallback() -> None:
    redis: Final = fake_redis_cache(fakeredis.FakeServer())
    trip_breaker(redis)
    store: Final = TransportMcpPerUserTokenStore(redis.transport())
    read: Final = await store.read("u", "s")
    assert isinstance(read, RedisFailure) and read.kind == "breaker_open"
    cache: Final = UserApiKeyCache(redis_cache=redis)
    backend: Final = DualCacheTokenCacheBackend(cache, _legacy_codec(MCPPerUserTokenCache()))
    await backend.set("u", "s", OAuthToken("local"), 60)
    assert await backend.get("u", "s") == OAuthToken("local")
    await backend.delete("u", "s")
    assert await backend.get("u", "s") is None


@pytest.mark.asyncio
async def test_legacy_delete_evicts_redis_local_and_peer_through_auth_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="oauth")
    local: Final = UserApiKeyCache(redis_cache=redis)
    peer: Final = UserApiKeyCache(redis_cache=redis)
    legacy: Final = MCPPerUserTokenCache()
    backend: Final = DualCacheTokenCacheBackend(peer, _legacy_codec(legacy))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", local)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", redis)
    await legacy.set("u", "s", "revoke-me", 60)
    assert await backend.get("u", "s") == OAuthToken("revoke-me")
    pubsub: Final = fakeredis.FakeAsyncRedis(server=server).pubsub()
    await pubsub.subscribe(auth_cache_invalidation_channel(redis))
    await pubsub.get_message()
    await legacy.delete("u", "s")
    message: Final = _MESSAGE.validate_python(await pubsub.get_message(ignore_subscribe_messages=True, timeout=1))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # redis-py pubsub replies are untyped
    data: Final = message["data"]
    assert isinstance(data, (str, bytes))
    assert json.loads(data) == {"cache_key": "mcp:per_user_token:u:s"}
    AuthCacheInvalidationSubscriber(redis, peer)._apply_message(message)  # pyright: ignore[reportPrivateUsage]  # drives the subscriber without a live pubsub loop
    assert await backend.get("u", "s") is None
    assert await legacy.get("u", "s") is None
    assert await fakeredis.FakeAsyncRedis(server=server).get("oauth:mcp:per_user_token:u:s") is None
    await pubsub.aclose()  # pyright: ignore[reportAttributeAccessIssue]  # redis-py stubs predate PubSub.aclose
