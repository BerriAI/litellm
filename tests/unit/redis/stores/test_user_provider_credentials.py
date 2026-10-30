import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import pytest
from redis.exceptions import ConnectionError, ResponseError

from litellm.caching.dual_cache import DualCache
from litellm.constants import GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.credential_endpoints.user_provider_credentials import (
    GithubCopilotUserConnectionPayload,
    aget_user_provider_credentials,
    decode_user_provider_credential,
    drop_user_provider_credential_cache,
    invalidate_user_provider_credential_cache,
    set_user_provider_credential_cache,
)
from litellm.redis.stores.user_provider_credentials import InvalidCredentialValue, TransportUserProviderCredentialStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-master")
    monkeypatch.setattr("time.time", lambda: 1_800_000_000.0)


@pytest.mark.asyncio
async def test_old_and_new_connections_share_ciphertext_key_tombstones_and_ttl() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="credentials")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    cache: Final = DualCache(redis_cache=redis)
    user: Final = 'user:é"'
    name: Final = "provider:東京"
    key: Final = "user_provider_credential:" + json.dumps([user, name], separators=(",", ":"))
    payload: Final = GithubCopilotUserConnectionPayload(access_token="test-token-é", github_login="octo")
    ciphertext: Final = encrypt_value_helper(payload.model_dump_json())
    prisma: Final = MagicMock()
    await redis.async_set_cache(key, ciphertext, ttl=60)
    assert await raw.get("credentials:" + key) == json.dumps(ciphertext).encode()
    assert await aget_user_provider_credentials(prisma, cache, user, [name]) == {
        name: {"access_token": payload.access_token, "github_host": "github.com"}
    }
    assert await invalidate_user_provider_credential_cache(cache, user, name)
    assert await raw.get("credentials:" + key) == b'"__not_connected__"'
    assert await redis.async_get_cache(key) == "__not_connected__"
    assert await aget_user_provider_credentials(prisma, cache, user, [name]) == {}
    assert await set_user_provider_credential_cache(cache, user, name, payload)
    old_read: Final = await redis.async_get_cache(key)
    assert decode_user_provider_credential(old_read) == payload
    assert await raw.get("credentials:" + key) == json.dumps(old_read).encode()
    assert await raw.pttl("credentials:" + key) == GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS * 1000
    assert await raw.keys("*") == [("credentials:" + key).encode()]
    assert await cache.in_memory_cache.async_get_cache(key) is None
    await drop_user_provider_credential_cache(cache, user, name)
    assert await raw.get("credentials:" + key) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [False, True])
async def test_ambiguous_tombstone_requires_readback(applied: bool) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="credentials")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    cache: Final = DualCache(redis_cache=redis)
    key: Final = 'credentials:user_provider_credential:["u","c"]'

    async def lost_reply(*args: object, **kwargs: object) -> None:
        if applied:
            await raw.set(key, b'"__not_connected__"', ex=60)
        raise ConnectionError("lost reply")

    with patch.object(redis.init_async_client(), "set", lost_reply):
        assert await invalidate_user_provider_credential_cache(cache, "u", "c") is applied
    assert await raw.get(key) == (b'"__not_connected__"' if applied else None)


@pytest.mark.asyncio
async def test_ambiguous_connection_write_is_read_before_delete() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    payload: Final = GithubCopilotUserConnectionPayload(access_token="test-token", github_login="octo")
    key: Final = 'user_provider_credential:["u","c"]'
    ciphertext: Final = encrypt_value_helper(payload.model_dump_json())

    async def lost_reply(*args: object, **kwargs: object) -> None:
        await raw.set(key, json.dumps(ciphertext), ex=60)
        raise ConnectionError("lost reply")

    with patch.object(redis.init_async_client(), "set", lost_reply):
        assert await set_user_provider_credential_cache(DualCache(redis_cache=redis), "u", "c", payload)
    assert await raw.get(key) == json.dumps(ciphertext).encode()


@pytest.mark.asyncio
async def test_outage_uses_database_but_refuses_disconnect_and_keeps_connect_fallback() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    payload: Final = GithubCopilotUserConnectionPayload(access_token="test-token", github_login="octo")
    prisma: Final = MagicMock()
    prisma.writer_db.litellm_userprovidercredentials.find_many = AsyncMock(
        return_value=[SimpleNamespace(credential_name="c", credential_b64=encrypt_value_helper(payload.model_dump_json()))]
    )
    server.connected = False
    assert await aget_user_provider_credentials(prisma, cache, "u", ["c"]) == {
        "c": {"access_token": "test-token", "github_host": "github.com"}
    }
    assert not await invalidate_user_provider_credential_cache(cache, "u", "c")
    assert await set_user_provider_credential_cache(cache, "u", "c", payload)
    store: Final = TransportUserProviderCredentialStore(redis.transport())
    assert isinstance(await store.read("u", "c"), RedisFailure)
    redis._circuit_breaker.record_failure()
    redis._circuit_breaker.record_failure()
    redis._circuit_breaker.record_failure()
    result: Final = await store.read("u", "c")
    assert isinstance(result, RedisFailure) and result.kind == "breaker_open"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [b"garbage", b"\xff"])
async def test_malformed_values_are_typed_failures_and_caller_database_misses(value: bytes) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set('user_provider_credential:["u","c"]', value)
    store: Final = TransportUserProviderCredentialStore(redis.transport())
    result: Final = await store.read("u", "c")
    assert isinstance(result, RedisFailure) and result.kind == "decode"
    prisma: Final = MagicMock()
    prisma.writer_db.litellm_userprovidercredentials.find_many = AsyncMock(return_value=[])
    assert await aget_user_provider_credentials(prisma, DualCache(redis_cache=redis), "u", ["c"]) == {}
    assert prisma.writer_db.litellm_userprovidercredentials.find_many.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [b"{}", b"42", b"[]", b"false"])
async def test_non_string_surviving_failed_write_and_delete_is_not_a_confirmed_connect(value: bytes) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    key: Final = 'user_provider_credential:["u","c"]'
    await raw.set(key, value)
    store: Final = TransportUserProviderCredentialStore(redis.transport())
    assert isinstance(await store.read("u", "c"), InvalidCredentialValue)
    payload: Final = GithubCopilotUserConnectionPayload(access_token="test-token", github_login="octo")
    with (
        patch.object(redis.init_async_client(), "set", AsyncMock(side_effect=ResponseError("readonly"))),
        patch.object(redis.init_async_client(), "delete", AsyncMock(side_effect=ResponseError("readonly"))),
    ):
        assert not await set_user_provider_credential_cache(DualCache(redis_cache=redis), "u", "c", payload)
    assert await raw.get(key) == value


@pytest.mark.asyncio
async def test_tombstone_expires_and_reloads_from_database(monkeypatch: pytest.MonkeyPatch) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    payload: Final = GithubCopilotUserConnectionPayload(access_token="test-token", github_login="octo")
    prisma: Final = MagicMock()
    prisma.writer_db.litellm_userprovidercredentials.find_many = AsyncMock(
        return_value=[SimpleNamespace(credential_name="c", credential_b64=encrypt_value_helper(payload.model_dump_json()))]
    )
    assert await invalidate_user_provider_credential_cache(cache, "u", "c")
    assert await aget_user_provider_credentials(prisma, cache, "u", ["c"]) == {}
    monkeypatch.setattr("time.time", lambda: 1_800_000_061.0)
    assert await aget_user_provider_credentials(prisma, cache, "u", ["c"]) == {
        "c": {"access_token": "test-token", "github_host": "github.com"}
    }


@pytest.mark.asyncio
async def test_fill_nx_retains_tombstone_and_server_errors_are_values() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    store: Final = TransportUserProviderCredentialStore(redis.transport())
    assert await invalidate_user_provider_credential_cache(cache, "u", "c")
    assert await store.fill("u", "c", "stale-ciphertext", ttl=timedelta(seconds=60)) is False
    assert await store.read("u", "c") == "__not_connected__"
    with patch.object(redis.init_async_client(), "set", AsyncMock(side_effect=ResponseError("readonly"))):
        result: Final = await store.disconnect("u", "c", ttl=timedelta(seconds=60))
    assert isinstance(result, RedisFailure) and result.kind == "server"


@pytest.mark.asyncio
async def test_in_flight_database_fill_cannot_resurrect_peer_disconnect() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server)
    reader: Final = DualCache(redis_cache=redis)
    peer: Final = DualCache(redis_cache=redis)
    payload: Final = GithubCopilotUserConnectionPayload(access_token="stale-token", github_login="octo")

    async def database_read(**kwargs: object) -> list[SimpleNamespace]:
        assert await invalidate_user_provider_credential_cache(peer, "u", "c")
        return [SimpleNamespace(credential_name="c", credential_b64=encrypt_value_helper(payload.model_dump_json()))]

    prisma: Final = MagicMock()
    prisma.writer_db.litellm_userprovidercredentials.find_many = database_read
    assert await aget_user_provider_credentials(prisma, reader, "u", ["c"]) == {
        "c": {"access_token": "stale-token", "github_host": "github.com"}
    }
    assert await aget_user_provider_credentials(prisma, reader, "u", ["c"]) == {}
    assert await fakeredis.FakeAsyncRedis(server=server).get('user_provider_credential:["u","c"]') == b'"__not_connected__"'
