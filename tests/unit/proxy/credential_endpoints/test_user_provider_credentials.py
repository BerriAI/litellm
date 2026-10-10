"""Tests for the per-user provider credentials DB/cache module."""

from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.credential_endpoints.user_provider_credentials import (
    GithubCopilotUserConnectionPayload,
    aget_user_provider_tokens,
    decode_user_provider_credential,
    delete_user_provider_credential,
    delete_user_provider_credentials_for_credential,
    drop_user_provider_credential_cache,
    get_user_provider_credential,
    invalidate_user_provider_credential_cache,
    set_user_provider_credential_cache,
    upsert_user_provider_credential,
)


def _prisma(table=None):
    prisma_client = MagicMock()
    prisma_client.db.litellm_userprovidercredentials = table or MagicMock()
    prisma_client.writer_db = prisma_client.db
    return prisma_client


def _row(user_id="user-a", credential_name="copilot-cred", credential_b64="cipher", provider="github_copilot"):
    return SimpleNamespace(
        user_id=user_id,
        credential_name=credential_name,
        credential_b64=credential_b64,
        provider=provider,
    )


def _fake_redis(monkeypatch):
    """A RedisCache backed by a fakeredis client; two DualCaches can share it."""
    import fakeredis

    from litellm.caching.redis_cache import RedisCache

    fake = fakeredis.FakeAsyncRedis()
    redis = RedisCache(host="localhost", port=6379)
    monkeypatch.setattr(redis, "init_async_client", lambda: fake)
    return redis


@pytest.fixture(autouse=True)
def _master_key(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-master")


@pytest.mark.asyncio
async def test_upsert_then_get_roundtrips_payload():
    stored = {}

    async def fake_upsert(*, where, data):
        stored["row"] = _row(
            user_id=where["user_id_credential_name"]["user_id"],
            credential_name=where["user_id_credential_name"]["credential_name"],
            credential_b64=data["create"]["credential_b64"],
        )

    async def fake_find_unique(*, where):
        return stored.get("row")

    table = MagicMock()
    table.upsert = AsyncMock(side_effect=fake_upsert)
    table.find_unique = AsyncMock(side_effect=fake_find_unique)
    prisma_client = _prisma(table)

    await upsert_user_provider_credential(
        prisma_client,
        "user-a",
        "copilot-cred",
        "github_copilot",
        GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo"),
    )
    payload = await get_user_provider_credential(prisma_client, "user-a", "copilot-cred")
    assert payload is not None
    assert payload.access_token == "gho_secret"
    assert payload.github_login == "octo"
    # ciphertext at rest, not plaintext
    assert "gho_secret" not in stored["row"].credential_b64


@pytest.mark.asyncio
async def test_delete_returns_prior_payload_and_is_idempotent():
    table = MagicMock()
    table.delete_many = AsyncMock(return_value=None)
    table.find_unique = AsyncMock(return_value=None)
    prisma_client = _prisma(table)
    assert await delete_user_provider_credential(prisma_client, "user-a", "copilot-cred") is None
    table.delete_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_for_credential_returns_affected_user_ids():
    table = MagicMock()
    table.find_many = AsyncMock(return_value=[_row(user_id="u1"), _row(user_id="u2"), _row(user_id="u1")])
    table.delete_many = AsyncMock(return_value=None)
    prisma_client = _prisma(table)
    user_ids = await delete_user_provider_credentials_for_credential(prisma_client, "copilot-cred")
    assert sorted(user_ids) == ["u1", "u2"]
    table.delete_many.assert_awaited_once_with(where={"credential_name": "copilot-cred"})


@pytest.mark.asyncio
async def test_aget_tokens_returns_plaintext_and_caches_ciphertext(monkeypatch):
    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    ciphertext = upc._encode(payload)
    table = MagicMock()
    table.find_many = AsyncMock(return_value=[_row(credential_b64=ciphertext)])
    prisma_client = _prisma(table)
    redis = _fake_redis(monkeypatch)
    cache = DualCache(redis_cache=redis)

    tokens = await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert tokens == {"copilot-cred": "gho_secret"}

    # second call served from cache: no new DB read, and cache holds ciphertext not plaintext
    tokens2 = await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert tokens2 == {"copilot-cred": "gho_secret"}
    assert table.find_many.await_count == 1
    cached = await redis.async_get_cache(upc._cache_key("user-a", "copilot-cred"))
    assert cached == ciphertext
    assert "gho_secret" not in cached
    assert await cache.in_memory_cache.async_get_cache(upc._cache_key("user-a", "copilot-cred")) is None


@pytest.mark.asyncio
async def test_aget_tokens_negative_caches_missing_connection(monkeypatch):
    table = MagicMock()
    table.find_many = AsyncMock(return_value=[])
    prisma_client = _prisma(table)
    cache = DualCache(redis_cache=_fake_redis(monkeypatch))

    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {}
    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {}
    assert table.find_many.await_count == 1


@pytest.mark.asyncio
async def test_invalidate_cache_forces_db_refetch(monkeypatch):
    payload = GithubCopilotUserConnectionPayload(access_token="gho_new", github_login="octo")
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    table = MagicMock()
    table.find_many = AsyncMock(return_value=[_row(credential_b64=upc._encode(payload))])
    prisma_client = _prisma(table)
    cache = DualCache(redis_cache=_fake_redis(monkeypatch))

    await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    await drop_user_provider_credential_cache(cache, "user-a", "copilot-cred")
    tokens = await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert tokens == {"copilot-cred": "gho_new"}
    assert table.find_many.await_count == 2


def test_decode_rejects_garbage():
    assert decode_user_provider_credential("not-a-cipher") is None


@pytest.mark.asyncio
async def test_disconnect_on_one_worker_invalidates_other_workers_via_redis(monkeypatch):
    """Two DualCache instances (two workers) sharing one Redis: a disconnect on A must
    be visible on B, not just in A's in-memory layer."""
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    redis = _fake_redis(monkeypatch)
    cache_a = DualCache(redis_cache=redis)
    cache_b = DualCache(redis_cache=redis)

    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    ciphertext = upc._encode(payload)
    table = MagicMock()
    table.find_many = AsyncMock(side_effect=[[_row(credential_b64=ciphertext)], []])
    prisma_client = _prisma(table)

    # worker A "connects" (DB read + cache write)
    tokens_a = await aget_user_provider_tokens(prisma_client, cache_a, "user-a", ["copilot-cred"])
    assert tokens_a == {"copilot-cred": "gho_secret"}

    # worker B reads the same connection straight from Redis (its in-memory layer is empty)
    tokens_b = await aget_user_provider_tokens(prisma_client, cache_b, "user-a", ["copilot-cred"])
    assert tokens_b == {"copilot-cred": "gho_secret"}

    # worker A disconnects: the key is overwritten with the not-connected tombstone
    await invalidate_user_provider_credential_cache(cache_a, "user-a", "copilot-cred")

    # worker B sees the tombstone from Redis, no worker-local staleness, no DB read needed
    tokens_b_after = await aget_user_provider_tokens(prisma_client, cache_b, "user-a", ["copilot-cred"])
    assert tokens_b_after == {}
    assert table.find_many.await_count == 1


@pytest.mark.asyncio
async def test_stale_fill_cannot_overwrite_a_disconnect_tombstone(monkeypatch):
    """A token read racing a disconnect: the read's fill uses set-if-absent, so it
    must lose to the tombstone the disconnect wrote."""
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    redis = _fake_redis(monkeypatch)
    cache = DualCache(redis_cache=redis)

    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    ciphertext = upc._encode(payload)

    async def _read_while_disconnecting(*args, **kwargs):
        # the row was fetched before the disconnect landed; the fill that follows must
        # lose to the tombstone instead of resurrecting the token
        await invalidate_user_provider_credential_cache(cache, "user-a", "copilot-cred")
        return [_row(credential_b64=ciphertext)]

    table = MagicMock()
    table.find_many = AsyncMock(side_effect=_read_while_disconnecting)
    prisma_client = _prisma(table)

    await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {}


@pytest.mark.asyncio
async def test_redis_read_failure_falls_back_to_the_database(monkeypatch):
    """A broken Redis must not 401 a connected user: reads degrade to a cache miss."""
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    redis = _fake_redis(monkeypatch)
    monkeypatch.setattr(redis, "async_get_cache", AsyncMock(side_effect=RuntimeError("redis down")))
    cache = DualCache(redis_cache=redis)

    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    table = MagicMock()
    table.find_many = AsyncMock(return_value=[_row(credential_b64=upc._encode(payload))])
    prisma_client = _prisma(table)

    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {
        "copilot-cred": "gho_secret"
    }


@pytest.mark.asyncio
async def test_redis_write_and_delete_failures_do_not_fail_the_request(monkeypatch):
    redis = _fake_redis(monkeypatch)
    monkeypatch.setattr(redis, "async_set_cache", AsyncMock(side_effect=RuntimeError("redis down")))
    monkeypatch.setattr(redis, "async_delete_cache", AsyncMock(side_effect=RuntimeError("redis down")))
    cache = DualCache(redis_cache=redis)

    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    table = MagicMock()
    table.find_many = AsyncMock(return_value=[_row(credential_b64=upc._encode(payload))])
    prisma_client = _prisma(table)

    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {
        "copilot-cred": "gho_secret"
    }
    await invalidate_user_provider_credential_cache(cache, "user-a", "copilot-cred")


@pytest.mark.asyncio
async def test_without_redis_no_worker_local_entries_are_kept():
    """No Redis -> no cache at all: a disconnect on another worker cannot leave a
    stale local entry, because none was ever written."""
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    ciphertext = upc._encode(payload)
    table = MagicMock()
    table.find_many = AsyncMock(side_effect=[[_row(credential_b64=ciphertext)], [_row(credential_b64=ciphertext)], []])
    prisma_client = _prisma(table)
    cache_a = DualCache()
    cache_b = DualCache()

    assert await aget_user_provider_tokens(prisma_client, cache_a, "user-a", ["copilot-cred"]) == {
        "copilot-cred": "gho_secret"
    }
    assert await aget_user_provider_tokens(prisma_client, cache_b, "user-a", ["copilot-cred"]) == {
        "copilot-cred": "gho_secret"
    }
    await invalidate_user_provider_credential_cache(cache_b, "user-a", "copilot-cred")
    assert await aget_user_provider_tokens(prisma_client, cache_a, "user-a", ["copilot-cred"]) == {}
    assert table.find_many.await_count == 3


@pytest.mark.asyncio
async def test_connect_overwrites_a_not_connected_tombstone(monkeypatch):
    """Connect writes the new connection over any cached tombstone so the next
    read sees the token immediately, no TTL wait."""
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    redis = _fake_redis(monkeypatch)
    cache = DualCache(redis_cache=redis)
    await invalidate_user_provider_credential_cache(cache, "user-a", "copilot-cred")

    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    assert await set_user_provider_credential_cache(cache, "user-a", "copilot-cred", payload)

    table = MagicMock()
    table.find_many = AsyncMock(return_value=[])
    prisma_client = _prisma(table)
    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {
        "copilot-cred": "gho_secret"
    }
    table.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_not_connected_fill_cannot_overwrite_a_connect_write(monkeypatch):
    """A read that fetched "not connected" before the poll saved the row fills
    with set-if-absent; the connect's plain overwrite must win so the marker
    never lingers for the TTL."""
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    redis = _fake_redis(monkeypatch)
    cache = DualCache(redis_cache=redis)

    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")

    async def _read_while_connecting(*args, **kwargs):
        assert await set_user_provider_credential_cache(cache, "user-a", "copilot-cred", payload)
        return []

    table = MagicMock()
    table.find_many = AsyncMock(side_effect=_read_while_connecting)
    prisma_client = _prisma(table)

    await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {
        "copilot-cred": "gho_secret"
    }


@pytest.mark.asyncio
async def test_set_reports_connected_without_redis():
    """No Redis configured: there is no stale entry to defend against, so the
    cache refresh succeeds by definition."""
    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    assert await set_user_provider_credential_cache(DualCache(), "user-a", "copilot-cred", payload)


@pytest.mark.asyncio
async def test_set_reports_connected_when_delete_clears_a_dropped_write():
    """async_set_cache swallows errors: a set that reports success but writes
    nothing is verified by read-back, and a working delete still resolves the
    stale entry."""
    store: dict = {}
    silent_redis = SimpleNamespace(
        async_set_cache=AsyncMock(return_value=None),
        async_get_cache=AsyncMock(side_effect=lambda key: store.get(key)),
        async_delete_cache=AsyncMock(side_effect=lambda key: store.pop(key, None)),
    )
    cache = DualCache(redis_cache=silent_redis)
    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    assert await set_user_provider_credential_cache(cache, "user-a", "copilot-cred", payload)


@pytest.mark.asyncio
async def test_set_reports_failure_when_a_stale_entry_survives_both_attempts():
    """A write silently dropped while a not-connected marker stays cached means
    the caller cannot trust the connect: the helper reports failure instead of
    claiming success."""
    stale = "stale-value"
    silent_redis = SimpleNamespace(
        async_set_cache=AsyncMock(return_value=None),
        async_get_cache=AsyncMock(return_value=stale),
        async_delete_cache=AsyncMock(return_value=None),
    )
    cache = DualCache(redis_cache=silent_redis)
    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    assert not await set_user_provider_credential_cache(cache, "user-a", "copilot-cred", payload)


@pytest.mark.asyncio
async def test_reads_hit_the_writer_engine_not_a_stale_replica():
    """With DATABASE_URL_READ_REPLICA set, prisma_client.db is the reader. A row
    saved by connect exists only on the writer, so every lookup must route to
    writer_db or a fresh connection 401s as not-connected."""
    reader_table: Final = MagicMock()
    reader_table.find_many = AsyncMock(return_value=[])
    writer_table: Final = MagicMock()
    writer_table.find_many = AsyncMock(return_value=[])
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_userprovidercredentials = reader_table
    prisma_client.writer_db.litellm_userprovidercredentials = writer_table

    await aget_user_provider_tokens(prisma_client, DualCache(), "user-a", ["copilot-cred"])

    writer_table.find_many.assert_awaited_once()
    reader_table.find_many.assert_not_awaited()
