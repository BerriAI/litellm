"""Tests for the per-user provider credentials DB/cache module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.credential_endpoints.user_provider_credentials import (
    GithubCopilotUserConnectionPayload,
    aget_user_provider_tokens,
    decode_user_provider_credential,
    delete_user_provider_credential,
    delete_user_provider_credentials_for_credential,
    get_user_provider_credential,
    invalidate_user_provider_credential_cache,
    upsert_user_provider_credential,
)


def _prisma(table=None):
    prisma_client = MagicMock()
    prisma_client.db.litellm_userprovidercredentials = table or MagicMock()
    return prisma_client


def _row(user_id="user-a", credential_name="copilot-cred", credential_b64="cipher", provider="github_copilot"):
    return SimpleNamespace(
        user_id=user_id,
        credential_name=credential_name,
        credential_b64=credential_b64,
        provider=provider,
    )


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
async def test_aget_tokens_returns_plaintext_and_caches_ciphertext():
    payload = GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login="octo")
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    ciphertext = upc._encode(payload)
    table = MagicMock()
    table.find_many = AsyncMock(return_value=[_row(credential_b64=ciphertext)])
    prisma_client = _prisma(table)
    cache = DualCache()

    tokens = await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert tokens == {"copilot-cred": "gho_secret"}

    # second call served from cache: no new DB read, and cache holds ciphertext not plaintext
    tokens2 = await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert tokens2 == {"copilot-cred": "gho_secret"}
    assert table.find_many.await_count == 1
    cached = await cache.async_get_cache(upc._cache_key("user-a", "copilot-cred"))
    assert cached == ciphertext
    assert "gho_secret" not in cached


@pytest.mark.asyncio
async def test_aget_tokens_negative_caches_missing_connection():
    table = MagicMock()
    table.find_many = AsyncMock(return_value=[])
    prisma_client = _prisma(table)
    cache = DualCache()

    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {}
    assert await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"]) == {}
    assert table.find_many.await_count == 1


@pytest.mark.asyncio
async def test_invalidate_cache_forces_db_refetch():
    payload = GithubCopilotUserConnectionPayload(access_token="gho_new", github_login="octo")
    from litellm.proxy.credential_endpoints import user_provider_credentials as upc

    table = MagicMock()
    table.find_many = AsyncMock(return_value=[_row(credential_b64=upc._encode(payload))])
    prisma_client = _prisma(table)
    cache = DualCache()

    await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    await invalidate_user_provider_credential_cache(cache, "user-a", "copilot-cred")
    tokens = await aget_user_provider_tokens(prisma_client, cache, "user-a", ["copilot-cred"])
    assert tokens == {"copilot-cred": "gho_new"}
    assert table.find_many.await_count == 2


def test_decode_rejects_garbage():
    assert decode_user_provider_credential("not-a-cipher") is None
