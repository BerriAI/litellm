"""Tests for the v2-native per-user token read store: validate the decoded blob into a typed token."""

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.v2_token_store import (
    V2PerUserTokenStore,
)


def _reader(payload):
    async def read(user_id: str, server_id: str):
        return payload

    return read


@pytest.mark.asyncio
async def test_builds_typed_token_with_epoch_expiry():
    store = V2PerUserTokenStore(
        _reader(
            {
                "access_token": "at",
                "expires_at": "2099-01-01T00:00:00+00:00",
                "refresh_token": "rt",
            }
        )
    )
    token = await store.fetch("alice", "srv")
    assert token is not None
    assert token.access_token == "at"
    assert token.refresh_token == "rt"
    # ISO string is converted to epoch seconds for RefreshingTokenStore to compare against.
    assert token.expires_at == 4070908800.0


@pytest.mark.asyncio
async def test_optional_fields_absent_yield_none():
    store = V2PerUserTokenStore(_reader({"access_token": "at"}))
    token = await store.fetch("alice", "srv")
    assert token is not None
    assert token.expires_at is None and token.refresh_token is None


@pytest.mark.asyncio
async def test_unparseable_expiry_is_dropped_not_raised():
    store = V2PerUserTokenStore(
        _reader({"access_token": "at", "expires_at": "not-a-date"})
    )
    token = await store.fetch("alice", "srv")
    assert token is not None and token.expires_at is None


@pytest.mark.asyncio
async def test_missing_access_token_is_not_authorized():
    store = V2PerUserTokenStore(_reader({"expires_at": "2099-01-01T00:00:00+00:00"}))
    assert await store.fetch("alice", "srv") is None


@pytest.mark.asyncio
async def test_no_credential_is_not_authorized():
    assert await V2PerUserTokenStore(_reader(None)).fetch("alice", "srv") is None


@pytest.mark.asyncio
async def test_empty_user_short_circuits_without_reading():
    calls = []

    async def read(user_id: str, server_id: str):
        calls.append((user_id, server_id))
        return {"access_token": "at"}

    assert await V2PerUserTokenStore(read).fetch("", "srv") is None
    assert calls == []
