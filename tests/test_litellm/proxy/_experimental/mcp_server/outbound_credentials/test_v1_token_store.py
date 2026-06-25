"""Tests for the v1-backed per-user OAuth token source (V1PerUserTokenStore)."""

from unittest.mock import AsyncMock, patch

from litellm.proxy._experimental.mcp_server.outbound_credentials.v1_token_store import (
    V1PerUserTokenStore,
)

_RESOLVE = "litellm.proxy._experimental.mcp_server.db.resolve_user_oauth_access_token"


def _store_for(server: object) -> V1PerUserTokenStore:
    return V1PerUserTokenStore(server_lookup=lambda _server_id: server)


async def test_wraps_the_resolved_access_token():
    with patch(_RESOLVE, new=AsyncMock(return_value="at-123")):
        token = await _store_for(object()).fetch("alice", "s")
    assert token is not None and token.access_token == "at-123"


async def test_missing_token_is_none():
    with patch(_RESOLVE, new=AsyncMock(return_value=None)):
        assert await _store_for(object()).fetch("alice", "s") is None


async def test_empty_user_short_circuits_without_resolving():
    resolve = AsyncMock(return_value="at")
    with patch(_RESOLVE, new=resolve):
        assert await _store_for(object()).fetch("", "s") is None
    resolve.assert_not_called()


async def test_unknown_server_is_none_without_resolving():
    resolve = AsyncMock(return_value="at")
    store = V1PerUserTokenStore(server_lookup=lambda _server_id: None)
    with patch(_RESOLVE, new=resolve):
        assert await store.fetch("alice", "missing") is None
    resolve.assert_not_called()


async def test_passes_user_and_resolved_server_to_the_core():
    server = object()
    resolve = AsyncMock(return_value="at")
    with patch(_RESOLVE, new=resolve):
        await _store_for(server).fetch("alice", "srv-1")
    resolve.assert_awaited_once_with("alice", server)
