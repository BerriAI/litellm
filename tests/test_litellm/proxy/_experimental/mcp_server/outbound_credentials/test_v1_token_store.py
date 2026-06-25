"""Tests for the v1-backed per-user OAuth token source (V1PerUserTokenStore)."""

from unittest.mock import AsyncMock, patch

from litellm.proxy._experimental.mcp_server.outbound_credentials.v1_token_store import (
    V1PerUserTokenStore,
)

_GET = (
    "litellm.proxy._experimental.mcp_server.oauth2_token_cache."
    "mcp_per_user_token_cache.get"
)


async def test_wraps_the_v1_access_token():
    with patch(_GET, new=AsyncMock(return_value="at-123")):
        token = await V1PerUserTokenStore().fetch("alice", "s")
    assert token is not None and token.access_token == "at-123"


async def test_missing_token_is_none():
    with patch(_GET, new=AsyncMock(return_value=None)):
        assert await V1PerUserTokenStore().fetch("alice", "s") is None


async def test_empty_user_short_circuits_without_hitting_v1():
    get = AsyncMock(return_value="at")
    with patch(_GET, new=get):
        assert await V1PerUserTokenStore().fetch("", "s") is None
    get.assert_not_called()


async def test_passes_user_and_server_through_to_v1():
    get = AsyncMock(return_value="at")
    with patch(_GET, new=get):
        await V1PerUserTokenStore().fetch("alice", "srv-1")
    get.assert_awaited_once_with("alice", "srv-1")
