"""Tests for the create/test-preview presented OAuth token store."""

from typing import Final

import pytest

from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy.mcp_server.outbound_credentials.presented_token_store import (
    PresentedOAuthTokenStore,
)


@pytest.mark.asyncio
async def test_serves_the_presented_token_regardless_of_key():
    token: Final = OAuthToken(access_token="at", scopes=("read",))
    store: Final = PresentedOAuthTokenStore(token)
    # one-shot store backs a single preview call, so the lookup key is irrelevant
    assert await store.fetch("alice", "srv-1") is token
    assert await store.fetch("", "") is token
    assert await store.peek("alice", "srv-1") is token


@pytest.mark.asyncio
async def test_a_rejected_presented_token_cannot_be_refreshed_in_place():
    token: Final = OAuthToken(access_token="at")
    store: Final = PresentedOAuthTokenStore(token)

    outcome: Final = await store.refresh_now("alice", "srv-1", "at")

    assert outcome.is_error()
    assert await store.fetch("alice", "srv-1") is token
