"""Tests for the create/test-preview presented OAuth token store."""

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.presented_token_store import (
    PresentedOAuthTokenStore,
)


@pytest.mark.asyncio
async def test_serves_the_presented_token_regardless_of_key():
    token = OAuthToken(access_token="at", scopes=("read",))
    store = PresentedOAuthTokenStore(token)
    # one-shot store backs a single preview call, so the lookup key is irrelevant
    assert await store.fetch("alice", "srv-1") is token
    assert await store.fetch("", "") is token
