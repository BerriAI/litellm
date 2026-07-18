"""Tests for the IdP subject-source composition root.

The composition wires an expiry-aware cache in front of the DB read (mirroring the authorization_code
token store chain), so a warm IdP grant is served without a DB round-trip on every delegated call.
"""

import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_subject_provider import (
    build_idp_subject_source,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    TokenExchangeConfig,
)

_CONFIG = TokenExchangeConfig(
    token_exchange_endpoint="https://idp.example.com/token",
    client_id="gateway-client",
    client_secret=SecretStr("gateway-secret"),
)


@pytest.mark.asyncio
async def test_grant_read_is_cached_across_delegated_calls():
    reads: list[tuple[str, str]] = []

    async def counting_read_credential(user_id, idp_key):
        reads.append((user_id, idp_key))
        # A far-future expiry so the grant is fresh -> cacheable, no refresh.
        return {"type": "idp_grant", "access_token": "live-at", "expires_at": "2999-01-01T00:00:00+00:00"}

    source = build_idp_subject_source(read_credential=counting_read_credential)

    first = await source.subject_token("alice", _CONFIG)
    second = await source.subject_token("alice", _CONFIG)

    assert first == "live-at"
    assert second == "live-at"
    # The second delegated call is served from the shared cache: exactly one DB read, not one per call.
    assert len(reads) == 1


@pytest.mark.asyncio
async def test_distinct_users_each_read_once():
    reads: list[tuple[str, str]] = []

    async def counting_read_credential(user_id, idp_key):
        reads.append((user_id, idp_key))
        return {"type": "idp_grant", "access_token": f"at-{user_id}", "expires_at": "2999-01-01T00:00:00+00:00"}

    source = build_idp_subject_source(read_credential=counting_read_credential)
    assert await source.subject_token("alice", _CONFIG) == "at-alice"
    assert await source.subject_token("bob", _CONFIG) == "at-bob"
    # Cache is keyed per user, so the two users do not share an entry (and neither re-reads).
    assert sorted(reads) == [("alice", "idp::https://idp.example.com/token"), ("bob", "idp::https://idp.example.com/token")]
