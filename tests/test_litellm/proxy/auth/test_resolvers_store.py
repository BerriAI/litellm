from __future__ import annotations

import logging
from typing import Dict, Optional

import pytest

from litellm.constants import INVALID_API_KEY_ERROR_MESSAGE
from litellm.proxy._types import UserAPIKeyAuth, hash_token
from litellm.proxy.auth.resolvers.exceptions import (
    KeyNotFoundError,
    NoDatabaseConnectionError,
    PrincipalMissingSourceKeyError,
)
from litellm.proxy.auth.auth_method import AuthMethod
from litellm.proxy.auth.resolvers.models import Principal, PrincipalType
from litellm.proxy.auth.resolvers.store import IdentityStore


class _FakeCache:
    """Stands in for the DualCache get_key_object reads. It returns a cache hit
    before the DB is touched, so seeding it exercises resolve without a database
    (a non-None prisma client is still required; it is never reached on a hit)."""

    def __init__(self, entries: Optional[Dict[str, object]] = None) -> None:
        self._entries = entries or {}

    async def async_get_cache(self, key, *args, **kwargs):
        return self._entries.get(key)

    async def async_set_cache(self, *args, **kwargs):
        return None


class _FakeEmptyKeyTable:
    """Stands in for the PrismaClient the store reads the key row through. Returning
    no row is what a submitted key that was never issued looks like."""

    async def get_data(self, *args, **kwargs):
        return None


async def test_resolve_returns_a_principal_projected_from_the_looked_up_key():
    raw = "sk-live-abc"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1", team_id="t-1")
    store = IdentityStore(object(), _FakeCache({hash_token(raw): key}))

    principal = await store.resolve(hashed_token=hash_token(raw))

    assert isinstance(principal, Principal)
    assert principal.principal_type == PrincipalType.HUMAN
    assert principal.user is not None and principal.user.id == "u-1"
    assert [t.id for t in principal.teams] == ["t-1"]


async def test_resolve_carries_the_key_for_key_from_principal():
    raw = "sk-live-abc"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1", team_id="t-1")
    store = IdentityStore(object(), _FakeCache({hash_token(raw): key}))

    principal = await store.resolve(hashed_token=hash_token(raw))
    recovered = IdentityStore.key_from_principal(principal)

    assert recovered.user_id == "u-1"
    assert recovered.team_id == "t-1"


def test_key_from_principal_raises_when_no_source_key_is_carried():
    bare = Principal(
        principal_type=PrincipalType.SERVICE_ACCOUNT,
        subject="svc",
        auth_method=AuthMethod.API_KEY,
    )
    with pytest.raises(PrincipalMissingSourceKeyError):
        IdentityStore.key_from_principal(bare)


async def test_resolve_raises_without_a_db_connection():
    store = IdentityStore(None, _FakeCache())
    with pytest.raises(NoDatabaseConnectionError):
        await store.resolve(hashed_token="missing")


async def test_resolve_logs_the_hash_when_no_key_row_exists(caplog):
    """Every path that resolves a key reaches this raise site, the JWT key-mapping
    ones included, so it is the only place the operator's lookup detail can be logged
    once the 401 itself stopped carrying it. Killing the log leaves a JWT-mapped miss
    with nothing but a generic 401 in the operator's logs."""
    missing = hash_token("sk-live-not-a-key")
    store = IdentityStore(_FakeEmptyKeyTable(), _FakeCache())

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        with pytest.raises(KeyNotFoundError) as exc_info:
            await store.resolve(hashed_token=missing)

    assert exc_info.value.message == INVALID_API_KEY_ERROR_MESSAGE
    assert missing not in str(exc_info.value)
    assert missing in caplog.text
