"""Tests for the v2 OAuth token cache (CachedOAuthTokenStore)."""

from typing import Dict, List, Optional, Tuple

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    CachedOAuthTokenStore,
    OAuthToken,
    TokenStoreUnavailable,
)


class _FakeStore:
    """An OAuthTokenStore that records calls and returns canned tokens."""

    def __init__(self, values: Dict[Tuple[str, str], Optional[OAuthToken]]) -> None:
        self._values = values
        self.calls: List[Tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
        self.calls.append((user_id, server_id))
        return self._values.get((user_id, server_id))


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


async def test_serves_token_until_its_expiry():
    token = OAuthToken(access_token="at", expires_at=1100.0)
    inner = _FakeStore({("u", "s"): token})
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(
        inner, default_ttl_seconds=60, expiry_skew_seconds=30, clock=clock
    )

    assert await store.fetch("u", "s") is token
    clock.t = 1060.0  # still before expiry - skew (1100 - 30 = 1070)
    assert await store.fetch("u", "s") is token
    assert inner.calls == [("u", "s")]  # served from cache, store hit once


async def test_refetches_once_token_has_expired():
    token = OAuthToken(access_token="at", expires_at=1100.0)
    inner = _FakeStore({("u", "s"): token})
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(
        inner, default_ttl_seconds=60, expiry_skew_seconds=30, clock=clock
    )

    await store.fetch("u", "s")
    clock.t = 1080.0  # past expiry - skew (1070)
    await store.fetch("u", "s")
    assert len(inner.calls) == 2  # re-read after the cached token expired


async def test_caches_not_authorized_none_for_default_ttl():
    inner = _FakeStore({})  # user has not authorized
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=clock)

    assert await store.fetch("u", "s") is None
    clock.t = 1059.0
    assert await store.fetch("u", "s") is None
    assert inner.calls == [("u", "s")]  # None cached for the TTL window


async def test_default_ttl_applies_to_tokens_without_expiry():
    token = OAuthToken(access_token="at", expires_at=None)
    inner = _FakeStore({("u", "s"): token})
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=clock)

    await store.fetch("u", "s")
    clock.t = 1061.0
    await store.fetch("u", "s")
    assert len(inner.calls) == 2  # no-expiry token re-read after the default TTL


async def test_invalidate_forces_refetch():
    inner = _FakeStore({})
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=_Clock())

    assert await store.fetch("u", "s") is None
    inner._values[("u", "s")] = OAuthToken(access_token="fresh")
    store.invalidate("u", "s")
    result = await store.fetch("u", "s")
    assert result is not None and result.access_token == "fresh"


async def test_store_unavailable_is_not_cached():
    class _FailingStore:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
            self.calls += 1
            raise TokenStoreUnavailable("down")

    inner = _FailingStore()
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=_Clock())

    for _ in range(2):
        with pytest.raises(TokenStoreUnavailable):
            await store.fetch("u", "s")
    assert inner.calls == 2  # outage re-attempted, not cached


async def test_isolates_by_subject():
    a = OAuthToken(access_token="a")
    b = OAuthToken(access_token="b")
    inner = _FakeStore({("u1", "s"): a, ("u2", "s"): b})
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=_Clock())

    first = await store.fetch("u1", "s")
    second = await store.fetch("u2", "s")
    assert first is not None and first.access_token == "a"
    assert second is not None and second.access_token == "b"
