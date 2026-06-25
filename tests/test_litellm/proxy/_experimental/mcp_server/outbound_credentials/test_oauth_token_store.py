"""Tests for the v2 OAuth token cache and refresh (CachedOAuthTokenStore, RefreshingTokenStore)."""

import asyncio
from typing import Dict, List, Optional, Tuple

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    CachedOAuthTokenStore,
    OAuthToken,
    RefreshingTokenStore,
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


class _RefreshablePair:
    """A store + refresher pair that simulates persistence: refresh() updates what fetch returns,
    and yields once so concurrent callers actually contend on the single-flight lock."""

    def __init__(self, initial: Optional[OAuthToken]) -> None:
        self._current = initial
        self.fetch_calls = 0
        self.refresh_calls = 0

    async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
        self.fetch_calls += 1
        return self._current

    async def refresh(self, token: OAuthToken) -> Optional[OAuthToken]:
        self.refresh_calls += 1
        await asyncio.sleep(
            0
        )  # yield so other concurrent callers reach the lock and wait
        self._current = OAuthToken(access_token="refreshed", expires_at=9999.0)
        return self._current


async def test_refreshing_passes_through_a_fresh_token():
    pair = _RefreshablePair(OAuthToken(access_token="ok", expires_at=9999.0))
    store = RefreshingTokenStore(
        pair, pair, expiry_skew_seconds=30, clock=_Clock(1000.0)
    )

    token = await store.fetch("u", "s")
    assert token is not None and token.access_token == "ok"
    assert pair.refresh_calls == 0  # not near expiry -> no refresh


async def test_refreshing_mints_a_fresh_token_when_expired():
    pair = _RefreshablePair(OAuthToken(access_token="old", expires_at=900.0))
    store = RefreshingTokenStore(
        pair, pair, expiry_skew_seconds=30, clock=_Clock(1000.0)
    )

    token = await store.fetch("u", "s")
    assert token is not None and token.access_token == "refreshed"
    assert pair.refresh_calls == 1


async def test_refreshing_returns_none_when_it_cannot_refresh():
    class _NoRefresh:
        async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
            return OAuthToken(access_token="old", expires_at=900.0)

        async def refresh(self, token: OAuthToken) -> Optional[OAuthToken]:
            return None  # e.g. no refresh_token

    src = _NoRefresh()
    store = RefreshingTokenStore(src, src, expiry_skew_seconds=30, clock=_Clock(1000.0))
    # expired and unrefreshable -> None (the arm challenges), never a stale bearer
    assert await store.fetch("u", "s") is None


async def test_refreshing_is_single_flight_under_concurrency():
    pair = _RefreshablePair(OAuthToken(access_token="old", expires_at=900.0))
    store = RefreshingTokenStore(
        pair, pair, expiry_skew_seconds=30, clock=_Clock(1000.0)
    )

    results = await asyncio.gather(*[store.fetch("u", "s") for _ in range(5)])
    assert pair.refresh_calls == 1  # one refresh shared across 5 concurrent callers
    assert all(r is not None and r.access_token == "refreshed" for r in results)


async def test_refresh_failure_is_shared_by_joiners_not_re_run():
    class _FailingRefresher:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
            return OAuthToken(access_token="old", expires_at=900.0)

        async def refresh(self, token: OAuthToken) -> Optional[OAuthToken]:
            self.calls += 1
            await asyncio.sleep(0)  # let the concurrent callers join the same task
            raise RuntimeError("refresh boom")

    src = _FailingRefresher()
    store = RefreshingTokenStore(src, src, expiry_skew_seconds=30, clock=_Clock(1000.0))

    results = await asyncio.gather(
        *[store.fetch("u", "s") for _ in range(3)], return_exceptions=True
    )
    assert src.calls == 1  # single-flight: one attempt, the failure is shared
    assert all(isinstance(r, RuntimeError) for r in results)


def test_oauth_token_repr_masks_the_secrets():
    token = OAuthToken(
        access_token="super-secret", expires_at=123.0, refresh_token="rt-secret"
    )
    rendered = repr(token)
    assert "super-secret" not in rendered
    assert "rt-secret" not in rendered
    assert "access_token=***" in rendered
    assert "has_refresh_token=True" in rendered
