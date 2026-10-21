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
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, expiry_skew_seconds=30, clock=clock)

    assert await store.fetch("u", "s") is token
    clock.t = 1060.0  # still before expiry - skew (1100 - 30 = 1070)
    assert await store.fetch("u", "s") is token
    assert inner.calls == [("u", "s")]  # served from cache, store hit once


async def test_refetches_once_token_has_expired():
    token = OAuthToken(access_token="at", expires_at=1100.0)
    inner = _FakeStore({("u", "s"): token})
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, expiry_skew_seconds=30, clock=clock)

    await store.fetch("u", "s")
    clock.t = 1080.0  # past expiry - skew (1070)
    await store.fetch("u", "s")
    assert len(inner.calls) == 2  # re-read after the cached token expired


async def test_does_not_cache_the_not_authorized_miss():
    inner = _FakeStore({})  # user has not authorized
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=clock)

    assert await store.fetch("u", "s") is None
    clock.t = 1059.0
    assert await store.fetch("u", "s") is None
    assert inner.calls == [
        ("u", "s"),
        ("u", "s"),
    ]  # misses re-read the store, never cached


async def test_token_stored_after_a_miss_is_visible_immediately():
    # No invalidation needed: a miss is never cached, so a token written after the OAuth flow is
    # served on the very next call (matching v1, where misses always re-read the source).
    inner = _FakeStore({})
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=_Clock())

    assert await store.fetch("u", "s") is None
    inner._values[("u", "s")] = OAuthToken(access_token="fresh")
    result = await store.fetch("u", "s")
    assert result is not None and result.access_token == "fresh"


async def test_default_ttl_applies_to_tokens_without_expiry():
    token = OAuthToken(access_token="at", expires_at=None)
    inner = _FakeStore({("u", "s"): token})
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=clock)

    await store.fetch("u", "s")
    clock.t = 1061.0
    await store.fetch("u", "s")
    assert len(inner.calls) == 2  # no-expiry token re-read after the default TTL


async def test_invalidate_drops_a_cached_token():
    # invalidate covers rotation/revocation of a *cached* token (the miss path needs no invalidate).
    inner = _FakeStore({("u", "s"): OAuthToken(access_token="t1")})
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, clock=_Clock())

    first = await store.fetch("u", "s")
    assert first is not None and first.access_token == "t1"  # cached
    inner._values[("u", "s")] = OAuthToken(access_token="t2")  # rotated
    await store.invalidate("u", "s")
    second = await store.fetch("u", "s")
    assert second is not None and second.access_token == "t2"  # re-read after invalidate


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


async def test_bounded_cache_evicts_oldest_not_everything():
    inner = _FakeStore(
        {
            ("u1", "s"): OAuthToken(access_token="k1"),
            ("u2", "s"): OAuthToken(access_token="k2"),
            ("u3", "s"): OAuthToken(access_token="k3"),
        }
    )
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=60, max_size=2, clock=_Clock())

    await store.fetch("u1", "s")
    await store.fetch("u2", "s")
    await store.fetch("u3", "s")  # at capacity -> evict the oldest (u1), keep u2
    await store.fetch("u2", "s")  # still cached
    await store.fetch("u1", "s")  # was evicted -> re-read

    assert inner.calls.count(("u2", "s")) == 1  # only the oldest was evicted, not everything
    assert inner.calls.count(("u1", "s")) == 2


class _RefreshablePair:
    """A store + refresher pair that simulates persistence: refresh() updates what fetch returns,
    and yields once so concurrent callers actually contend on the single-flight lock."""

    def __init__(self, initial: Optional[OAuthToken]) -> None:
        self._current = initial
        self.fetch_calls = 0
        self.refresh_calls = 0
        self.refresh_args: List[Tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
        self.fetch_calls += 1
        return self._current

    async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> Optional[OAuthToken]:
        self.refresh_calls += 1
        self.refresh_args.append((user_id, server_id))
        await asyncio.sleep(0)  # yield so other concurrent callers reach the lock and wait
        self._current = OAuthToken(access_token="refreshed", expires_at=9999.0)
        return self._current


async def test_refreshing_passes_through_a_fresh_token():
    pair = _RefreshablePair(OAuthToken(access_token="ok", expires_at=9999.0))
    store = RefreshingTokenStore(pair, pair, expiry_skew_seconds=30, clock=_Clock(1000.0))

    token = await store.fetch("u", "s")
    assert token is not None and token.access_token == "ok"
    assert pair.refresh_calls == 0  # not near expiry -> no refresh


async def test_refreshing_mints_a_fresh_token_when_expired():
    pair = _RefreshablePair(OAuthToken(access_token="old", expires_at=900.0))
    store = RefreshingTokenStore(pair, pair, expiry_skew_seconds=30, clock=_Clock(1000.0))

    token = await store.fetch("u", "s")
    assert token is not None and token.access_token == "refreshed"
    assert pair.refresh_calls == 1
    assert pair.refresh_args == [("u", "s")]  # the seam threads the grant/persist key through


async def test_refreshing_returns_none_when_it_cannot_refresh():
    class _NoRefresh:
        async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
            return OAuthToken(access_token="old", expires_at=900.0)

        async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> Optional[OAuthToken]:
            return None  # e.g. no refresh_token

    src = _NoRefresh()
    store = RefreshingTokenStore(src, src, expiry_skew_seconds=30, clock=_Clock(1000.0))
    # expired and unrefreshable -> None (the arm challenges), never a stale bearer
    assert await store.fetch("u", "s") is None


async def test_refreshing_is_single_flight_under_concurrency():
    pair = _RefreshablePair(OAuthToken(access_token="old", expires_at=900.0))
    store = RefreshingTokenStore(pair, pair, expiry_skew_seconds=30, clock=_Clock(1000.0))

    results = await asyncio.gather(*[store.fetch("u", "s") for _ in range(5)])
    assert pair.refresh_calls == 1  # one refresh shared across 5 concurrent callers
    assert all(r is not None and r.access_token == "refreshed" for r in results)


class _StaleReadRacePair:
    def __init__(self) -> None:
        self._old_token = OAuthToken(access_token="old", expires_at=900.0)
        self._current = self._old_token
        self.fetch_calls = 0
        self.refresh_calls = 0
        self.refresh_started = asyncio.Event()
        self.finish_refresh = asyncio.Event()
        self.stale_read_started = asyncio.Event()
        self.finish_stale_read = asyncio.Event()

    async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
        self.fetch_calls += 1
        if self.fetch_calls == 3:
            self.stale_read_started.set()
            await self.finish_stale_read.wait()
            return self._old_token
        return self._current

    async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> Optional[OAuthToken]:
        self.refresh_calls += 1
        self.refresh_started.set()
        await self.finish_refresh.wait()
        self._current = OAuthToken(access_token="refreshed", expires_at=9999.0)
        return self._current


async def test_stale_read_after_refresh_rereads_before_starting_new_refresh():
    pair = _StaleReadRacePair()
    store = RefreshingTokenStore(pair, pair, expiry_skew_seconds=30, clock=_Clock(1000.0))

    first = asyncio.create_task(store.fetch("u", "s"))
    await pair.refresh_started.wait()
    second = asyncio.create_task(store.fetch("u", "s"))
    await pair.stale_read_started.wait()

    pair.finish_refresh.set()
    first_result = await first
    pair.finish_stale_read.set()
    second_result = await second

    assert first_result is not None and first_result.access_token == "refreshed"
    assert second_result is not None and second_result.access_token == "refreshed"
    assert pair.refresh_calls == 1


async def test_refresh_failure_is_shared_by_joiners_not_re_run():
    class _FailingRefresher:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
            return OAuthToken(access_token="old", expires_at=900.0)

        async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> Optional[OAuthToken]:
            self.calls += 1
            await asyncio.sleep(0)  # let the concurrent callers join the same task
            raise RuntimeError("refresh boom")

    src = _FailingRefresher()
    store = RefreshingTokenStore(src, src, expiry_skew_seconds=30, clock=_Clock(1000.0))

    results = await asyncio.gather(*[store.fetch("u", "s") for _ in range(3)], return_exceptions=True)
    assert src.calls == 1  # single-flight: one attempt, the failure is shared
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_cache_default_skew_is_60_seconds():
    # The default expiry skew is the industry-standard 60s (Spring Security's clock-skew and
    # refresh-buffer default; within RFC 7519's "a few minutes" leeway), so a cached token stops
    # being served 60s before its real expiry. The boundary sits at expires_at - 60 = 1040.
    token = OAuthToken(access_token="at", expires_at=1100.0)
    inner = _FakeStore({("u", "s"): token})
    clock = _Clock(1000.0)
    store = CachedOAuthTokenStore(inner, default_ttl_seconds=600, clock=clock)

    await store.fetch("u", "s")
    clock.t = 1039.0  # just inside expires_at - 60 -> still served from cache
    await store.fetch("u", "s")
    assert len(inner.calls) == 1
    clock.t = 1041.0  # just past expires_at - 60 -> re-read (a 30s skew would still cache here)
    await store.fetch("u", "s")
    assert len(inner.calls) == 2


async def test_refreshing_default_skew_is_60_seconds():
    # Same 60s default for the proactive refresh threshold: a token within 60s of expiry refreshes,
    # one further out does not. At clock 1000 the boundary expires_at - 60 = 1000 lands on expires_at
    # 1060 (refresh) vs 1061 (no refresh), pinning the default to exactly 60 (a 30s skew would not
    # refresh either case).
    not_near = _RefreshablePair(OAuthToken(access_token="ok", expires_at=1061.0))
    not_near_store = RefreshingTokenStore(not_near, not_near, clock=_Clock(1000.0))
    not_near_token = await not_near_store.fetch("u", "s")
    assert not_near_token is not None and not_near_token.access_token == "ok"
    assert not_near.refresh_calls == 0

    near = _RefreshablePair(OAuthToken(access_token="old", expires_at=1060.0))
    near_store = RefreshingTokenStore(near, near, clock=_Clock(1000.0))
    near_token = await near_store.fetch("u", "s")
    assert near_token is not None and near_token.access_token == "refreshed"
    assert near.refresh_calls == 1


def test_oauth_token_repr_masks_the_secrets():
    token = OAuthToken(access_token="super-secret", expires_at=123.0, refresh_token="rt-secret")
    rendered = repr(token)
    assert "super-secret" not in rendered
    assert "rt-secret" not in rendered
    assert "access_token=***" in rendered
    assert "has_refresh_token=True" in rendered


class _RecordingBackend:
    """A TokenCacheBackend that records calls, proving CachedOAuthTokenStore delegates storage."""

    def __init__(self) -> None:
        self.sets: List[Tuple[str, str, OAuthToken, float]] = []
        self.deletes: List[Tuple[str, str]] = []
        self._store: Dict[Tuple[str, str], OAuthToken] = {}

    async def get(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
        return self._store.get((user_id, server_id))

    async def set(self, user_id: str, server_id: str, token: OAuthToken, ttl_seconds: float) -> None:
        self.sets.append((user_id, server_id, token, ttl_seconds))
        self._store[(user_id, server_id)] = token

    async def delete(self, user_id: str, server_id: str) -> None:
        self.deletes.append((user_id, server_id))
        self._store.pop((user_id, server_id), None)


async def test_cache_delegates_storage_to_an_injected_backend():
    backend = _RecordingBackend()
    inner = _FakeStore({("u", "s"): OAuthToken(access_token="at", expires_at=1100.0)})
    store = CachedOAuthTokenStore(
        inner,
        default_ttl_seconds=60,
        expiry_skew_seconds=30,
        backend=backend,
        clock=_Clock(1000.0),
    )

    token = await store.fetch("u", "s")
    assert token is not None and token.access_token == "at"
    # written through to the injected backend, TTL = expires_at - skew - now = 1100 - 30 - 1000
    assert backend.sets == [("u", "s", token, 70.0)]
    again = await store.fetch("u", "s")  # served by the backend, not the inner store
    assert again is not None
    assert inner.calls == [("u", "s")]


async def test_cache_miss_deletes_from_the_injected_backend():
    backend = _RecordingBackend()
    store = CachedOAuthTokenStore(_FakeStore({}), default_ttl_seconds=60, backend=backend, clock=_Clock())
    assert await store.fetch("u", "s") is None
    assert backend.deletes == [("u", "s")]  # a miss is never cached


class _RecordingCoordinator:
    """A RefreshCoordinator that records the call and runs the refresh, proving delegation."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, user_id, server_id, refresh, reread):
        self.calls += 1
        return await refresh()


async def test_refreshing_delegates_single_flight_to_an_injected_coordinator():
    pair = _RefreshablePair(OAuthToken(access_token="old", expires_at=900.0))
    coordinator = _RecordingCoordinator()
    store = RefreshingTokenStore(
        pair,
        pair,
        expiry_skew_seconds=30,
        coordinator=coordinator,
        clock=_Clock(1000.0),
    )

    token = await store.fetch("u", "s")
    assert token is not None and token.access_token == "refreshed"
    assert coordinator.calls == 1  # the injected coordinator drove the refresh


class _LoserCoordinator:
    """Simulates losing the cross-replica election: the loser only re-reads what the winner persisted,
    it never refreshes itself."""

    async def run(self, user_id, server_id, refresh, reread):
        return await reread()


async def test_loser_surfaces_none_not_a_stale_token_when_the_winner_refresh_failed():
    # The winner's refresh failed, so the store still holds the expired token. A loser must surface
    # None (-> re-auth challenge) like the winner did, never the still-expired bearer (the upstream
    # would 401 it). _RefreshablePair only updates on refresh, so a loser that never refreshes keeps
    # re-reading the expired token.
    pair = _RefreshablePair(OAuthToken(access_token="expired", expires_at=900.0))
    store = RefreshingTokenStore(
        pair,
        pair,
        expiry_skew_seconds=30,
        coordinator=_LoserCoordinator(),
        clock=_Clock(1000.0),
    )

    assert await store.fetch("u", "s") is None
    assert pair.refresh_calls == 0  # the loser never refreshes; it only re-reads


async def test_loser_rereads_the_fresh_token_a_successful_winner_persisted():
    class _ExpiredThenWinnerPersisted:
        # First read sees the expired token (triggers the refresh path); the re-read sees the fresh
        # token a winner persisted in between.
        def __init__(self) -> None:
            self._reads = 0
            self.refresh_calls = 0

        async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
            self._reads += 1
            if self._reads == 1:
                return OAuthToken(access_token="expired", expires_at=900.0)
            return OAuthToken(access_token="winner-fresh", expires_at=9999.0)

        async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> Optional[OAuthToken]:
            self.refresh_calls += 1
            raise AssertionError("a loser must not refresh")

    src = _ExpiredThenWinnerPersisted()
    store = RefreshingTokenStore(
        src,
        src,
        expiry_skew_seconds=30,
        coordinator=_LoserCoordinator(),
        clock=_Clock(1000.0),
    )

    token = await store.fetch("u", "s")
    assert token is not None and token.access_token == "winner-fresh"
    assert src.refresh_calls == 0
