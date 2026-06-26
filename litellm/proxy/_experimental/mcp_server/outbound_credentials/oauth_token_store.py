"""Per-user OAuth token store for the ``authorization_code`` mode.

The resolver reads a user's token through the injected ``OAuthTokenStore`` seam;
``CachedOAuthTokenStore`` is an expiry-aware cache in front of it. ``TokenStoreUnavailable``
signals an unreachable backing store, so an outage is never cached or read as "not authorized".

``RefreshingTokenStore`` mints a fresh token through an injected ``TokenRefresher`` when the stored
one is near expiry, under in-process per-(user, server) single-flight so concurrent callers share
one refresh. Distributed (cross-replica) single-flight and reactive-401 refresh are the later
hardening. The mode plugs in its own source and refresher; the cache, store seam, and refresh
machinery are shared across the oauth2 modes (authorization_code / client_credentials /
token_exchange).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True, repr=False)
class OAuthToken:
    """A user's OAuth credential: the bearer value, when it expires, and how to refresh it.

    ``expires_at`` is epoch seconds (``None`` means no known expiry). ``refresh_token`` is what a
    ``TokenRefresher`` uses to mint a new access token when this one nears expiry (the refresh
    mechanism, ``RefreshingTokenStore``, is in this module; the concrete per-mode refresher lands
    with each mode); it is never minted into a header directly. ``repr`` masks both secrets so a
    stray log line cannot leak them (the values are still plain ``str`` for the header path, since
    ``SecretStr`` resolves as unknown under this repo's basedpyright).
    """

    access_token: str
    expires_at: float | None = None
    refresh_token: str | None = None

    def __repr__(self) -> str:
        has_refresh = self.refresh_token is not None
        return f"OAuthToken(access_token=***, expires_at={self.expires_at!r}, has_refresh_token={has_refresh})"


class TokenStoreUnavailable(Exception):
    """Raised by ``fetch`` when the backing token store is unreachable (e.g. the DB is down).

    Distinct from returning ``None`` for "the user has not authorized this server": a read-through
    cache skips caching the failure, and the resolver maps it to its fail-closed status rather than
    treating an outage as a definite absence.
    """


class OAuthTokenStore(Protocol):
    """Per-user OAuth token lookup for the ``authorization_code`` mode.

    Returns the user's token for an upstream, or ``None`` when they have not completed the OAuth
    flow (the arm turns that into a 401 challenge). The ``(user_id, server_id)`` pair fully scopes
    the lookup, so an implementation must never return one subject's token to another. Raises
    ``TokenStoreUnavailable`` when the backing store is unreachable, so an outage is never cached or
    read as a definite absence.
    """

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None: ...


class TokenRefresher(Protocol):
    """Mints a fresh token from an expired one and persists it, returning the new token.

    The action is mode-specific: the ``authorization_code`` refresh_token grant, the
    ``client_credentials`` grant, or an RFC 8693 re-exchange. Returns ``None`` when it cannot
    refresh (e.g. no ``refresh_token``), which the caller turns into a 401 challenge. It must
    persist the new token so later requests (and the surrounding cache) read it without refreshing.
    """

    async def refresh(self, token: OAuthToken) -> OAuthToken | None: ...


class CachedOAuthTokenStore:
    """Expiry-aware cache over an ``OAuthTokenStore``. Caches positive tokens only.

    A cached token is served only while it is unexpired (minus ``expiry_skew_seconds``), or for
    ``default_ttl_seconds`` if it carries no expiry; past that the inner store is read again. A
    "not authorized" (``None``) result is never cached: every miss re-reads the inner store, so a
    token written after the OAuth flow is visible immediately on every replica, matching v1 (which
    never caches misses). The clock is injected (wall-clock, since ``expires_at`` is epoch) so
    expiry is deterministic in tests, and a store outage (``TokenStoreUnavailable``) propagates
    without being cached.
    """

    def __init__(
        self,
        inner: OAuthTokenStore,
        *,
        default_ttl_seconds: float,
        expiry_skew_seconds: float = 60.0,
        max_size: int = 4096,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._inner = inner
        self._default_ttl_seconds = default_ttl_seconds
        self._expiry_skew_seconds = expiry_skew_seconds
        self._max_size = max_size
        self._clock = clock
        self._cache: dict[tuple[str, str], tuple[OAuthToken, float]] = {}

    def _valid_until(self, token: OAuthToken) -> float:
        if token.expires_at is not None:
            return token.expires_at - self._expiry_skew_seconds
        return self._clock() + self._default_ttl_seconds

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        key = (user_id, server_id)
        hit = self._cache.get(key)
        if hit is not None:
            token, valid_until = hit
            if self._clock() < valid_until:
                return token

        token = await self._inner.fetch(user_id, server_id)
        if token is None:
            # Never cache "not authorized": drop any stale entry and re-read on the next call, so
            # a token stored after the OAuth flow is seen immediately rather than after a TTL.
            self._cache.pop(key, None)
            return token
        if key not in self._cache and len(self._cache) >= self._max_size:
            # Evict the oldest entry (insertion order) to make room, rather than clearing the
            # whole cache and forcing every key to re-read the store at once.
            self._cache.pop(next(iter(self._cache)), None)
        self._cache[key] = (token, self._valid_until(token))
        return token

    def invalidate(self, user_id: str, server_id: str) -> None:
        """Drop a cached entry after the user (re)authorizes or revokes, so a stale token or a
        stale "not authorized" None cannot mask the change."""
        self._cache.pop((user_id, server_id), None)


class RefreshingTokenStore:
    """An ``OAuthTokenStore`` that proactively refreshes a near-expiry token.

    Reads from an inner store; if the token is within ``expiry_skew_seconds`` of expiry, it mints a
    fresh one via the injected ``TokenRefresher`` under per-(user, server) single-flight: the first
    caller refreshes while concurrent callers await the same in-flight future and share its result,
    instead of stampeding the IdP. The refresher persists the new token so later requests (and the
    surrounding cache) read it without refreshing again. An expired token the refresher cannot renew
    (``None``) is surfaced as ``None`` so the arm challenges, never a stale bearer.

    Single-flight here is in-process (one event loop). Cross-replica single-flight (Redis SET NX)
    and reactive-401 refresh are the later distributed hardening. Composes under
    ``CachedOAuthTokenStore`` so the refreshed token is cached until its own expiry.
    """

    def __init__(
        self,
        inner: OAuthTokenStore,
        refresher: TokenRefresher,
        *,
        expiry_skew_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._inner = inner
        self._refresher = refresher
        self._expiry_skew_seconds = expiry_skew_seconds
        self._clock = clock
        # In-flight refreshes, one task per (user, server). Each entry is removed by the task's
        # done-callback, so the map is bounded by concurrent refreshes, not by the number of
        # distinct users/servers ever seen.
        self._inflight: dict[tuple[str, str], asyncio.Future[OAuthToken | None]] = {}

    def _is_expired(self, token: OAuthToken) -> bool:
        return (
            token.expires_at is not None
            and self._clock() >= token.expires_at - self._expiry_skew_seconds
        )

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        token = await self._inner.fetch(user_id, server_id)
        if token is None or not self._is_expired(token):
            return token
        return await self._refresh_single_flight(user_id, server_id, token)

    async def _refresh_single_flight(
        self, user_id: str, server_id: str, token: OAuthToken
    ) -> OAuthToken | None:
        key = (user_id, server_id)
        task = self._inflight.get(key)
        if task is None:
            # First caller starts the refresh; concurrent callers await the same task and share its
            # result (or exception). The done-callback removes the entry, so the map self-cleans and
            # is bounded by in-flight refreshes, not by the number of distinct users/servers. The
            # task is detached from the caller, so a cancelled caller does not abort the refresh.
            task = asyncio.ensure_future(self._refresher.refresh(token))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        return await task
