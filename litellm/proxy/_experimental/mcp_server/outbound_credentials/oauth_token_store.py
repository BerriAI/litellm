"""Per-user OAuth token store for the ``authorization_code`` mode.

The resolver reads a user's token through the injected ``OAuthTokenStore`` seam;
``CachedOAuthTokenStore`` is an expiry-aware cache in front of it. ``TokenStoreUnavailable``
signals an unreachable backing store, so an outage is never cached or read as "not authorized".

Refresh (using ``refresh_token`` once the access token has expired) and distributed single-flight
are the later hardening; this cache only avoids serving a token past its own expiry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple


@dataclass(frozen=True, slots=True)
class OAuthToken:
    """A user's OAuth credential: the bearer value, when it expires, and how to refresh it.

    ``expires_at`` is epoch seconds (``None`` means no known expiry). ``refresh_token`` is kept for
    the later refresh step; it is never minted into a header directly.
    """

    access_token: str
    expires_at: Optional[float] = None
    refresh_token: Optional[str] = None


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

    async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]: ...


class CachedOAuthTokenStore:
    """Expiry-aware cache over an ``OAuthTokenStore``.

    A cached token is served only while it is unexpired (minus ``expiry_skew_seconds``); past that
    the inner store is read again. Tokens with no known expiry, and the ``None`` "not authorized"
    result, are held for ``default_ttl_seconds`` so the store is not hit on every call. The clock is
    injected (wall-clock, since ``expires_at`` is epoch) so expiry is deterministic in tests, and a
    store outage (``TokenStoreUnavailable``) propagates without being cached.
    """

    def __init__(
        self,
        inner: OAuthTokenStore,
        *,
        default_ttl_seconds: float,
        expiry_skew_seconds: float = 30.0,
        max_size: int = 4096,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._inner = inner
        self._default_ttl_seconds = default_ttl_seconds
        self._expiry_skew_seconds = expiry_skew_seconds
        self._max_size = max_size
        self._clock = clock
        self._cache: Dict[Tuple[str, str], Tuple[Optional[OAuthToken], float]] = {}

    def _valid_until(self, token: Optional[OAuthToken]) -> float:
        if token is not None and token.expires_at is not None:
            return token.expires_at - self._expiry_skew_seconds
        return self._clock() + self._default_ttl_seconds

    async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
        key = (user_id, server_id)
        hit = self._cache.get(key)
        if hit is not None:
            token, valid_until = hit
            if self._clock() < valid_until:
                return token

        token = await self._inner.fetch(user_id, server_id)
        if len(self._cache) >= self._max_size:
            self._cache.clear()
        self._cache[key] = (token, self._valid_until(token))
        return token

    def invalidate(self, user_id: str, server_id: str) -> None:
        """Drop a cached entry after the user (re)authorizes or revokes, so a stale token or a
        stale "not authorized" None cannot mask the change."""
        self._cache.pop((user_id, server_id), None)
