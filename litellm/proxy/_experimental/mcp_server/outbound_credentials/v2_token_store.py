"""v2-native per-user OAuth token read store for the ``authorization_code`` mode.

The raw "inner" store that ``RefreshingTokenStore`` and ``CachedOAuthTokenStore`` wrap: it reads the
user's persisted credential and returns a typed ``OAuthToken`` (access token, epoch expiry, refresh
token), validating the decoded credential blob at this boundary so no ``Any`` leaks past it. It does
not cache or refresh - those are the decorators. This replaces ``V1PerUserTokenStore`` (which handed
the whole read + cache + refresh to v1's core) as step 1b: the ``read_credential`` collaborator is
injected, so the DB/decoding plumbing stays testable and out of this seam.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)

CredentialReader = Callable[[str, str], Awaitable["dict[str, object] | None"]]


def _iso_to_epoch(expires_at: str) -> float | None:
    try:
        dt = datetime.fromisoformat(expires_at)
    except ValueError:
        return None
    # A timezone-naive expiry is stored as UTC (db.py writes ``datetime.now(timezone.utc)``),
    # so anchor it to UTC before ``.timestamp()`` - otherwise a non-UTC host would read it as
    # local time and skew the expiry, diverging from v1's ``_remaining_token_seconds``.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _to_scopes(raw: object) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(s for s in raw if isinstance(s, str))
    return ()


def _to_oauth_token(payload: dict[str, object]) -> OAuthToken | None:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str):
        return None
    refresh_token = payload.get("refresh_token")
    expires_at = payload.get("expires_at")
    return OAuthToken(
        access_token=access_token,
        expires_at=_iso_to_epoch(expires_at) if isinstance(expires_at, str) else None,
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        scopes=_to_scopes(payload.get("scopes")),
    )


class V2PerUserTokenStore:
    """``OAuthTokenStore`` that reads the user's persisted authorization_code credential, typed.

    The injected ``read_credential`` returns the decoded credential payload for a ``(user, server)``
    pair, or ``None`` when the user has not completed OAuth. A backing-store outage surfaces as
    ``TokenStoreUnavailable`` from the reader, which the arm turns into a challenge rather than a
    500, so ``fetch`` lets it propagate. Refresh is the wrapping ``RefreshingTokenStore``'s job, so
    the returned token carries ``expires_at`` and ``refresh_token`` for it to act on.
    """

    def __init__(self, read_credential: CredentialReader) -> None:
        self._read_credential = read_credential

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        if not user_id:
            return None
        payload = await self._read_credential(user_id, server_id)
        if payload is None:
            return None
        return _to_oauth_token(payload)
