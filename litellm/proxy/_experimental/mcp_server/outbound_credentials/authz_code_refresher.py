"""v2-native refresher for the ``authorization_code`` mode: the refresh_token grant, then persist.

Mints a fresh access token from a stored refresh_token by POSTing the RFC 6749 refresh_token grant to
the server's token endpoint, persists the rotated triple, and returns the new typed ``OAuthToken`` for
``RefreshingTokenStore`` to cache. The HTTP post and the persist are injected, so the orchestration
and the (untyped) response parsing stay testable without a live IdP or DB. Replaces v1's
``refresh_user_oauth_token`` as part of step 1b; rotation safety - one refresh per (user, server)
across replicas - is the wrapping store's distributed single-flight, not this refresher's concern.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

ServerLookup = Callable[[str], "MCPServer | None"]
TokenEndpointPost = Callable[[str, dict[str, str]], Awaitable["dict[str, object] | None"]]


class CredentialPersist(Protocol):
    async def __call__(
        self,
        user_id: str,
        server_id: str,
        access_token: str,
        refresh_token: str | None,
        expires_in: int | None,
        scopes: tuple[str, ...] | None,
    ) -> None: ...


def _parse_expires_in(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _parse_scopes(raw: object) -> tuple[str, ...] | None:
    return tuple(raw.split()) if isinstance(raw, str) and raw else None


class AuthorizationCodeRefresher:
    """``TokenRefresher`` for authorization_code: refresh_token grant against the server, then persist.

    ``token_endpoint`` POSTs the OAuth form and returns the parsed JSON body (``None`` on any
    transport/HTTP failure, mirroring v1: a failed refresh is a miss, not a 500). ``persist`` writes
    the rotated triple for ``(user, server)`` - the v1 ``store_user_oauth_credential`` write, which
    stays. Returns ``None`` (the arm challenges) when there is no refresh_token, the server lacks a
    token endpoint, or the grant fails; never a stale or partial token. A rotated refresh_token from
    the response replaces the old one; an omitted one is carried forward, as are the recorded scopes
    when the response omits ``scope``.
    """

    def __init__(
        self,
        server_lookup: ServerLookup,
        token_endpoint: TokenEndpointPost,
        persist: CredentialPersist,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._server_lookup = server_lookup
        self._token_endpoint = token_endpoint
        self._persist = persist
        self._clock = clock

    async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> OAuthToken | None:
        if token.refresh_token is None:
            return None
        server = self._server_lookup(server_id)
        if server is None or not server.token_url:
            return None

        form = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            **({"client_id": server.client_id} if server.client_id else {}),
            **({"client_secret": server.client_secret} if server.client_secret else {}),
        }
        body = await self._token_endpoint(server.token_url, form)
        if body is None:
            return None
        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None

        rotated = body.get("refresh_token")
        new_refresh = rotated if isinstance(rotated, str) and rotated else token.refresh_token
        expires_in = _parse_expires_in(body.get("expires_in"))
        scopes = _parse_scopes(body.get("scope")) or token.scopes

        await self._persist(user_id, server_id, access_token, new_refresh, expires_in, scopes or None)
        return OAuthToken(
            access_token=access_token,
            expires_at=self._clock() + expires_in if expires_in is not None else None,
            refresh_token=new_refresh,
            scopes=scopes,
        )
