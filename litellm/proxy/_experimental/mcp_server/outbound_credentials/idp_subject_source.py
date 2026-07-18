"""Source the delegated user's live IdP subject token for an OBO (token_exchange) mint.

Path B of the agent-delegation flow: at consent the user's IdP (e.g. Okta) grant is captured and
stored; at runtime the ``token_exchange`` arm needs the user's *own* subject token, not the agent's
admission bearer, to feed into the RFC 8693 exchange. ``StoredIdpGrantSource`` reads the stored grant
keyed by ``(user, idp)``, refreshes it to a live access token when it has expired, and returns that
token. It returns ``None`` when the user has no usable grant (never consented, or an expired grant with
no refresh_token) so the arm fails closed with a 401 that tells the client to authenticate the *user*
with the IdP; the agent's own token is never a fallback (the escalation the flow forbids).

The IdP is identified by its token endpoint, so one grant serves every ``token_exchange`` upstream
fronted by the same authorization server (keyed by IdP, not by upstream server). The DB read and the
IdP refresh POST are injected as edges (this module stays v1-free and deals only in ``OAuthToken``);
the refresh is single-flighted per ``(user, idp)`` so concurrent callers collapse to one refresh,
which also avoids invalidating each other's grant under IdP refresh-token rotation.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    TokenEndpointAuthConfigError,
    build_token_endpoint_client_auth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InProcessRefreshCoordinator,
    OAuthToken,
    RefreshCoordinator,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    TokenExchangeConfig,
)

# Prefix the (user, idp) credential-store key so an IdP grant row is not mistaken for a
# per-upstream-server credential row that shares the same table, and does not collide with a normal
# server_id (a UUID or hostname) absent a deliberately pathological "idp::"-prefixed one. The row is
# additionally tagged with a distinct payload type, so the oauth2 readers ignore it regardless.
_IDP_GRANT_KEY_PREFIX = "idp::"

_EXPIRY_SKEW_SECONDS = 60.0

# Reads the stored IdP grant for (user_id, idp_key) as an OAuthToken, or None when the user has none.
ReadIdpGrant = Callable[[str, str], Awaitable["OAuthToken | None"]]
# Refreshes the grant against the IdP (refresh_token grant) and persists the result, returning the
# fresh token or None when it cannot be refreshed. Threads the config so the endpoint and client
# credentials come from the calling server's token_exchange config.
RefreshIdpGrant = Callable[[str, str, TokenExchangeConfig, "OAuthToken"], Awaitable["OAuthToken | None"]]
# POSTs an OAuth form to a token endpoint and returns the parsed JSON body, or None on any failure.
TokenEndpointPost = Callable[[str, "dict[str, str]", "dict[str, str]"], Awaitable["dict[str, object] | None"]]


class PersistIdpGrant(Protocol):
    async def __call__(
        self,
        user_id: str,
        idp_key: str,
        access_token: str,
        refresh_token: str | None,
        expires_in: int | None,
        scopes: tuple[str, ...] | None,
    ) -> None: ...


def _parse_expires_in(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(float(raw))
        except ValueError:
            return None
    return None


def idp_grant_key(token_exchange_endpoint: str) -> str:
    """The (user, idp) storage key's idp component, derived from the AS token endpoint.

    All ``token_exchange`` servers fronted by the same authorization server share one endpoint, so
    they share one grant. Normalized (trailing slash stripped) so trivially different spellings of the
    same endpoint resolve to one key.
    """
    return f"{_IDP_GRANT_KEY_PREFIX}{token_exchange_endpoint.rstrip('/')}"


class IdpSubjectTokenSource(Protocol):
    """Produces the delegated user's live IdP subject token for a ``token_exchange`` mint, or None."""

    async def subject_token(self, user_id: str, config: TokenExchangeConfig) -> str | None: ...


class StoredIdpGrantSource:
    """Reads the user's stored IdP grant and refreshes it to a live subject token when expired.

    The DB read and the refresh POST are injected so the pure logic (expiry, single-flight, fail-closed
    on miss) is testable without I/O. The clock is injected for deterministic expiry in tests.
    """

    def __init__(
        self,
        read_grant: ReadIdpGrant,
        refresh_grant: RefreshIdpGrant,
        *,
        coordinator: RefreshCoordinator | None = None,
        clock: Callable[[], float] = time.time,
        expiry_skew_seconds: float = _EXPIRY_SKEW_SECONDS,
    ) -> None:
        self._read_grant = read_grant
        self._refresh_grant = refresh_grant
        self._coordinator: RefreshCoordinator = coordinator or InProcessRefreshCoordinator()
        self._clock = clock
        self._expiry_skew_seconds = expiry_skew_seconds

    async def subject_token(self, user_id: str, config: TokenExchangeConfig) -> str | None:
        endpoint = config.token_exchange_endpoint
        if not endpoint:
            # No IdP endpoint to source or refresh against; the exchanger separately fails closed (412)
            # for a non-delegated call, so here we simply have no subject material for the user.
            return None
        idp_key = idp_grant_key(endpoint)
        grant = await self._read_grant(user_id, idp_key)
        if grant is None:
            return None
        if not self._is_expired(grant):
            return grant.access_token
        refreshed = await self._coordinator.run(
            user_id,
            idp_key,
            refresh=lambda: self._refresh_if_still_expired(user_id, idp_key, config),
            reread=lambda: self._read_grant(user_id, idp_key),
        )
        if refreshed is None or self._is_expired(refreshed):
            return None
        return refreshed.access_token

    async def _refresh_if_still_expired(
        self, user_id: str, idp_key: str, config: TokenExchangeConfig
    ) -> OAuthToken | None:
        # Re-read under the single-flight so a token another caller just refreshed is reused rather
        # than refreshed again (and its refresh_token spent again under IdP rotation).
        latest = await self._read_grant(user_id, idp_key)
        if latest is None:
            return None
        if not self._is_expired(latest):
            return latest
        if latest.refresh_token is None:
            return None
        return await self._refresh_grant(user_id, idp_key, config, latest)

    def _is_expired(self, token: OAuthToken) -> bool:
        return token.expires_at is not None and self._clock() >= token.expires_at - self._expiry_skew_seconds


class IdpGrantRefresher:
    """Refreshes a stored IdP grant via the RFC 6749 refresh_token grant, then persists the rotation.

    Unlike the authorization_code refresher, the token endpoint and client credentials come from the
    calling server's ``TokenExchangeConfig`` (the IdP is the token-exchange authorization server), not
    a per-server lookup, since one IdP grant is shared across every ``token_exchange`` upstream it
    fronts. The HTTP POST and the persist are injected so the form-building and (untyped) response
    parsing stay testable without a live IdP or DB. Returns ``None`` (the source then challenges) when
    there is no refresh_token, the config lacks the client credentials to authenticate to the endpoint,
    or the grant fails; never a stale or partial token. A rotated refresh_token replaces the old one;
    an omitted one is carried forward.
    """

    def __init__(
        self,
        token_endpoint: TokenEndpointPost,
        persist: PersistIdpGrant,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._token_endpoint = token_endpoint
        self._persist = persist
        self._clock = clock

    async def refresh(
        self, user_id: str, idp_key: str, config: TokenExchangeConfig, token: OAuthToken
    ) -> OAuthToken | None:
        endpoint = config.token_exchange_endpoint
        if token.refresh_token is None or not endpoint or not config.client_id or config.client_secret is None:
            return None
        try:
            client_auth = build_token_endpoint_client_auth(
                auth_method=config.token_endpoint_auth_method,
                client_id=config.client_id,
                client_secret=config.client_secret.get_secret_value(),
            )
        except TokenEndpointAuthConfigError as exc:
            verbose_logger.warning("MCP IdP grant refresh misconfigured for %s: %s", idp_key, exc)
            return None
        form = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            **client_auth.body,
        }
        body = await self._token_endpoint(endpoint, form, client_auth.headers)
        if body is None:
            return None
        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None
        rotated = body.get("refresh_token")
        new_refresh = rotated if isinstance(rotated, str) and rotated else token.refresh_token
        expires_in = _parse_expires_in(body.get("expires_in"))
        await self._persist(user_id, idp_key, access_token, new_refresh, expires_in, token.scopes or None)
        return OAuthToken(
            access_token=access_token,
            expires_at=self._clock() + expires_in if expires_in is not None else None,
            refresh_token=new_refresh,
            scopes=token.scopes,
        )
