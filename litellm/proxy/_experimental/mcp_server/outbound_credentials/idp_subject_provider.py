"""Composition root for the delegated-OBO IdP subject-token source (Path B).

Wires the pure ``StoredIdpGrantSource`` to its runtime edges: the DB read/persist of the user's stored
IdP grant (a distinct ``idp_grant``-typed credential in the per-user MCP credential store, keyed by IdP
rather than upstream server, so the oauth2-only server listings never surface it) and the httpx
refresh_token POST against the IdP. The token-endpoint POST is the same OAuth2 helper the
authorization_code refresh uses, reused here rather than re-implementing the untyped-httpx boundary.
``store_user_idp_grant`` is the store-back path the first-time consent flow calls to persist the
captured grant; the consent UI itself is out of scope here. Nothing reads a runtime global at build
time (prisma/httpx are acquired per call), so no lazy wrapper is needed, mirroring
``build_token_exchanger``.
"""

from __future__ import annotations

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_subject_source import (
    IdpGrantRefresher,
    PersistIdpGrant,
    StoredIdpGrantSource,
    TokenEndpointPost,
    idp_grant_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    CachedOAuthTokenStore,
    OAuthToken,
    TokenStoreUnavailable,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.v2_token_store import (
    CredentialReader,
    V2PerUserTokenStore,
)

# A grant with no declared expiry is cached for this long; one with an expiry is cached until then.
# Mirrors the authorization_code store's default so a warm grant avoids a DB read per delegated call.
_GRANT_READ_CACHE_TTL_SECONDS = 300.0


async def _read_credential(user_id: str, idp_key: str) -> dict[str, object] | None:
    from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415  # lazy: avoids import cycle
        get_user_idp_grant,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # lazy: runtime global

    if prisma_client is None:
        raise TokenStoreUnavailable("Database not connected")
    return await get_user_idp_grant(prisma_client, user_id, idp_key)


async def _persist_credential(
    user_id: str,
    idp_key: str,
    access_token: str,
    refresh_token: str | None,
    expires_in: int | None,
    scopes: tuple[str, ...] | None,
) -> bool:
    """Persist the grant, returning whether it was actually stored.

    A refresh treats the skipped save as best-effort (it still returns the fresh token), but the
    first-time consent capture must surface a False so the callback never reports a connection that
    was never stored.
    """
    from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415  # lazy: avoids import cycle
        store_user_idp_grant,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # lazy: runtime global

    if prisma_client is None:
        # Symmetric with _read_credential (which raises TokenStoreUnavailable): make the skipped
        # persist observable, since a refresh that rotated the IdP refresh_token then failed to save
        # it strands the user until re-consent.
        verbose_logger.warning("MCP IdP grant persist skipped: database not connected; a rotated grant may be lost")
        return False
    await store_user_idp_grant(
        prisma_client=prisma_client,
        user_id=user_id,
        idp_key=idp_key,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        scopes=list(scopes) if scopes else None,
    )
    return True


async def _post_token_endpoint(url: str, form: dict[str, str], headers: dict[str, str]) -> dict[str, object] | None:
    # get_async_httpx_client's signature carries an untyped param, so the imported symbol reads as
    # partially unknown; the httpx.Response JSON body is Any and the refresher validates each field, so
    # the untyped boundary is contained here. A failed refresh is a miss, not a 500 (matching the
    # authorization_code refresher), so any error becomes None.
    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415  # lazy import; avoids cycle
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # httpx handler untyped
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415  # lazy import

    request_headers = {"Accept": "application/json", **headers}
    try:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        response = await client.post(url, headers=request_headers, data=form)  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post signature is untyped
        if response is None:
            return None
        response.raise_for_status()
        body: dict[str, object] = response.json()  # pyright: ignore[reportAny]  # untyped JSON body, validated by the refresher
    except Exception as exc:  # noqa: BLE001  # any IdP/transport error is a refresh miss, not a 500
        verbose_logger.warning("MCP IdP grant refresh request failed: %s", exc)
        return None
    else:
        return body


def build_idp_subject_source(
    *,
    read_credential: CredentialReader = _read_credential,
    token_endpoint: TokenEndpointPost = _post_token_endpoint,
    persist: PersistIdpGrant = _persist_credential,
) -> StoredIdpGrantSource:
    # An expiry-aware cache in front of the DB read, mirroring the authorization_code chain
    # (Cached(V2PerUserTokenStore)), so a warm grant is served without a DB round-trip on every
    # delegated call; the config-aware refresh stays in the source. Built once so the cache is shared
    # across requests. Collaborators are injectable for tests.
    cached_read = CachedOAuthTokenStore(
        V2PerUserTokenStore(read_credential), default_ttl_seconds=_GRANT_READ_CACHE_TTL_SECONDS
    )

    async def read_grant(user_id: str, idp_key: str) -> OAuthToken | None:
        try:
            return await cached_read.fetch(user_id, idp_key)
        except TokenStoreUnavailable:
            return None

    refresher = IdpGrantRefresher(token_endpoint, persist)
    return StoredIdpGrantSource(read_grant, refresher.refresh)


async def capture_user_idp_grant(
    user_id: str,
    token_exchange_endpoint: str,
    access_token: str,
    *,
    refresh_token: str | None = None,
    expires_in: int | None = None,
    scopes: tuple[str, ...] | None = None,
) -> bool:
    """Persist a user's IdP grant for on-behalf-of exchange, keyed by the IdP (the AS token endpoint).

    The store-back path the first-time consent flow calls once it has captured the user's IdP grant
    (``authorization_code + offline_access`` against the IdP). One grant serves every ``token_exchange``
    upstream that IdP fronts, since it is keyed by the IdP endpoint rather than an upstream server.

    Returns whether the grant was stored, so the caller can distinguish a real connection from a
    skipped save (database unavailable) rather than reporting a connection that never persisted.
    """
    return await _persist_credential(
        user_id,
        idp_grant_key(token_exchange_endpoint),
        access_token,
        refresh_token,
        expires_in,
        scopes,
    )
