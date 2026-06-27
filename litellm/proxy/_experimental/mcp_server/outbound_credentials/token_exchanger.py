"""v2-native RFC 8693 token exchange (OBO): swap the caller's token for an upstream one.

The pure core of the ``token_exchange`` mode. Given the caller's ``subject_token`` and the server's
``TokenExchangeConfig``, ``Rfc8693TokenExchanger.exchange`` POSTs the RFC 8693 token-exchange grant to
the configured endpoint and returns the upstream-bound ``access_token`` as a typed ``OAuthToken``, or a
typed ``CredError`` - never a raise (the HTTP edge is the injected ``ExchangeHttpPost``, whose adapter
contains the I/O). The exchanged token is cached and single-flighted per ``(subject_token, server)`` so
a repeated caller token skips the IdP round-trip and concurrent calls collapse to one exchange, reusing
the shared in-process cache + coordinator foundation. A rotated caller token hashes to a new key and
re-exchanges. Pure v2: no imports from v1.

A missing/expired exchange is an error, never a fall-through to a weaker source (§1.5): the caller
presenting no token is the resolver arm's 401, and an IdP that does not return a usable token is an
``upstream_unavailable`` here.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InMemoryTokenCacheBackend,
    InProcessRefreshCoordinator,
    OAuthToken,
    RefreshCoordinator,
    TokenCacheBackend,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    CredError,
    ServerSpec,
    TokenExchangeConfig,
)

# A token with no declared expiry is cached for this long; one with an expiry is cached until then
# minus the skew buffer, floored at the minimum. Values mirror v1's MCP_OAUTH2_* constants; the
# composition root injects the configured ones.
_DEFAULT_TTL_SECONDS = 3600.0
_MIN_TTL_SECONDS = 10.0
_EXPIRY_BUFFER_SECONDS = 60.0

_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"

# The IdP returns an opaque JSON object; the post adapter hands it over untyped and the exchanger
# validates each field, so no Any leaks past this seam (None == any transport/HTTP failure).
ExchangeHttpPost = Callable[[str, "dict[str, str]"], Awaitable["dict[str, object] | None"]]


class TokenExchanger(Protocol):
    """Exchanges a caller token for an upstream-bound one, per the server's token_exchange config."""

    async def exchange(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[OAuthToken, CredError]: ...


def _cache_key(subject_token: str, config: TokenExchangeConfig) -> str:
    """Bind the cache entry to the caller token AND the exchange config that minted it.

    A rotated caller token, endpoint, audience, scope, client_id, secret, or subject_token_type all
    change the key, so a config change forces a fresh exchange instead of serving a token minted for
    the old config until TTL. Everything is hashed, so no secret is held in the key.
    """
    secret = config.client_secret.get_secret_value() if config.client_secret else ""
    material = "\x00".join(
        (
            subject_token,
            config.token_exchange_endpoint or "",
            config.audience or "",
            config.subject_token_type,
            config.client_id or "",
            secret,
            " ".join(config.scopes),
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()


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


def _build_exchange_form(
    *,
    subject_token: str,
    subject_token_type: str,
    client_id: str,
    client_secret: str,
    audience: str | None,
    scopes: tuple[str, ...],
) -> dict[str, str]:
    return {
        "grant_type": _GRANT_TYPE,
        "subject_token": subject_token,
        "subject_token_type": subject_token_type,
        "client_id": client_id,
        "client_secret": client_secret,
        **({"audience": audience} if audience else {}),
        **({"scope": " ".join(scopes)} if scopes else {}),
    }


class Rfc8693TokenExchanger:
    """``TokenExchanger`` that runs the RFC 8693 grant once per caller token, then caches the result.

    The HTTP post is injected (``None`` on any IdP failure, mirroring v1: a failed exchange is a miss,
    not a 500). The cache and single-flight coordinator default to the in-process foundation; a
    deployment with no shared state needs nothing more (v1's exchanged-token cache is per-process too).
    The clock is injected so TTL/expiry is deterministic in tests.
    """

    def __init__(
        self,
        http_post: ExchangeHttpPost,
        *,
        cache: TokenCacheBackend | None = None,
        coordinator: RefreshCoordinator | None = None,
        clock: Callable[[], float] = time.time,
        default_ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        min_ttl_seconds: float = _MIN_TTL_SECONDS,
        expiry_buffer_seconds: float = _EXPIRY_BUFFER_SECONDS,
    ) -> None:
        self._http_post = http_post
        self._cache: TokenCacheBackend = cache or InMemoryTokenCacheBackend(clock=clock)
        self._coordinator: RefreshCoordinator = coordinator or InProcessRefreshCoordinator()
        self._clock = clock
        self._default_ttl_seconds = default_ttl_seconds
        self._min_ttl_seconds = min_ttl_seconds
        self._expiry_buffer_seconds = expiry_buffer_seconds

    async def exchange(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[OAuthToken, CredError]:
        endpoint = config.token_exchange_endpoint
        client_id = config.client_id
        client_secret = config.client_secret
        if not endpoint or not client_id or client_secret is None:
            return Error(
                CredError.of_misconfigured(
                    "token_exchange requires token_exchange_endpoint, client_id and client_secret"
                )
            )

        cache_key = _cache_key(subject_token, config)
        server_id = server.server_id
        cached = await self._cache.get(cache_key, server_id)
        if cached is not None:
            return Ok(cached)

        form = _build_exchange_form(
            subject_token=subject_token,
            subject_token_type=config.subject_token_type,
            client_id=client_id,
            client_secret=client_secret.get_secret_value(),
            audience=config.audience,
            scopes=config.scopes,
        )

        async def run_exchange() -> OAuthToken | None:
            fresh = await self._cache.get(cache_key, server_id)
            if fresh is not None:
                return fresh
            body = await self._http_post(endpoint, form)
            if body is None:
                return None
            token = self._token_from_body(body)
            if token is None:
                return None
            await self._cache.set(cache_key, server_id, token, self._ttl_seconds(token))
            return token

        async def reread() -> OAuthToken | None:
            return await self._cache.get(cache_key, server_id)

        token = await self._coordinator.run(cache_key, server_id, refresh=run_exchange, reread=reread)
        if token is None:
            return Error(CredError.of_upstream_unavailable("token exchange did not return a usable access token"))
        return Ok(token)

    def _token_from_body(self, body: dict[str, object]) -> OAuthToken | None:
        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None
        expires_in = _parse_expires_in(body.get("expires_in"))
        expires_at = self._clock() + expires_in if expires_in is not None else None
        return OAuthToken(access_token=access_token, expires_at=expires_at)

    def _ttl_seconds(self, token: OAuthToken) -> float:
        if token.expires_at is None:
            return self._default_ttl_seconds
        lifetime = token.expires_at - self._clock()
        return max(lifetime - self._expiry_buffer_seconds, self._min_ttl_seconds)
