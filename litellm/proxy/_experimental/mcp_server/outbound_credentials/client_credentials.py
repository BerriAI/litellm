"""The ``client_credentials`` (M2M) arm's token source and retrying bearer auth.

Implements the client-credentials behavior contract for the v2 resolver:

- **Acquisition**: POST ``grant_type=client_credentials`` to the configured token endpoint with
  the configured scopes and (when set) the IdP's ``audience`` parameter, authenticating the
  client per ``token_endpoint_auth_method`` (RFC 6749 section 2.3.1, shared helper).
- **Caching**: tokens are cached per ``(client identity, server)`` where the identity key hashes
  ``token_url`` / ``client_id`` / ``client_secret`` / auth method / scopes / audience — rotating
  or re-scoping the credentials changes the key, so a stale token can never be served for the
  new identity (the contract's rotation-invalidation clause).
- **Expiry**: the cache TTL respects ``expires_in`` minus a skew so an entry lapses before the
  real token does; a response with no ``expires_in`` is cached briefly
  (``default_ttl_seconds``), not assumed long-lived. No refresh_token is ever expected.
- **401 recovery**: ``ClientCredentialsBearerAuth`` retries an upstream request exactly once
  after a 401 — discard the cached token, mint a fresh one, resend; a second failure surfaces
  the upstream's own auth error unchanged.
- **No user context**: nothing here reads a ``Subject``; every caller shares the one client
  identity.

The token-endpoint POST is injected (``M2MTokenEndpointPost``) so the grant orchestration is
testable without a live IdP; ``post_client_credentials_grant`` is the httpx edge and the one
place the untyped response boundary is contained. Failures are values: the source returns
``Result[OAuthToken, CredError]``; only the httpx edge touches exceptions.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InMemoryTokenCacheBackend,
    OAuthToken,
    TokenCacheBackend,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ClientCredentialsConfig,
    CredError,
)


class TokenEndpointSuccess(BaseModel):
    """The endpoint returned a JSON object; field validation is the caller's job."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["success"] = "success"
    body: dict[str, object]


class TokenEndpointDenied(BaseModel):
    """The endpoint answered but did not grant a token (an HTTP error or a non-JSON body)."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["denied"] = "denied"
    status_code: int
    detail: str


class TokenEndpointUnreachable(BaseModel):
    """The endpoint could not be reached (DNS, TLS, connect/read failure)."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["unreachable"] = "unreachable"
    detail: str


TokenEndpointOutcome = Annotated[
    TokenEndpointSuccess | TokenEndpointDenied | TokenEndpointUnreachable,
    Field(discriminator="tag"),
]

M2MTokenEndpointPost = Callable[[str, "dict[str, str]", "dict[str, str]"], Awaitable[TokenEndpointOutcome]]


_TOKEN_BODY_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


async def post_client_credentials_grant(
    url: str, form: dict[str, str], headers: dict[str, str]
) -> TokenEndpointOutcome:
    """POST the grant to the token endpoint and classify the transport outcome.

    The httpx edge: litellm's handler is partially typed (and raises ``HTTPStatusError`` itself on
    a 4xx/5xx), so the untyped boundary is contained here and every field the caller reads comes
    out of a validated ``TokenEndpointOutcome``.
    """
    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415  # defer heavy handler import to call time
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # handler is partially typed
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415  # deferred with the handler import

    try:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        response = await client.post(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # handler is partially typed
            url, headers={"Accept": "application/json", **headers}, data=form
        )
    except httpx.HTTPStatusError as status_err:
        status_code = status_err.response.status_code
        return TokenEndpointDenied(status_code=status_code, detail=f"token endpoint returned HTTP {status_code}")
    except Exception as exc:  # noqa: BLE001  # any transport failure is the same outcome: unreachable
        return TokenEndpointUnreachable(detail=str(exc))
    if not isinstance(response, httpx.Response):
        return TokenEndpointUnreachable(detail="token endpoint returned no response")
    try:
        body = _TOKEN_BODY_ADAPTER.validate_json(response.content)
    except ValidationError:
        return TokenEndpointDenied(
            status_code=response.status_code, detail="token endpoint returned a non-JSON-object body"
        )
    return TokenEndpointSuccess(body=body)


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


def _parse_granted_scopes(raw: object) -> tuple[str, ...] | None:
    return tuple(raw.split()) if isinstance(raw, str) and raw else None


@dataclass(frozen=True, slots=True)
class _PreparedGrant:
    """A validated, ready-to-POST grant plus the identity key its token caches under."""

    token_url: str
    form: dict[str, str]
    headers: dict[str, str]
    identity_key: str


class ClientCredentialsTokenSource:
    """Cached M2M access tokens, one per ``(client identity, server)``.

    ``get`` serves from the cache while the entry's TTL (derived from ``expires_in`` minus
    ``expiry_skew_seconds``) holds, fetching under a per-server lock so concurrent misses
    produce one grant. ``refetch`` is the 401-recovery path: it drops the failed token and
    mints a fresh one, unless a concurrent caller already replaced it.
    """

    def __init__(
        self,
        post: M2MTokenEndpointPost = post_client_credentials_grant,
        *,
        backend: TokenCacheBackend | None = None,
        default_ttl_seconds: float = 300.0,
        expiry_skew_seconds: float = 60.0,
        min_cache_seconds: float = 10.0,
        max_locks: int = 1024,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._post = post
        self._backend: TokenCacheBackend = backend or InMemoryTokenCacheBackend(clock=clock)
        self._default_ttl_seconds = default_ttl_seconds
        self._expiry_skew_seconds = expiry_skew_seconds
        self._min_cache_seconds = min_cache_seconds
        self._max_locks = max_locks
        self._clock = clock
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, server_id: str) -> asyncio.Lock:
        """Per-server single-flight lock, bounded so ephemeral server ids (e.g. the REST tools
        preview mints a fresh id per call) cannot grow the dict for the life of the process.
        Evicting the oldest entry while a task still holds it only means a concurrent caller for
        that server may run its own grant — single-flight is an optimization, not correctness.
        """
        if server_id not in self._locks and len(self._locks) >= self._max_locks:
            self._locks.pop(next(iter(self._locks)), None)
        return self._locks.setdefault(server_id, asyncio.Lock())

    async def get(self, server_id: str, config: ClientCredentialsConfig) -> Result[OAuthToken, CredError]:
        match _prepare_grant(config):
            case Error(err):
                return Error(err)
            case Ok(grant):
                cached = await self._backend.get(grant.identity_key, server_id)
                if cached is not None:
                    return Ok(cached)
                async with self._lock(server_id):
                    cached = await self._backend.get(grant.identity_key, server_id)
                    if cached is not None:
                        return Ok(cached)
                    return await self._fetch_and_cache(server_id, grant)

    async def refetch(self, server_id: str, config: ClientCredentialsConfig, failed_access_token: str) -> str | None:
        """Replace a token the upstream just 401'd; returns the fresh bearer value or ``None``.

        Runs under the same per-server lock as ``get``: if a concurrent caller already replaced
        the failed token, that replacement is returned without another grant, so a burst of 401s
        yields one fetch. A failed refetch returns ``None`` and the caller surfaces the
        upstream's original auth error (the contract's retry-once-then-give-up clause).
        """
        match _prepare_grant(config):
            case Error(_):
                return None
            case Ok(grant):
                async with self._lock(server_id):
                    cached = await self._backend.get(grant.identity_key, server_id)
                    if cached is not None and cached.access_token != failed_access_token:
                        return cached.access_token
                    await self._backend.delete(grant.identity_key, server_id)
                    match await self._fetch_and_cache(server_id, grant):
                        case Ok(token):
                            return token.access_token
                        case Error(_):
                            return None

    async def _fetch_and_cache(self, server_id: str, grant: _PreparedGrant) -> Result[OAuthToken, CredError]:
        outcome = await self._post(grant.token_url, grant.form, grant.headers)
        match outcome:
            case TokenEndpointUnreachable():
                return Error(CredError.of_upstream_unavailable(f"OAuth2 token endpoint unreachable: {outcome.detail}"))
            case TokenEndpointDenied():
                if outcome.status_code >= 500:
                    return Error(CredError.of_upstream_unavailable(f"OAuth2 token endpoint failed: {outcome.detail}"))
                return Error(CredError.of_misconfigured(f"OAuth2 client_credentials grant rejected: {outcome.detail}"))
            case TokenEndpointSuccess():
                return await self._cache_token(server_id, grant, outcome.body)
        assert_never(outcome)

    async def _cache_token(
        self, server_id: str, grant: _PreparedGrant, body: dict[str, object]
    ) -> Result[OAuthToken, CredError]:
        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return Error(CredError.of_misconfigured("OAuth2 token response is missing 'access_token'"))
        expires_in = _parse_expires_in(body.get("expires_in"))
        token = OAuthToken(
            access_token=access_token,
            expires_at=self._clock() + expires_in if expires_in is not None else None,
            scopes=_parse_granted_scopes(body.get("scope")) or (),
        )
        # The min-cache floor is itself capped at the token's real lifetime, so a token whose
        # expires_in is below the skew is never served past its actual expiry; a non-positive
        # expires_in caches nothing (every request re-fetches, serialized by the per-server lock).
        ttl = (
            max(expires_in - self._expiry_skew_seconds, min(float(expires_in), self._min_cache_seconds), 0.0)
            if expires_in is not None
            else self._default_ttl_seconds
        )
        if ttl > 0:
            await self._backend.set(grant.identity_key, server_id, token, ttl)
        return Ok(token)


def _prepare_grant(config: ClientCredentialsConfig) -> Result[_PreparedGrant, CredError]:
    if not config.client_id or not config.client_secret or not config.token_url:
        missing = ", ".join(
            name
            for name, present in (
                ("client_id", bool(config.client_id)),
                ("client_secret", bool(config.client_secret)),
                ("token_url", bool(config.token_url)),
            )
            if not present
        )
        return Error(CredError.of_misconfigured(f"client_credentials config is missing: {missing}"))

    from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (  # noqa: PLC0415  # keep package v1-free at import time
        build_token_endpoint_client_auth,
    )

    client_auth = build_token_endpoint_client_auth(
        auth_method=config.token_endpoint_auth_method,
        client_id=config.client_id,
        client_secret=config.client_secret.get_secret_value(),
    )
    form = {
        "grant_type": "client_credentials",
        **client_auth.body,
        **({"scope": " ".join(config.scopes)} if config.scopes else {}),
        **({"audience": config.audience} if config.audience else {}),
    }
    return Ok(
        _PreparedGrant(
            token_url=config.token_url,
            form=form,
            headers=client_auth.headers,
            identity_key=_identity_key(config),
        )
    )


def _identity_key(config: ClientCredentialsConfig) -> str:
    """Hash of everything that names the client identity; any rotation yields a new key."""
    material = "\n".join(
        (
            config.token_url or "",
            config.client_id or "",
            config.client_secret.get_secret_value() if config.client_secret else "",
            config.token_endpoint_auth_method or "",
            " ".join(config.scopes),
            config.audience or "",
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ClientCredentialsBearerAuth(httpx.Auth):
    """Bearer auth that retries an upstream 401 exactly once with a freshly minted token.

    The initial token was already resolved (so config/IdP failures surfaced as typed errors
    before any upstream request); ``refetch`` is the source's 401-recovery callback. If the
    refetch fails, or the retried request 401s again, the upstream's response stands.
    """

    def __init__(self, access_token: str, refetch: Callable[[str], Awaitable[str | None]]) -> None:
        self.header_name = "Authorization"
        self._access_token = SecretStr(access_token)
        self._refetch = refetch

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = self._access_token.get_secret_value()
        request.headers[self.header_name] = f"Bearer {token}"
        response = yield request
        if response.status_code != 401:
            return
        fresh = await self._refetch(token)
        if fresh is None:
            return
        request.headers[self.header_name] = f"Bearer {fresh}"
        yield request

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        raise RuntimeError("ClientCredentialsBearerAuth only supports async httpx clients")
