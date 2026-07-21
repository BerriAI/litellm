"""An authenticated OAuth token-endpoint call plus a short-lived-token cache.

`TokenEndpointClient.fetch` POSTs one grant to a token endpoint, authenticating the gateway as
an OAuth client via `client_auth` (RFC 7523 private-key JWT, or `client_secret_post`), and returns
the minted token or a typed `CredError`. `ExchangedTokenCache` memoizes the final token string per
opaque cache key with per-key single-flight, so concurrent callers share one round-trip and a hit
skips the endpoint entirely.

Pure v2: no imports from the v1 MCP auth handlers. The multi-leg flows that compose these (ID-JAG,
and later token_exchange / client_credentials) live in the resolver arms; this collaborator owns
only the single authenticated call and the cache.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

import httpx
import jwt
from pydantic import BaseModel, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
    MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
)
from litellm.exceptions import Timeout
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # litellm http handler is untyped
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ClientAuth,
    ClientSecretAuth,
    CredError,
    PrivateKeyJwtAuth,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
CLIENT_ASSERTION_LIFETIME_SECONDS = 60


@dataclass(frozen=True, slots=True)
class ExchangedToken:
    access_token: str
    expires_in: int | None


class _TokenEndpointResponse(BaseModel):
    access_token: str
    expires_in: int | None = None


class TokenEndpointClient:
    """One authenticated POST to an OAuth token endpoint, returning the minted token as a value."""

    async def fetch(
        self,
        endpoint: str,
        client_id: str,
        grant_params: Mapping[str, str],
        client_auth: ClientAuth,
    ) -> Result[ExchangedToken, CredError]:
        try:
            data = {**grant_params, **_client_auth_params(endpoint, client_id, client_auth)}
        except (ValueError, TypeError, NotImplementedError, jwt.PyJWTError):
            verbose_proxy_logger.warning("MCP token endpoint %s: could not sign the client assertion", endpoint)
            return Error(
                CredError.of_misconfigured(
                    "token exchange failed: could not sign the client assertion; "
                    "check client_private_key and client_assertion_signing_alg"
                )
            )
        try:
            raw = await _post_form(endpoint, data)
        except httpx.HTTPStatusError as exc:
            verbose_proxy_logger.warning(
                "MCP token endpoint %s failed with status %s", endpoint, exc.response.status_code
            )
            return Error(
                CredError.of_upstream_unavailable(f"token exchange failed with status {exc.response.status_code}")
            )
        except (httpx.RequestError, Timeout) as exc:
            verbose_proxy_logger.warning("MCP token endpoint %s unreachable: %s", endpoint, type(exc).__name__)
            return Error(
                CredError.of_upstream_unavailable(
                    f"token exchange failed: token endpoint unreachable ({type(exc).__name__})"
                )
            )
        except json.JSONDecodeError:
            verbose_proxy_logger.warning("MCP token endpoint %s returned a non-JSON response", endpoint)
            return Error(
                CredError.of_upstream_unavailable("token exchange failed: token endpoint returned a non-JSON response")
            )
        if raw is None:
            verbose_proxy_logger.warning("MCP token endpoint %s returned no response", endpoint)
            return Error(CredError.of_upstream_unavailable("token exchange failed: no response from token endpoint"))
        try:
            parsed = _TokenEndpointResponse.model_validate(raw)
        except ValidationError:
            verbose_proxy_logger.warning("MCP token endpoint %s response missing access_token", endpoint)
            return Error(
                CredError.of_upstream_unavailable("token exchange failed: token endpoint response missing access_token")
            )
        return Ok(ExchangedToken(access_token=parsed.access_token, expires_in=parsed.expires_in))


class ExchangedTokenCache:
    """Memoizes the final token string per key, single-flighting concurrent misses on one lock."""

    def __init__(self) -> None:
        self._cache = InMemoryCache(
            max_size_in_memory=MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
            default_ttl=MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
        )
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    async def get_or_compute(
        self,
        cache_key: str,
        compute: Callable[[], Awaitable[Result[ExchangedToken, CredError]]],
    ) -> Result[str, CredError]:
        cached = self._get(cache_key)
        if cached is not None:
            return Ok(cached)
        async with self._lock(cache_key):
            cached = self._get(cache_key)
            if cached is not None:
                return Ok(cached)
            match await compute():
                case Ok(token):
                    self._cache.set_cache(  # pyright: ignore[reportUnknownMemberType]  # InMemoryCache is untyped
                        cache_key,
                        token.access_token,
                        ttl=_cache_ttl_seconds(token.expires_in),
                    )
                    return Ok(token.access_token)
                case Error(err):
                    return Error(err)

    def invalidate(self, cache_key: str) -> None:
        """Evict one cached token so the next `get_or_compute` re-mints (e.g. after an upstream 401)."""
        self._cache.delete_cache(cache_key)  # pyright: ignore[reportUnknownMemberType]  # InMemoryCache is untyped

    def _get(self, cache_key: str) -> str | None:
        value = self._cache.get_cache(cache_key)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # InMemoryCache is untyped; narrowed by isinstance below
        return value if isinstance(value, str) else None

    def _lock(self, cache_key: str) -> asyncio.Lock:
        lock = self._locks.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[cache_key] = lock
        return lock


def _cache_ttl_seconds(expires_in: int | None) -> int:
    lifetime = expires_in if expires_in is not None else MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL
    return max(
        lifetime - MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
        MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    )


async def _post_form(endpoint: str, data: dict[str, str]) -> object | None:
    # litellm's httpx handler and httpx.Response are only partially typed; the token endpoint
    # returns a JSON object that `_TokenEndpointResponse` validates, so the untyped boundary is
    # contained here. A non-2xx raises `httpx.HTTPStatusError`, an unreachable endpoint raises
    # `httpx.RequestError` (or litellm's `Timeout`, which the handler substitutes for
    # `httpx.TimeoutException`), and a non-JSON body raises `json.JSONDecodeError`; `fetch` maps
    # each to a CredError.
    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)  # pyright: ignore[reportUnknownVariableType]  # litellm http handler is untyped
    response = await client.post(endpoint, data=data)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # litellm http handler is untyped
    if response is None:
        return None
    response.raise_for_status()
    return response.json()  # pyright: ignore[reportAny]  # untyped JSON; validated by _TokenEndpointResponse in fetch


def _client_auth_params(endpoint: str, client_id: str, client_auth: ClientAuth) -> dict[str, str]:
    match client_auth:
        case PrivateKeyJwtAuth() as auth:
            return {
                "client_id": client_id,
                "client_assertion_type": CLIENT_ASSERTION_TYPE,
                "client_assertion": _client_assertion(endpoint, client_id, auth),
            }
        case ClientSecretAuth() as auth:
            return {
                "client_id": client_id,
                "client_secret": auth.client_secret.get_secret_value(),
            }
    assert_never(client_auth)


def _client_assertion(endpoint: str, client_id: str, auth: PrivateKeyJwtAuth) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": client_id,
            "sub": client_id,
            "aud": endpoint,
            "jti": uuid.uuid4().hex,
            "iat": now,
            "exp": now + CLIENT_ASSERTION_LIFETIME_SECONDS,
        },
        auth.private_key.get_secret_value(),
        algorithm=auth.signing_alg,
        headers={"kid": auth.key_id} if auth.key_id else None,
    )
