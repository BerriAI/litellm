"""Concrete `httpx2.Auth` objects the resolver returns for the self-contained modes.

These are the egress credential as the SDK consumes it: an `httpx2.Auth` attached to the
upstream `AsyncClient`. The OAuth-flow modes (`authorization_code`, `client_credentials`,
`token_exchange`) return SDK-provided auth objects instead and land later.

`auth_flow` mutating the outbound request is the `httpx2.Auth` contract, not a house-style
violation: the request is httpx2's object, and these carry no state of their own.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Final

import httpx2
from pydantic import SecretStr

from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.outbound_credentials.types import CredError, HeaderCarrier


class NoOpAuth(httpx2.Auth):
    """Attaches nothing — the `none` mode (and the seam-level default)."""

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        yield request


class StaticHeaderAuth(httpx2.Auth):
    """Sets one fixed header on every request — the `api_key` family and `passthrough`.

    The header value is a live credential (a bearer token, an API key, a forwarded user
    token), so it is held as a `SecretStr` and unwrapped only when written onto the request.
    That keeps it masked in reprs, `vars()`, tracebacks, and structured logs, matching the
    `SecretStr` discipline the config models use.
    """

    def __init__(self, header_value: str, header_name: str = "Authorization") -> None:
        self.header_name = header_name
        self._header_value = SecretStr(header_value)

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        request.headers[self.header_name] = self._header_value.get_secret_value()
        yield request


PER_USER_OAUTH_RECOVERY_EXTENSION: Final = "litellm.mcp.per_user_oauth_recovery"


class PerUserOAuthBearerAuth(httpx2.Auth):
    def __init__(
        self,
        access_token: str,
        refresh_now: Callable[[str], Awaitable[Result[OAuthToken, CredError]]],
        carrier: HeaderCarrier,
    ) -> None:
        self.header_name = carrier.header_name
        self._carrier = carrier
        self._access_token = SecretStr(access_token)
        self._refresh_now = refresh_now

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        token: Final = self._access_token.get_secret_value()
        name, value = self._carrier.header(token)
        request.headers[name] = value  # rebind-ok: httpx auth mutates the outgoing request
        response: Final = yield request
        if response.status_code != 401:
            return
        match await self._refresh_now(token):
            case Error(err):
                response.extensions[PER_USER_OAUTH_RECOVERY_EXTENSION] = err
                return
            case Ok(fresh):
                self._access_token = SecretStr(fresh.access_token)
                fresh_name, fresh_value = self._carrier.header(fresh.access_token)
                request.headers[fresh_name] = fresh_value  # rebind-ok: httpx auth mutates the retried request
                yield request

    def sync_auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        raise RuntimeError("PerUserOAuthBearerAuth only supports async httpx2 clients")
