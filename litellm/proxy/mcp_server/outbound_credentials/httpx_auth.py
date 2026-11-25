"""Concrete `httpx2.Auth` objects the resolver returns for the self-contained modes.

These are the egress credential as the SDK consumes it: an `httpx2.Auth` attached to the
upstream `AsyncClient`. The OAuth-flow modes (`authorization_code`, `client_credentials`,
`token_exchange`) return SDK-provided auth objects instead and land later.

`auth_flow` mutating the outbound request is the `httpx2.Auth` contract, not a house-style
violation: the request is httpx2's object, and these hold no mutable state.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Final
from urllib.request import parse_http_list

import httpx2
from pydantic import SecretStr

from litellm._logging import verbose_logger
from litellm.proxy.mcp_server.outbound_credentials.types import HeaderCarrier


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


def _challenge_error(response: httpx2.Response) -> str | None:
    _, _, params = (response.headers.get("www-authenticate") or "").partition(" ")
    fields: Final = (field.partition("=") for field in parse_http_list(params))
    return next((value.strip().strip('"') for key, _, value in fields if key.strip().lower() == "error"), None)


class UserGrantBearerAuth(httpx2.Auth):
    def __init__(
        self, access_token: str, carrier: HeaderCarrier, on_rejected: Callable[[str, str | None], Awaitable[None]]
    ) -> None:
        self._carrier = carrier
        self.header_name = carrier.header_name
        self._access_token = SecretStr(access_token)
        self._on_rejected = on_rejected

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        self._attach(request)
        yield request

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        token: Final = self._attach(request)
        response: Final = yield request
        sent: Final = request.headers.get(self.header_name)
        if response.status_code != 401 or response.request.headers.get(self.header_name) != sent:
            return  # a cross-origin redirect drops the credential, so its 401 says nothing about the token
        try:
            await self._on_rejected(token, _challenge_error(response))
        except Exception as exc:  # noqa: BLE001  # a failed report must not replace the upstream's 401
            verbose_logger.warning("MCP rejected OAuth token was not recorded: %s", type(exc).__name__)

    def _attach(self, request: httpx2.Request) -> str:
        token: Final = self._access_token.get_secret_value()
        name, value = self._carrier.header(token)
        request.headers[name] = value  # rebind-ok: an httpx2.Auth sets the header on the outbound request
        return token
