"""Concrete `httpx.Auth` objects the resolver returns for the self-contained modes.

These are the egress credential as the SDK consumes it: an `httpx.Auth` attached to the
upstream `AsyncClient`. The OAuth-flow modes (`authorization_code`, `client_credentials`,
`token_exchange`) return SDK-provided auth objects instead and land later.

`auth_flow` mutating the outbound request is the `httpx.Auth` contract, not a house-style
violation: the request is httpx's object, and these carry no state of their own.
"""

from __future__ import annotations

from collections.abc import Generator

import httpx
from pydantic import SecretStr


class NoOpAuth(httpx.Auth):
    """Attaches nothing — the `none` mode (and the seam-level default)."""

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        yield request


class StaticHeaderAuth(httpx.Auth):
    """Sets one fixed header on every request — the `api_key` family and `passthrough`.

    The header value is a live credential (a bearer token, an API key, a forwarded user
    token), so it is held as a `SecretStr` and unwrapped only when written onto the request.
    That keeps it masked in reprs, `vars()`, tracebacks, and structured logs, matching the
    `SecretStr` discipline the config models use.
    """

    def __init__(self, header_value: str, header_name: str = "Authorization") -> None:
        self.header_name = header_name
        self._header_value = SecretStr(header_value)

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers[self.header_name] = self._header_value.get_secret_value()
        yield request
