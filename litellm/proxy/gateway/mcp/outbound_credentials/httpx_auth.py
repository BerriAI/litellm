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


class NoOpAuth(httpx.Auth):
    """Attaches nothing — the `none` mode (and the seam-level default)."""

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        yield request


class StaticHeaderAuth(httpx.Auth):
    """Sets one fixed header on every request — the `api_key` family and `passthrough`."""

    def __init__(self, header_value: str, header_name: str = "Authorization") -> None:
        self.header_name = header_name
        self.header_value = header_value

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers[self.header_name] = self.header_value
        yield request
