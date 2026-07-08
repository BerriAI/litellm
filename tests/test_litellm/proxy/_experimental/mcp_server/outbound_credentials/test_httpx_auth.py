"""Tests for the concrete httpx.Auth objects the resolver returns.

NoOpAuth must attach nothing; StaticHeaderAuth must set exactly the configured header. These
pin the header emission the api_key family and passthrough depend on.
"""

import httpx

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    NoOpAuth,
    StaticHeaderAuth,
)


def _apply(auth: httpx.Auth, request: httpx.Request) -> httpx.Request:
    flow = auth.auth_flow(request)
    sent = next(flow)
    flow.close()
    return sent


def test_noop_auth_attaches_no_authorization_header():
    request = httpx.Request("GET", "https://upstream.example.com/mcp")
    _apply(NoOpAuth(), request)
    assert "authorization" not in request.headers


def test_static_header_auth_defaults_to_authorization():
    request = httpx.Request("GET", "https://upstream.example.com/mcp")
    _apply(StaticHeaderAuth("Bearer abc"), request)
    assert request.headers["Authorization"] == "Bearer abc"


def test_static_header_auth_honors_custom_header_name():
    request = httpx.Request("GET", "https://upstream.example.com/mcp")
    _apply(StaticHeaderAuth("raw-key", header_name="X-API-Key"), request)
    assert request.headers["X-API-Key"] == "raw-key"
    assert "authorization" not in request.headers


def test_static_header_auth_masks_credential_from_introspection():
    auth = StaticHeaderAuth("Bearer super-secret-token")
    assert "super-secret-token" not in repr(auth)
    assert "super-secret-token" not in str(vars(auth))
