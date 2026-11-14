"""The HTTP side of a WebSocket upgrade on the realtime routes.

Realtime sessions start as an HTTP/1.1 `GET` with `Connection: Upgrade` and `Upgrade: websocket` on
`/v1/realtime`, `/realtime` or `/openai/v1/realtime`. The proxy authenticates before it accepts: a caller without
a valid key is refused during the handshake with an HTTP 403 and never gets `101 Switching Protocols`. A Rust
forwarder treats `Connection` and `Upgrade` as hop-by-hop headers for ordinary requests but must relay them, and
every `Sec-WebSocket-*` header, for an upgrade (`litellm-rust/crates/gateway-forward/src/rewrite.rs`). These
tests speak the handshake over a plain socket so the status line and headers are observed exactly. Session
behaviour after the upgrade is pinned in `tests/integration/providers/test_openai_realtime_transcription_url.py`.
"""

from __future__ import annotations

import base64
import hashlib
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.upstream import delete_scenario, register_scenario
from integration.compatibility.dec1_http_support import (
    RawResponse,
    connection,
    host_header,
    read_response,
    request_bytes,
)

from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

WEBSOCKET_GUID: Final = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
REALTIME_PATHS: Final = ("/v1/realtime", "/realtime", "/openai/v1/realtime")


def _nonce() -> str:
    return base64.b64encode(os.urandom(16)).decode()


def _accept_for(nonce: str) -> str:
    return base64.b64encode(hashlib.sha1(f"{nonce}{WEBSOCKET_GUID}".encode(), usedforsecurity=False).digest()).decode()


def _upgrade(
    gateway: Gateway, target: str, nonce: str, key: str | None, extra: tuple[tuple[str, str], ...] = ()
) -> RawResponse:
    headers: Final = (
        ("Host", host_header(gateway)),
        ("Connection", "Upgrade"),
        ("Upgrade", "websocket"),
        ("Sec-WebSocket-Version", "13"),
        ("Sec-WebSocket-Key", nonce),
        *((("Authorization", f"Bearer {key}"),) if key is not None else ()),
        *extra,
    )
    with connection(gateway) as (client, stream):
        client.sendall(request_bytes("GET", target, headers))
        return read_response(stream, bodiless=True)


@contextmanager
def _realtime_model(gateway: Gateway) -> Iterator[str]:
    """A model name served by a scripted realtime provider for the length of one test."""
    scenario_id: Final = f"dec1-http-realtime-{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(scenario_id, RealtimeResponse(content_type="application/x-realtime", events=()))
    try:
        with gateway.scenario() as scenario:
            yield scenario.model(
                model="openai/gpt-realtime-2.1", api_key=scenario_id, api_base=gateway.upstream_url.rstrip("/")
            )
    finally:
        delete_scenario(handle)


@pytest.mark.parametrize("path", REALTIME_PATHS)
def test_an_upgrade_without_a_key_is_refused_with_403_and_no_101(gateway: Gateway, path: str) -> None:
    """A WebSocket handshake is sent to a realtime route with no key at all. The proxy must answer the handshake
    with HTTP 403 and must not answer 101: authentication happens before the socket is accepted, so an
    unauthenticated caller never holds an open WebSocket."""
    response: Final = _upgrade(gateway, f"{path}?model=dec1-any", _nonce(), None)
    assert response.status == 403


def test_an_upgrade_with_an_unknown_key_is_refused_with_403(gateway: Gateway) -> None:
    """A WebSocket handshake is sent to `/v1/realtime` with a well-formed but unknown `sk-` key. The proxy must
    answer HTTP 403 and send no `Sec-WebSocket-Accept` header. Note the status differs from the 401 the same key
    gets on a plain HTTP route: a refused upgrade is always 403."""
    response: Final = _upgrade(gateway, "/v1/realtime?model=dec1-any", _nonce(), f"sk-dec1-{uuid.uuid4().hex}")
    assert response.status == 403
    assert response.header("sec-websocket-accept") is None


def test_a_plain_get_on_the_realtime_route_is_not_found(gateway: Gateway) -> None:
    """The master key sends an ordinary `GET /v1/realtime` with no upgrade headers. The proxy must answer 404
    with `{"detail": "Not Found"}`: the route exists only for WebSocket connections. A forwarder that strips
    `Connection: Upgrade` and `Upgrade` from a handshake would turn every realtime session into this 404."""
    response: Final = gateway.request("GET", "/v1/realtime", params={"model": "dec1-any"})
    assert (response.status_code, response.json()) == (404, {"detail": "Not Found"})


def test_an_upgrade_on_a_plain_http_route_is_refused_with_403(gateway: Gateway) -> None:
    """A WebSocket handshake with the master key is sent to `/v1/chat/completions`, which is not a WebSocket
    route. The proxy must refuse the handshake with HTTP 403 and never answer 101. UNVERIFIED: the web framework
    closes an unrouted WebSocket before accepting, which the server reports as 403."""
    response: Final = _upgrade(gateway, "/v1/chat/completions", _nonce(), gateway.key)
    assert response.status == 403


@pytest.mark.parametrize("path", REALTIME_PATHS)
def test_a_valid_upgrade_is_answered_101_with_the_matching_accept_token(gateway: Gateway, path: str) -> None:
    """The master key sends a WebSocket handshake for a realtime model whose provider is a scripted realtime
    peer. On each of the three route aliases the proxy must answer `101 Switching Protocols` with
    `Upgrade: websocket`, a `Connection` header containing `upgrade`, and a `Sec-WebSocket-Accept` equal to the
    base64 SHA-1 of the client's `Sec-WebSocket-Key` plus the protocol GUID. A forwarder that rewrote the key or
    answered the upgrade itself would produce a different token and the client would drop the connection."""
    nonce: Final = _nonce()
    with _realtime_model(gateway) as model:
        response: Final = _upgrade(gateway, f"{path}?model={model}", nonce, gateway.key)
    assert response.status == 101, response
    assert (response.header("upgrade") or "").lower() == "websocket"
    assert "upgrade" in (response.header("connection") or "").lower()
    assert response.header("sec-websocket-accept") == _accept_for(nonce)


def test_a_key_in_the_api_key_header_authenticates_an_upgrade(gateway: Gateway) -> None:
    """A WebSocket handshake to `/v1/realtime` carries the master key in an `api-key` header and has no
    `Authorization` header, as Azure-style realtime clients send it. The handshake must be answered 101.
    UNVERIFIED: assumes the WebSocket auth path reads `api-key` the same way the HTTP path does. The Rust
    forwarder lists `api-key` among the headers it always forwards, so both layers must agree."""
    with _realtime_model(gateway) as model:
        response: Final = _upgrade(
            gateway, f"/v1/realtime?model={model}", _nonce(), None, extra=(("api-key", gateway.key),)
        )
    assert response.status == 101, response


def test_an_upgrade_without_a_websocket_key_is_not_answered_101(gateway: Gateway) -> None:
    """A request to `/v1/realtime` carries `Connection: Upgrade` and `Upgrade: websocket` with the master key but
    no `Sec-WebSocket-Key` or `Sec-WebSocket-Version`. The proxy must answer with a 4xx status and never 101: a
    malformed handshake cannot open a session. UNVERIFIED: the exact 4xx status depends on the server's WebSocket
    implementation (400 is expected)."""
    headers: Final = (
        ("Host", host_header(gateway)),
        ("Connection", "Upgrade"),
        ("Upgrade", "websocket"),
        ("Authorization", f"Bearer {gateway.key}"),
    )
    with connection(gateway) as (client, stream):
        client.sendall(request_bytes("GET", "/v1/realtime?model=dec1-any", headers))
        response: Final = read_response(stream, bodiless=True)
    assert 400 <= response.status < 500
