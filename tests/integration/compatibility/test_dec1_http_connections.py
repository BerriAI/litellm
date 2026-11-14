"""Connection-level HTTP/1.x behaviour of the listening socket, driven with raw bytes.

uvicorn answers the public port today. These tests pin what a client observes below the level of routes:
persistent connections and pipelining, `Connection: close`, HTTP/1.0 clients, how a stream is framed for each
protocol version, how request targets are decoded, and what malformed requests get back. The Rust forwarder
(`litellm-rust/crates/gateway-forward`) terminates client connections with hyper and re-sends each request to
Python, dropping hop-by-hop headers on the way (`rewrite.rs`), so every case here crosses both parsers once the
forwarder is in front. Its own notes (`FORWARDING.md`) list where hyper and uvicorn's h11 differ; the cases
below are the ones callers depend on. Nothing here needs a provider except the two streaming cases.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration.compatibility.dec1_http_support import (
    CHAT,
    chat_body,
    connection,
    exchange,
    front_door,
    host_header,
    json_post,
    peer_closed,
    read_response,
    request_bytes,
)

ALIVE: Final = b'"I\'m alive!"'
LIVELINESS: Final = "/health/liveliness"


def _get(gateway: Gateway, target: str, *extra: tuple[str, str], version: str = "HTTP/1.1") -> bytes:
    return request_bytes("GET", target, (("Host", host_header(gateway)), *extra), version=version)


def test_two_requests_are_served_on_one_persistent_connection(gateway: Gateway) -> None:
    """The caller opens one TCP connection and sends a liveness GET, reads the answer, then sends a second one
    on the same socket. Both must be answered 200 with the JSON string "I'm alive!", and the first response must
    not announce `connection: close`. HTTP/1.1 connections are persistent unless a side says otherwise."""
    with connection(gateway) as (client, stream):
        client.sendall(_get(gateway, LIVELINESS))
        first: Final = read_response(stream)
        client.sendall(_get(gateway, LIVELINESS))
        second: Final = read_response(stream)
    assert [(first.status, first.body), (second.status, second.body)] == [(200, ALIVE), (200, ALIVE)]
    assert first.header("connection") != "close"


def test_two_pipelined_requests_are_answered_in_the_order_they_were_sent(gateway: Gateway) -> None:
    """The caller writes a liveness GET and a readiness GET back to back in one `send`, before reading anything.
    The proxy must answer both on that connection in order: first the JSON string "I'm alive!", then the
    readiness object `{"status": "healthy", "db": "connected"}`."""
    payload: Final = _get(gateway, LIVELINESS) + _get(gateway, "/health/readiness", ("Connection", "close"))
    first, second = exchange(gateway, payload, responses=2)
    assert (first.status, first.body) == (200, ALIVE)
    assert (second.status, second.json()) == (200, {"status": "healthy", "db": "connected"})


def test_a_slow_post_pipelined_before_a_fast_get_is_still_answered_first(gateway: Gateway) -> None:
    """A chat model is registered. The caller pipelines a chat POST, which needs a provider round trip, followed
    immediately by a liveness GET. The chat completion must come back first and the liveness answer second:
    responses on one HTTP/1.1 connection are never reordered, however long the first takes."""
    with front_door(gateway) as door:
        body: Final = json.dumps(chat_body(door.chat_model())).encode()
        payload: Final = json_post(gateway, CHAT, body) + _get(gateway, LIVELINESS, ("Connection", "close"))
        first, second = exchange(gateway, payload, responses=2)
        assert first.status == 200, first.body
        assert first.json()["object"] == "chat.completion"
        assert (second.status, second.body) == (200, ALIVE)


def test_connection_close_is_honoured_and_announced(gateway: Gateway) -> None:
    """The caller sends a liveness GET with `Connection: close`. The 200 response must carry `connection: close`
    and the server must then close the socket, so the caller reads end-of-file instead of waiting on an idle
    connection."""
    with connection(gateway) as (client, stream):
        client.sendall(_get(gateway, LIVELINESS, ("Connection", "close")))
        response: Final = read_response(stream)
        closed: Final = peer_closed(client, stream)
    assert (response.status, response.body) == (200, ALIVE)
    assert response.header("connection") == "close"
    assert closed


def test_an_http_1_0_client_gets_its_answer_and_then_the_connection_closes(gateway: Gateway) -> None:
    """The caller sends `GET /health/liveliness HTTP/1.0` with no keep-alive request. The proxy must answer 200
    with the complete body and a `content-length`, then close the connection, because HTTP/1.0 connections are
    not persistent by default. Old monitoring agents and some health checkers still speak HTTP/1.0."""
    with connection(gateway) as (client, stream):
        client.sendall(_get(gateway, LIVELINESS, version="HTTP/1.0"))
        response: Final = read_response(stream)
        closed: Final = peer_closed(client, stream)
    assert (response.status, response.body) == (200, ALIVE)
    assert response.header("content-length") == str(len(ALIVE))
    assert closed


def test_an_http_1_0_client_without_a_host_header_is_served(gateway: Gateway) -> None:
    """The caller sends `GET /health/liveliness HTTP/1.0` with no headers at all. HTTP/1.0 does not require
    `Host`, so the proxy must answer 200 with the liveness body. The forwarder fills in the public address it
    was reached on for such requests (`public_host`, `rewrite.rs`) instead of rejecting them."""
    (response,) = exchange(gateway, request_bytes("GET", LIVELINESS, (), version="HTTP/1.0"))
    assert (response.status, response.body) == (200, ALIVE)


def test_an_http_1_1_request_without_a_host_header_is_served_today(gateway: Gateway) -> None:
    """The caller sends `GET /health/liveliness HTTP/1.1` with no `Host` header. HTTP/1.1 makes Host mandatory and
    the forwarder answers 400 itself (`Rejection::MissingHost`).

    Pinned current behaviour: the proxy installs `httptools` (pyproject `httptools>=0.7.1`), so uvicorn's `auto`
    HTTP implementation parses with httptools, which does not enforce Host, and the liveness handler answers 200
    with its usual body. Putting the forwarder in front turns this into a 400, and this test then has to change
    on purpose."""
    (response,) = exchange(gateway, request_bytes("GET", LIVELINESS, (("Connection", "close"),)))
    assert (response.status, response.body) == (200, ALIVE)


def test_bytes_that_are_not_http_get_a_400_invalid_request(gateway: Gateway) -> None:
    """The caller connects and sends a line that is not an HTTP request line. The proxy must answer 400 with the
    plain-text body `Invalid HTTP request received.` and close the connection. Port scanners and misdirected TLS
    clients produce this constantly, and it must never reach application code."""
    with connection(gateway) as (client, stream):
        client.sendall(b"THIS IS NOT HTTP\r\n\r\n")
        response: Final = read_response(stream)
        closed: Final = peer_closed(client, stream)
    assert response.status == 400
    assert response.body == b"Invalid HTTP request received."
    assert closed


def test_a_transfer_coding_other_than_chunked_is_a_400(gateway: Gateway) -> None:
    """The caller posts to the chat path with `Transfer-Encoding: gzip`. Only `chunked` is a transfer coding the
    server can decode, so the proxy must answer 400 and must not treat the bytes as a JSON body. The forwarder
    rejects the same request itself (`only_chunked`, `rewrite.rs`) so Python never sees a body it would misframe."""
    payload: Final = request_bytes(
        "POST",
        CHAT,
        (
            ("Host", host_header(gateway)),
            ("Authorization", f"Bearer {gateway.key}"),
            ("Content-Type", "application/json"),
            ("Transfer-Encoding", "gzip"),
            ("Connection", "close"),
        ),
        b"{}",
    )
    (response,) = exchange(gateway, payload)
    assert response.status == 400, response.body


def test_a_json_response_on_http_1_1_is_length_framed_and_dated(gateway: Gateway) -> None:
    """The caller reads the raw liveness response. It must be `HTTP/1.1 200 OK` with a `content-length` equal to
    the body size, a `date` header, `content-type: application/json` and no `transfer-encoding`. Each header
    name must appear once."""
    (response,) = exchange(gateway, _get(gateway, LIVELINESS, ("Connection", "close")))
    assert (response.version, response.status, response.reason) == ("HTTP/1.1", 200, "OK")
    assert response.values("content-length") == (str(len(ALIVE)),)
    assert response.values("content-type") == ("application/json",)
    assert len(response.values("date")) == 1
    assert response.values("transfer-encoding") == ()


def test_a_stream_to_an_http_1_1_client_is_chunked(gateway: Gateway) -> None:
    """A chat model is registered. The caller requests a streamed chat completion over HTTP/1.1 on a raw socket.
    The response head must carry `transfer-encoding: chunked` and no `content-length`, and the de-chunked body
    must end with `data: [DONE]` followed by a blank line."""
    with front_door(gateway) as door:
        body: Final = json.dumps(chat_body(door.chat_model(), stream=True)).encode()
        (response,) = exchange(gateway, json_post(gateway, CHAT, body, extra=(("Connection", "close"),)))
        assert response.status == 200, response.body
        assert response.header("transfer-encoding") == "chunked"
        assert response.header("content-length") is None
        assert response.body.endswith(b"data: [DONE]\n\n")


def test_a_stream_to_an_http_1_0_client_is_chunked_and_ends_with_the_connection(gateway: Gateway) -> None:
    """A chat model is registered. The caller requests a streamed chat completion with an `HTTP/1.0` request
    line. HTTP/1.0 has no chunked coding, so a strict server would send the SSE text unframed and end it by
    closing the connection.

    Pinned current behaviour: uvicorn's httptools protocol (`httptools_impl.py`) adds
    `transfer-encoding: chunked` to any response without a `content-length`, whatever the request's HTTP
    version, and closes the connection afterwards because HTTP/1.0 is not kept alive. The test pins that framing,
    no `content-length`, a de-chunked body that is the SSE text ending `data: [DONE]`, and the closed connection.
    The forwarder (hyper) frames by close for HTTP/1.0, so this changes once it is in front."""
    with front_door(gateway) as door, connection(gateway) as (client, stream):
        body: Final = json.dumps(chat_body(door.chat_model(), stream=True)).encode()
        client.sendall(json_post(gateway, CHAT, body, version="HTTP/1.0"))
        response: Final = read_response(stream)
        closed: Final = peer_closed(client, stream)
    assert response.status == 200, response.body
    assert (response.header("transfer-encoding"), response.header("content-length")) == ("chunked", None)
    assert response.body.startswith(b"data: ")
    assert response.body.endswith(b"data: [DONE]\n\n")
    assert closed


def test_a_request_with_150_headers_is_served(gateway: Gateway) -> None:
    """The caller sends a liveness GET carrying 150 small custom headers. The proxy must answer 200. uvicorn
    counts no headers and only caps the size of the whole head, whereas hyper refuses more than 100 by default,
    which is why the forwarder raises its limit (`MAX_HEADERS`, `forward.rs`)."""
    extra: Final = tuple((f"x-dec1-filler-{index}", "v") for index in range(150))
    (response,) = exchange(gateway, _get(gateway, LIVELINESS, *extra, ("Connection", "close")))
    assert (response.status, response.body) == (200, ALIVE)


@pytest.mark.parametrize(
    "target",
    (
        pytest.param("/health/liveliness?probe=1&note=a%20b%23c", id="query-with-escapes"),
        pytest.param("/health/%6Civeliness", id="percent-encoded-path-letter"),
        pytest.param("/health/liveliness?", id="empty-query"),
    ),
)
def test_request_targets_are_percent_decoded_for_routing(gateway: Gateway, target: str) -> None:
    """The caller requests the liveness probe with a query string containing `%20` and `%23`, with one path
    letter percent-encoded, and with a bare trailing `?`. Each must route to the probe and answer 200: the path
    is percent-decoded before matching and the query never affects routing. The forwarder passes the target
    through byte for byte, so the decoding stays in one place."""
    (response,) = exchange(gateway, _get(gateway, target, ("Connection", "close")))
    assert (response.status, response.body) == (200, ALIVE)


def test_an_absolute_form_request_target_is_routed_by_its_path(gateway: Gateway) -> None:
    """The caller sends `GET http://<host>/health/liveliness HTTP/1.1`, the absolute form an HTTP proxy client
    uses. The proxy must route on the path and answer 200 with the liveness body. UNVERIFIED: read from the
    forwarder's own test `absolute_form_targets_reach_python_as_origin_form`, not from uvicorn's parser."""
    target: Final = f"http://{host_header(gateway)}{LIVELINESS}"
    (response,) = exchange(gateway, _get(gateway, target, ("Connection", "close")))
    assert (response.status, response.body) == (200, ALIVE)


def test_hop_by_hop_request_headers_do_not_change_the_answer(gateway: Gateway) -> None:
    """The caller sends a liveness GET with `Connection: keep-alive, x-dec1-hop`, `Keep-Alive: timeout=5`,
    `TE: trailers` and the nominated `x-dec1-hop` header. These describe the client's own connection and are the
    headers the forwarder strips (`HOP_BY_HOP`, `rewrite.rs`). The answer must be the same 200 liveness body
    with or without them, and the connection must stay usable for a second request."""
    hop: Final = (
        ("Connection", "keep-alive, x-dec1-hop"),
        ("Keep-Alive", "timeout=5"),
        ("TE", "trailers"),
        ("x-dec1-hop", "1"),
    )
    with connection(gateway) as (client, stream):
        client.sendall(_get(gateway, LIVELINESS, *hop))
        first: Final = read_response(stream)
        client.sendall(_get(gateway, LIVELINESS, ("Connection", "close")))
        second: Final = read_response(stream)
    assert [(first.status, first.body), (second.status, second.body)] == [(200, ALIVE), (200, ALIVE)]


def test_a_client_cannot_strip_its_credentials_by_naming_them_in_connection(gateway: Gateway) -> None:
    """A chat model is registered. The caller sends a chat POST with `Connection: close, authorization`, which
    asks an intermediary to treat Authorization as hop-by-hop. The request must still be authenticated and served
    with 200. The forwarder refuses to drop credential, content and forwarding headers named in Connection
    (`must_forward`, `rewrite.rs`); without that rule this request would arrive at Python with no key."""
    with front_door(gateway) as door:
        body: Final = json.dumps(chat_body(door.chat_model())).encode()
        payload: Final = json_post(gateway, CHAT, body, extra=(("Connection", "close, authorization"),))
        (response,) = exchange(gateway, payload)
        assert response.status == 200, response.body
        assert len(door.calls()) == 1


def test_a_non_websocket_upgrade_request_is_served_as_a_plain_request(gateway: Gateway) -> None:
    """The caller sends a liveness GET with `Connection: Upgrade` and `Upgrade: h2c`. The proxy does not speak
    h2c, so it must ignore the offer and answer the request as plain HTTP/1.1 with 200 and the liveness body,
    not 101. UNVERIFIED: uvicorn logs "Unsupported upgrade request" and continues; the forwarder forwards
    non-WebSocket upgrades as ordinary requests."""
    payload: Final = _get(gateway, LIVELINESS, ("Connection", "Upgrade"), ("Upgrade", "h2c"))
    with connection(gateway) as (client, stream):
        client.sendall(payload)
        response: Final = read_response(stream)
    assert (response.status, response.body) == (200, ALIVE)
