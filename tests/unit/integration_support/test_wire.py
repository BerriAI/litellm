from __future__ import annotations

import time
from collections.abc import Callable
from http.client import HTTPConnection, HTTPResponse
from typing import Final
from urllib.parse import urlsplit

import pytest

from tests.integration._support.wire import Reply, Request, wire_server

PAUSE_SECONDS: Final = 6
HANDLER_READ_SECONDS: Final = 0.3
FIRST: Final = b'{"call": "first"}'
LATE: Final = b'{"call": "late"}'


def _echo(request: Request) -> Reply:
    return Reply(body=request.body)


def _pause_then_send(connection: HTTPConnection, body: bytes) -> HTTPResponse:
    time.sleep(PAUSE_SECONDS)
    connection.request("POST", "/late", body)
    return connection.getresponse()


def _send_headers_then_pause_before_the_body(connection: HTTPConnection, body: bytes) -> HTTPResponse:
    connection.putrequest("POST", "/late")
    connection.putheader("content-length", str(len(body)))
    connection.endheaders()
    time.sleep(PAUSE_SECONDS)
    connection.send(body)
    return connection.getresponse()


@pytest.mark.parametrize(
    "send_late",
    (_pause_then_send, _send_headers_then_pause_before_the_body),
    ids=("pause-before-the-request", "pause-before-the-body"),
)
def test_a_kept_alive_connection_answers_a_request_whose_bytes_arrive_after_a_long_pause(
    send_late: Callable[[HTTPConnection, bytes], HTTPResponse],
) -> None:
    with wire_server(_echo, keep_alive=True) as wire:
        address: Final = urlsplit(wire.url)
        connection: Final = HTTPConnection(address.hostname or "", address.port)
        connection.request("POST", "/first", FIRST)
        first: Final = connection.getresponse().read()
        late: Final = send_late(connection, LATE)
        assert (first, late.status, late.read()) == (FIRST, 200, LATE)
        assert tuple(request.body for request in wire.drain()) == (FIRST, LATE)
        assert wire.connections() == 1
    assert connection.sock.recv(1) == b""
    connection.close()


def test_a_request_cut_off_by_teardown_never_reaches_respond() -> None:
    with wire_server(_echo, keep_alive=True) as wire:
        address: Final = urlsplit(wire.url)
        connection: Final = HTTPConnection(address.hostname or "", address.port)
        connection.request("POST", "/first", FIRST)
        first: Final = connection.getresponse().read()
        connection.sock.sendall(b"POST /cut HTTP/1.1\r\ncontent-type: application/json\r\ncontent-len")
        time.sleep(HANDLER_READ_SECONDS)
    assert (first, tuple(request.target for request in wire.drain())) == (FIRST, ("/first",))
    connection.close()
