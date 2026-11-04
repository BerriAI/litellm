from __future__ import annotations

import datetime
import socket
import ssl
import time
from collections.abc import Callable
from contextlib import ExitStack
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from tests.integration._support.wire import Reply, Request, wire_server

PAUSE_SECONDS: Final = 6
HANDLER_READ_SECONDS: Final = 0.3
FIRST: Final = b'{"call": "first"}'
LATE: Final = b'{"call": "late"}'
UNREAD: Final = b"x" * (32 * 1024 * 1024)


def _self_signed_server_context(directory: Path) -> ssl.SSLContext:
    key: Final = ec.generate_private_key(ec.SECP256R1())
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "wire peer")])
    now: Final = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=36500))
        .sign(key, hashes.SHA256())
    )
    (directory / "peer.pem").write_bytes(
        certificate.public_bytes(serialization.Encoding.PEM)
        + key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(directory / "peer.pem")
    return context


def _client(url: str) -> HTTPConnection:
    address: Final = urlsplit(url)
    if address.scheme == "http":
        return HTTPConnection(address.hostname or "", address.port)
    trusting: Final = ssl.create_default_context()
    trusting.check_hostname = False
    trusting.verify_mode = ssl.CERT_NONE
    return HTTPSConnection(address.hostname or "", address.port, context=trusting)


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


@pytest.mark.timeout(60)
def test_a_reply_its_client_stopped_reading_fails_teardown_instead_of_hanging() -> None:
    peer: Final = ExitStack()
    wire: Final = peer.enter_context(wire_server(lambda _: Reply(body=UNREAD), keep_alive=True, write_grace=0.5))
    address: Final = urlsplit(wire.url)
    with socket.create_connection((address.hostname or "", address.port or 0)) as client:
        client.sendall(b"GET /large HTTP/1.1\r\nhost: peer\r\n\r\n")
        while wire.received.empty():
            time.sleep(0.01)
        with pytest.raises(AssertionError, match="still being written"):
            peer.close()


def test_a_complete_request_sent_as_teardown_starts_is_still_answered() -> None:
    with wire_server(_echo, keep_alive=True) as wire:
        connection: Final = _client(wire.url)
        connection.request("POST", "/first", FIRST)
        first: Final = connection.getresponse().read()
        connection.request("POST", "/complete", LATE)
    late: Final = connection.getresponse()
    assert (first, late.status, late.read(), tuple(request.target for request in wire.drain())) == (
        FIRST,
        200,
        LATE,
        ("/first", "/complete"),
    )
    connection.close()


@pytest.mark.parametrize("secure", (False, True), ids=("plain", "tls"))
def test_a_reply_still_being_written_at_teardown_reaches_its_client_whole(secure: bool, tmp_path: Path) -> None:
    slow: Final = Reply(chunks=(b"first", b"second"), pause_between_chunks=HANDLER_READ_SECONDS)
    tls: Final = _self_signed_server_context(tmp_path) if secure else None
    with wire_server(lambda _: slow, tls=tls, keep_alive=True) as wire:
        connection: Final = _client(wire.url)
        connection.request("GET", "/slow")
        response: Final = connection.getresponse()
        head: Final = response.read(len(b"first"))
    assert head + response.read() == b"firstsecond"
    connection.close()
