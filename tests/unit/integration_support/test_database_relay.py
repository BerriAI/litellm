from __future__ import annotations

import socket
import socketserver
import threading
from collections.abc import Iterator
from typing import Final
from urllib.parse import urlsplit

import pytest

from tests.integration._support.database_relay import dropped_connection_relay

TRIGGER: Final = b'SELECT "startTime" FROM "LiteLLM_SpendLogs"'
SPLIT_AT: Final = len(TRIGGER) // 2


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while chunk := self.request.recv(65536):
            self.request.sendall(chunk)


@pytest.fixture
def echo_upstream_url() -> Iterator[str]:
    server: Final = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"postgresql://relay:relay@127.0.0.1:{server.server_address[1]}/relay"
    finally:
        server.shutdown()
        server.server_close()


def test_dropped_connection_relay_trips_on_a_trigger_split_across_two_reads(echo_upstream_url: str) -> None:
    with dropped_connection_relay(echo_upstream_url, TRIGGER) as (relay, relayed_url):
        relay.arm()
        port: Final = urlsplit(relayed_url).port
        assert port is not None
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client, client.makefile("rb") as echoed:
            client.sendall(TRIGGER[:SPLIT_AT])
            assert echoed.read(SPLIT_AT) == TRIGGER[:SPLIT_AT]
            client.sendall(TRIGGER[SPLIT_AT:])
            assert relay.dropped.wait(5), "the relay never saw the trigger that arrived in two reads"
            assert echoed.read() == b""
