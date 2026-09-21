import socket

import pytest
from pytest_socket import SocketConnectBlockedError


def test_external_connect_is_refused_before_a_packet_leaves() -> None:
    with pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("192.0.2.1", 9), timeout=1)


def test_loopback_connect_is_allowed() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.create_connection(server.getsockname(), timeout=1) as client:
            assert client.getpeername() == server.getsockname()
