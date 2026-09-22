import socket

import pytest
from pytest_socket import SocketConnectBlockedError


def test_external_connect_is_refused_before_a_packet_leaves() -> None:
    with pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("192.0.2.1", 9), timeout=1)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_connect_is_allowed(host: str) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.socket() as client:
            client.settimeout(1)
            client.connect((host, server.getsockname()[1]))
            assert client.getpeername() == server.getsockname()
