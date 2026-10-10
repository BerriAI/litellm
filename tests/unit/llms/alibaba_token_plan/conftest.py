import socket
from ipaddress import ip_address

import pytest


@pytest.fixture(autouse=True)
def resolve_mock_media_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve(
        host: str, port: int | None, *, proto: int = 0
    ) -> tuple[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]], ...]:
        if host == "localhost":
            return ((socket.AF_INET, socket.SOCK_STREAM, proto, "", ("127.0.0.1", port or 0)),)
        try:
            ip_address(host)
        except ValueError:
            assert host.endswith(".example"), f"Unexpected DNS lookup in unit test: {host}"
            return ((socket.AF_INET, socket.SOCK_STREAM, proto, "", ("93.184.216.34", port or 0)),)
        return ((socket.AF_INET, socket.SOCK_STREAM, proto, "", (host, port or 0)),)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
