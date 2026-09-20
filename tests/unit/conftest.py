from collections.abc import Iterator

import pytest
from pytest_socket import enable_socket, socket_allow_hosts


@pytest.fixture(autouse=True, scope="session")
def block_external_sockets() -> Iterator[None]:
    socket_allow_hosts(["127.0.0.1", "::1"], allow_unix_socket=True)
    yield
    enable_socket()


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup() -> None:
    socket_allow_hosts(["127.0.0.1", "::1"], allow_unix_socket=True)
