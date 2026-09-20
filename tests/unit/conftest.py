from collections.abc import Iterator
from typing import Final

import pytest
from pytest_socket import enable_socket, socket_allow_hosts

LOOPBACK_HOSTS: Final = ["127.0.0.1", "::1"]


def _allow_loopback_only() -> None:
    socket_allow_hosts(LOOPBACK_HOSTS, allow_unix_socket=True)


@pytest.fixture(autouse=True, scope="session")
def block_external_sockets() -> Iterator[None]:
    _allow_loopback_only()
    yield
    enable_socket()


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup() -> None:
    _allow_loopback_only()
