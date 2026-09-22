"""`client_apps` suite's `client` fixture.

Lifecycle (resources/scoped_key), proxy liveness gate, and the e2e/covers
markers all live in the parent tests/e2e/conftest.py. ClientAppsClient holds the
shared ProxyClient so the keys these tests mint tear down through it.
"""

from __future__ import annotations

import pytest

from client_apps_client import ClientAppsClient
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ClientAppsClient:
    return ClientAppsClient(proxy=proxy)
