"""`secret_manager` suite fixtures.

Lifecycle (resources/scoped_key), proxy liveness gate, and the e2e/covers
markers all live in the parent tests/e2e/conftest.py. The suite has no routes of
its own, so its `client` is the shared ProxyClient; `vault` is the manager the
proxy under test is configured against.
"""

from __future__ import annotations

import pytest

from proxy_client import ProxyClient
from vault_client import Vault, vault_from_env


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ProxyClient:
    return proxy


@pytest.fixture(scope="session")
def vault() -> Vault:
    return vault_from_env()
