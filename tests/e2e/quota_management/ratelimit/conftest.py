"""Quota-management suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. QuotaClient holds the shared ProxyClient,
so the `resources` fixture cleans up keys through it.
"""

import pytest

from quota_client import QuotaClient, build_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> QuotaClient:
    return build_client(proxy)
