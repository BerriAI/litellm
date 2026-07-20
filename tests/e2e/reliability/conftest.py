"""Reliability suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. ReliabilityClient holds the shared
ProxyClient, so the `resources` fixture cleans up the mock deployments and keys
this suite creates. No model is force-registered here: each test creates its own
mock deployments under unique names and tears them down.
"""

from __future__ import annotations

import pytest

from proxy_client import ProxyClient
from reliability_client import ReliabilityClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ReliabilityClient:
    return ReliabilityClient(proxy=proxy)
