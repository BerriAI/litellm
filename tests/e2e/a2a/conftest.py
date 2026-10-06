"""A2A suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. A2AClient holds the shared ProxyClient,
so the `resources` fixture cleans up keys this suite creates; agents are torn down
via `resources.defer(...)` in each test.
"""

import pytest

from a2a_client import A2AClient, build_a2a_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> A2AClient:
    return build_a2a_client(proxy)
