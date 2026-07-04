"""Spend-tracking suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. SpendClient exposes the shared Gateway
(GatewayProvider), so the `resources` fixture cleans up keys and customers this
suite creates.
"""

import pytest

from spend_e2e_client import SpendClient, build_client


@pytest.fixture(scope="session")
def client() -> SpendClient:
    return build_client()
