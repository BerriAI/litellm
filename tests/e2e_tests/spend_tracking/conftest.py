"""Spend-tracking suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e_tests/conftest.py. SpendE2EClient satisfies the
lifecycle.ResourceClient protocol, so the shared `resources` fixture cleans up
keys and customers this suite creates.
"""

import pytest

from spend_e2e_client import SpendE2EClient, build_client


@pytest.fixture(scope="session")
def client() -> SpendE2EClient:
    return build_client()
