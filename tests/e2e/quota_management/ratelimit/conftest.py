"""Quota-management suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. QuotaClient holds the shared Gateway,
so the `resources` fixture cleans up keys through it.
"""

import pytest

from quota_client import QuotaClient, build_client


@pytest.fixture(scope="session")
def client() -> QuotaClient:
    return build_client()
