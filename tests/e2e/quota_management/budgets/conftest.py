"""Budgets suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. BudgetClient holds the shared ProxyClient,
so the `resources` fixture cleans up keys through it; tests register entity deletes
via `resources.defer(...)`.
"""

import pytest

from budget_client import BudgetClient, build_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> BudgetClient:
    return build_client(proxy)
