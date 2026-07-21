"""Management suite's `client` fixture.

Lifecycle/liveness gate/marker live in the parent conftest. ManagementClient
holds the shared ProxyClient so `resources` / `scoped_key` clean up keys, teams,
users, and orgs this suite creates.
"""

import pytest

from management_client import ManagementClient, build_client
from proxy_client import ProxyClient
from sso_management_client import SSOManagementClient, build_sso_client


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. mgmt.key.generate.persists",
    )


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ManagementClient:
    return build_client(proxy)


@pytest.fixture(scope="session")
def sso_client(proxy: ProxyClient) -> SSOManagementClient:
    return build_sso_client(proxy)
