"""Management suite's `client` fixture.

Lifecycle/liveness gate/marker live in the parent conftest. ManagementClient
holds the shared ProxyClient so `resources` / `scoped_key` clean up keys, teams,
users, and orgs this suite creates.
"""

from collections.abc import Generator
from typing import Final

import pytest
from e2e_http import without_retries
from idp import Keycloak
from lifecycle import ResourceManager
from management.jwt_actors import ActorFactory
from management_client import ManagementClient, build_client
from proxy_client import ProxyClient


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. mgmt.key.generate.persists",
    )


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ManagementClient:
    return build_client(proxy)


@pytest.fixture
def actor_factory(proxy: ProxyClient, idp: Keycloak) -> Generator[ActorFactory]:
    bootstrap: Final = build_client(proxy)
    resources: Final = ResourceManager(client=proxy, strict_cleanup=True)
    with without_retries():
        try:
            yield ActorFactory(bootstrap=bootstrap, idp=idp, resources=resources)
        finally:
            resources.teardown()
