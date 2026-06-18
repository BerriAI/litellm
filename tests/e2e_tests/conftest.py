"""Shared fixtures for all live e2e suites under tests/e2e_tests/.

Design rule: skip on environment, fail on behavior. If the proxy is unreachable
the whole session skips; once a request reaches the proxy, behavior is asserted.

Lifecycle: the `resources` fixture maps the init -> run -> teardown contract
(lifecycle.E2ECase) onto pytest - setup is init(), the test body is run(), and
teardown deletes every resource the test created on the long-lived proxy.

Each suite provides its own `client` fixture (a lifecycle.ResourceClient); these
shared fixtures build on it.
"""

from typing import Iterator

import pytest
import requests

from e2e_config import PROXY_BASE_URL
from lifecycle import ResourceClient, ResourceManager


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: live test that requires a running proxy and real provider keys",
    )


@pytest.fixture(scope="session", autouse=True)
def _require_live_proxy() -> None:
    """Skip the entire session unless a proxy answers its liveness probe."""
    try:
        resp = requests.get(f"{PROXY_BASE_URL}/health/liveliness", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"No live proxy at {PROXY_BASE_URL}: {exc}")
        return
    if resp.status_code >= 500:
        pytest.skip(f"Proxy at {PROXY_BASE_URL} returned {resp.status_code}")


@pytest.fixture
def resources(client: ResourceClient) -> Iterator[ResourceManager]:
    """init -> run -> teardown: create a manager, run the test, release resources."""
    manager = ResourceManager(client=client)
    manager.init()
    yield manager
    manager.teardown()


@pytest.fixture
def scoped_key(resources: ResourceManager) -> str:
    """A fresh all-models key per test, auto-deleted by the resources teardown."""
    return resources.key()
