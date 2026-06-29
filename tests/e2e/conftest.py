"""Shared fixtures for all live e2e suites under tests/e2e/.

Design rule: skip on environment, fail on behavior. Live tests (marked `e2e`)
skip when no proxy answers; once a request reaches the proxy, behavior is
asserted. Pure unit coverage of the harness itself carries no `e2e` marker and
runs regardless of whether a proxy is up.

Lifecycle: the `resources` fixture maps the init -> run -> teardown contract
(lifecycle.E2ECase) onto pytest - setup is init(), the test body is run(), and
teardown deletes every resource the test created on the long-lived proxy.

Each suite provides its own `client` fixture (a lifecycle.ResourceClient); these
shared fixtures build on it.
"""

import functools
from typing import Iterator

import pytest
import requests

from e2e_config import CONTROL_PLANE_BASE_URL, PROXY_BASE_URL
from lifecycle import GatewayProvider, ResourceManager


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: live test that requires a running proxy and real provider keys",
    )


def _liveness_reason(label: str, base_url: str) -> str | None:
    """None if `base_url` answers its liveness probe, else a skip reason."""
    try:
        resp = requests.get(f"{base_url}/health/liveliness", timeout=5)
    except requests.RequestException as exc:
        return f"No live {label} at {base_url}: {exc}"
    if resp.status_code >= 500:
        return f"{label} at {base_url} returned {resp.status_code}"
    return None


@functools.lru_cache(maxsize=1)
def _proxy_skip_reason() -> str | None:
    """Probe the proxy once per session. None if it answers, else a skip reason. In
    a split deployment the management/admin control plane is a separate service, so
    require it too (when it differs) - else its tests would fail rather than skip."""
    reason = _liveness_reason("proxy", PROXY_BASE_URL)
    if reason is not None:
        return reason
    if CONTROL_PLANE_BASE_URL != PROXY_BASE_URL:
        return _liveness_reason("control plane", CONTROL_PLANE_BASE_URL)
    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip `e2e`-marked tests unless a proxy answers its liveness probe. Unmarked
    tests (unit coverage of the harness) don't touch the proxy, so they run even
    when none is up."""
    if item.get_closest_marker("e2e") is None:
        return
    reason = _proxy_skip_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture
def resources(client: GatewayProvider) -> Iterator[ResourceManager]:
    """init -> run -> teardown: create a manager, run the test, release resources.
    Cleanup goes through the shared Gateway, whatever the suite's client adds."""
    manager = ResourceManager(client=client.gateway)
    manager.init()
    yield manager
    manager.teardown()


@pytest.fixture
def scoped_key(resources: ResourceManager) -> str:
    """A fresh all-models key per test, auto-deleted by the resources teardown."""
    return resources.key()
