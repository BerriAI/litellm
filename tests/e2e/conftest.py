"""Shared fixtures for all live e2e suites under tests/e2e/.

Design rule: skip on environment, fail on behavior. If the proxy is unreachable
the whole session skips; once a request reaches the proxy, behavior is asserted.

Lifecycle: the `resources` fixture maps the init -> run -> teardown contract
(lifecycle.E2ECase) onto pytest - setup is init(), the test body is run(), and
teardown deletes every resource the test created on the long-lived proxy.

Each suite provides its own `client` fixture (a lifecycle.ResourceClient); these
shared fixtures build on it.
"""

import sys
from pathlib import Path
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


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Once the whole e2e session is done (all suites), truncate the spend logs so
    the DB doesn't accumulate test rows. Best-effort: a cleanup failure (no DB
    reachable) must not fail the run."""
    sys.path.insert(0, str(Path(__file__).parent / "spend_tracking"))
    try:
        from spend_e2e_client import reset_spend_logs  # pyright: ignore

        reset_spend_logs()
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        print(f"spend-log cleanup skipped: {exc}")


def _probe_live(label: str, base_url: str) -> None:
    """Skip the whole session (environment, not behavior) if `base_url` doesn't
    answer its liveness probe."""
    try:
        resp = requests.get(f"{base_url}/health/liveliness", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"No live {label} at {base_url}: {exc}")
        return
    if resp.status_code >= 500:
        pytest.skip(f"{label} at {base_url} returned {resp.status_code}")


@pytest.fixture(scope="session", autouse=True)
def _require_live_proxy() -> None:
    """Skip the entire session unless the proxy answers its liveness probe. In a
    split deployment the management/admin control plane is a separate service, so
    require it too (when it differs) — else its tests would fail rather than skip."""
    _probe_live("proxy", PROXY_BASE_URL)
    if CONTROL_PLANE_BASE_URL != PROXY_BASE_URL:
        _probe_live("control plane", CONTROL_PLANE_BASE_URL)


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
