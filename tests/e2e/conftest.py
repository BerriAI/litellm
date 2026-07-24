"""Shared fixtures for all live e2e suites under tests/e2e/.

Design rule: hard failures only. Live tests (marked `e2e`) fail when no proxy
answers or when credentials/env are missing; they never skip. Pure unit coverage
of the harness itself carries no `e2e` marker and runs regardless of whether a
proxy is up.

Lifecycle: the `resources` fixture hands each test a lifecycle.ResourceManager -
the test registers a cleanup for every resource it creates, and the fixture's
teardown deletes them all on the long-lived proxy, even when the test fails.

Each suite provides its own `client` fixture (a lifecycle.ResourceClient); these
shared fixtures build on it.
"""

import functools
import os
from collections.abc import Iterator

import pytest
import requests

from e2e_config import CONTROL_PLANE_BASE_URL, PROXY_BASE_URL
from e2e_db import RESET_OPT_IN_ENV, reset_spend_logs, run_spend_log_cleanup
from junit_properties import attach_result_properties
from lifecycle import ProxyClientProvider, ResourceManager
from proxy_client import ProxyClient, build_proxy_client


_E2E_TEST_RAN = pytest.StashKey[bool]()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: live test that requires a running proxy and real provider keys",
    )
    config.addinivalue_line(
        "markers",
        "covers(cell_id, *, exercised_on=()): coverage-registry cell(s) this test covers",
    )
    config.addinivalue_line(
        "markers",
        "load: heavy throughput/load test; collected last so it never perturbs latency-sensitive suites",
    )
    config.addinivalue_line(
        "markers",
        "weekly: real-provider anomaly load test that spends real money; deselected unless E2E_WEEKLY_ANOMALY is set",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach the two custom signals (suite package and covered cell ids) to every
    test's user_properties so the standard JUnit report (`--junitxml`) records them
    as `<property>` entries, on every outcome including skips and setup errors.
    Downstream (Loki/Grafana) reads outcome and duration from the standard report
    and these properties for package rollups and coverage drill-down. See
    junit_properties.py.

    Also sort `load`-marked items last so a whole-tree run drives heavy throughput
    traffic only after the latency-sensitive suites have finished."""
    for item in items:
        attach_result_properties(item)
    items.sort(key=lambda item: item.get_closest_marker("load") is not None)


def _liveness_reason(label: str, base_url: str) -> str | None:
    """None if `base_url` answers its liveness probe, else a failure reason."""
    try:
        resp = requests.get(f"{base_url}/health/liveliness", timeout=5)
    except requests.RequestException as exc:
        return f"No live {label} at {base_url}: {exc}"
    if resp.status_code >= 500:
        return f"{label} at {base_url} returned {resp.status_code}"
    return None


@functools.lru_cache(maxsize=1)
def _proxy_fail_reason() -> str | None:
    """Probe the proxy once per session. None if it answers, else a failure reason.
    In a split deployment the management/admin control plane is a separate service,
    so require it too when it differs."""
    reason = _liveness_reason("proxy", PROXY_BASE_URL)
    if reason is not None:
        return reason
    if CONTROL_PLANE_BASE_URL != PROXY_BASE_URL:
        return _liveness_reason("control plane", CONTROL_PLANE_BASE_URL)
    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Hard-fail `e2e`-marked tests unless a proxy answers its liveness probe.
    Unmarked tests (unit coverage of the harness) don't touch the proxy, so they
    run even when none is up. Never skip for a missing proxy."""
    if item.get_closest_marker("e2e") is None:
        return
    reason = _proxy_fail_reason()
    if reason is not None:
        pytest.fail(reason)


def pytest_runtest_call(item: pytest.Item) -> None:
    """Mark that an e2e test body actually ran (setup passed). Sessions that fail
    setup never reach this hook, so the session-finish cleanup can use it as a
    guard before truncating the spend-log DB. Tests under `tests/e2e/` without the
    `e2e` marker (pure unit coverage for the harness itself) never hit the proxy,
    so they must not arm the destructive DB truncate."""
    if item.get_closest_marker("e2e") is None:
        return
    item.session.stash[_E2E_TEST_RAN] = True


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Once the whole e2e session is done (all suites), optionally truncate the
    spend logs so the DB doesn't accumulate test rows. The truncate is destructive
    and irreversible, so it runs only when the operator explicitly opts in
    (`E2E_RESET_SPEND_LOGS=1`) and an e2e test body actually ran; otherwise a
    `DATABASE_URL` pointing at a shared or staging instance is left untouched.
    Best-effort: a cleanup failure (no DB reachable) must not fail the run."""
    run_spend_log_cleanup(
        opt_in=os.environ.get(RESET_OPT_IN_ENV),
        e2e_test_ran=session.stash.get(_E2E_TEST_RAN, False),
        truncate=reset_spend_logs,
    )


@pytest.fixture(scope="session")
def proxy() -> ProxyClient:
    """The shared ProxyClient every suite's client is built from. Suite `client`
    fixtures depend on this and inject it, so the proxy wiring lives in one place."""
    return build_proxy_client()


@pytest.fixture
def resources(client: ProxyClientProvider) -> Iterator[ResourceManager]:
    """init -> run -> teardown: create a manager, run the test, release resources.
    Cleanup goes through the shared ProxyClient, whatever the suite's client adds."""
    manager = ResourceManager(client=client.proxy)
    manager.init()
    yield manager
    manager.teardown()


@pytest.fixture
def scoped_key(resources: ResourceManager) -> str:
    """A fresh all-models key per test, auto-deleted by the resources teardown."""
    return resources.key()
