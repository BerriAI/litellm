"""Shared fixtures for all live e2e suites under tests/e2e/.

Design rule: hard failures only. Live tests (marked `e2e`) fail when no proxy
answers or when credentials/env are missing; they never skip. Pure unit coverage
of the harness itself carries no `e2e` marker and runs regardless of whether a
proxy is up.

Lifecycle: the `resources` fixture maps the init -> run -> teardown contract
(lifecycle.E2ECase) onto pytest - setup is init(), the test body is run(), and
teardown deletes every resource the test created on the long-lived proxy.

Each suite provides its own `client` fixture (a lifecycle.ResourceClient); these
shared fixtures build on it.
"""

import functools
import sys
from pathlib import Path
from typing import Iterator

import pytest
import requests

from e2e_config import CONTROL_PLANE_BASE_URL, PROXY_BASE_URL
from lifecycle import GatewayProvider, ResourceManager


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


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]):
    """Emit one structured E2E_RESULT line per finished test for Loki/Grafana.

    Status-history panels should aggregate by package (and optional covers), not
    scrape pytest progress basenames. See e2e_result_reporter.py.
    """
    outcome = yield
    report = outcome.get_result()
    from e2e_result_reporter import covers_from_item, format_e2e_result_line, result_from_pytest

    result = result_from_pytest(
        nodeid=report.nodeid,
        when=report.when,
        failed=report.failed,
        skipped=report.skipped,
        passed=report.passed,
        duration_seconds=float(report.duration),
        covers=covers_from_item(item),
    )
    if result is None:
        return
    print(format_e2e_result_line(result), flush=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Once the whole e2e session is done (all suites), truncate the spend logs so
    the DB doesn't accumulate test rows. Sessions where no e2e test body ran leave
    the DB alone so a `DATABASE_URL` pointing at a shared instance is never wiped
    without an e2e run. Best-effort: a cleanup failure (no DB reachable) must not
    fail the run. The spend_tracking dir goes on sys.path only for this import and
    is removed after, so a broader `pytest tests/` run is not left with a mutated
    path."""
    if not session.stash.get(_E2E_TEST_RAN, False):
        return
    spend_dir = str(Path(__file__).parent / "quota_management" / "spend_tracking")
    sys.path.insert(0, spend_dir)
    try:
        from spend_e2e_client import reset_spend_logs  # pyright: ignore

        reset_spend_logs()
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        print(f"spend-log cleanup best-effort failed: {exc}")
    finally:
        if spend_dir in sys.path:
            sys.path.remove(spend_dir)

    try:
        from bob_the_builder import remediate

        remediate(session)
    except Exception as exc:  # noqa: BLE001 - remediation is best-effort
        print(f"devin remediation best-effort failed: {exc}")


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
