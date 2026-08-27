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
from collections.abc import Generator, Iterator
from datetime import datetime, timezone

import pytest
import requests

from e2e_config import CONTROL_PLANE_BASE_URL, FIXTURE_DIR, FIXTURE_MODE_RAW, PROXY_BASE_URL
from e2e_db import RESET_OPT_IN_ENV, reset_spend_logs, run_spend_log_cleanup
from fixture_mode import fixture_mode_collection_error, fixture_report_lines
from provider_edge import replay_leftover_error
from junit_properties import attach_result_properties
from lifecycle import ProxyClientProvider, ResourceManager
from proxy_client import ProxyClient, build_proxy_client


_E2E_TEST_RAN = pytest.StashKey[bool]()
_CALL_PASSED = pytest.StashKey[bool]()


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
        "replayable: edge-wired test whose provider traffic replays from a fixture bundle, so it makes "
        "zero provider calls in replay mode; the record/replay CI lane selects it with -m replayable",
    )
    config.addinivalue_line(
        "markers",
        "load: heavy throughput/load test; collected last so it never perturbs latency-sensitive suites",
    )
    config.addinivalue_line(
        "markers",
        "weekly: real-provider anomaly load test that spends real money; deselected unless E2E_WEEKLY_ANOMALY is set",
    )
    config.addinivalue_line(
        "markers",
        "managed_files: needs a proxy running with require_managed_files enabled; deselected unless E2E_MANAGED_FILES_STACK is set",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Abort before collection when E2E_FIXTURE_MODE can never work: an unknown
    mode value, or replay against a missing, unreadable, or stale bundle (the
    stale message names the bundle's age). Live and record modes pass through."""
    reason = fixture_mode_collection_error(
        FIXTURE_MODE_RAW, FIXTURE_DIR, now=datetime.now(timezone.utc)
    )
    if reason is not None:
        raise pytest.UsageError(reason)


def pytest_report_header(config: pytest.Config) -> list[str]:
    return fixture_report_lines(FIXTURE_MODE_RAW, FIXTURE_DIR, now=datetime.now(timezone.utc))


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
    run even when none is up. Never skip for a missing proxy. Replay mode needs
    the proxy too: only provider-bound traffic replays from the bundle."""
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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Stash the call-phase outcome so teardown can tell a passed test from a
    failed one without re-deriving it."""
    report = yield
    if report.when == "call":
        item.stash[_CALL_PASSED] = report.passed
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, None, None]:
    """In replay mode a passing test must consume its whole recording: leftover
    interactions mean the test now makes fewer calls than it did at record time,
    so the replay proved less than the bundle claims. The check runs after the
    yield so fixture finalizers replay their recorded calls first. Failed tests
    are left alone - their own failure already explains any unconsumed tail."""
    result = yield
    if not item.stash.get(_CALL_PASSED, False):
        return result
    reason = replay_leftover_error(
        mode_raw=FIXTURE_MODE_RAW, bundle_dir=FIXTURE_DIR, test_key=item.nodeid
    )
    if reason is not None:
        pytest.fail(reason)
    return result


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
