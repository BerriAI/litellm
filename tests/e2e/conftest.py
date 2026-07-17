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
from collections.abc import Iterator
from pathlib import Path

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


def _package_from_nodeid(nodeid: str) -> str:
    """Top-level suite package under tests/e2e/, or 'root' for top-level files.

    Pytest nodeids are relative to the invocation cwd. Repo-root runs look like
    `tests/e2e/logging/...`; suite-cwd runs look like `logging/...`. Strip the
    `tests/e2e` prefix so package is the suite dir either way.
    """
    path_part = nodeid.split("::", 1)[0].replace("\\", "/")
    raw = tuple(p for p in path_part.split("/") if p and p != ".")
    parts = raw[2:] if len(raw) >= 3 and raw[0] == "tests" and raw[1] == "e2e" else raw
    if len(parts) <= 1:
        return "root"
    return parts[0]


def _covers_from_item(item: pytest.Item) -> tuple[str, ...]:
    """Read @pytest.mark.covers cell ids off a pytest Item, order-preserving."""
    marker_args: tuple[tuple[object, ...], ...] = tuple(marker.args for marker in item.iter_markers(name="covers"))
    return tuple(dict.fromkeys(arg for args in marker_args for arg in args if isinstance(arg, str) and arg))


def _result_properties(item: pytest.Item) -> tuple[tuple[str, str], ...]:
    """The only custom signal a standard reporter cannot derive on its own: the
    normalized suite package and the coverage-registry cell ids this test covers."""
    return (
        ("package", _package_from_nodeid(item.nodeid)),
        ("covers", ",".join(_covers_from_item(item))),
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach package + covers to every item's user_properties so a standard
    reporter (`--junitxml`) captures them per test, on every outcome including
    skips and setup errors. Downstream (Loki/Grafana) reads outcome and duration
    from the standard artifact and these two properties for package rollups and
    coverage drill-down; no bespoke log line is emitted."""
    for item in items:
        item.user_properties.extend(_result_properties(item))


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
