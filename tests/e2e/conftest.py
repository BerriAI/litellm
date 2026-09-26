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
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import requests
from e2e_config import (
    CLI_DETERMINISM_OPT_IN_ENV,
    CONTROL_PLANE_BASE_URL,
    FIXTURE_DIR,
    FIXTURE_MODE_RAW,
    MANAGED_FILES_OPT_IN_ENV,
    MCP_OAUTH_LIVE_OPT_IN_ENV,
    OTEL_TLS_OPT_IN_ENV,
    OTEL_V2_OPT_IN_ENV,
    OWNED_GATEWAY_OPT_IN_ENV,
    PROMPT_CACHING_OPT_IN_ENV,
    PROVIDER_EDGE_HOST_OPT_IN_ENV,
    PROXY_BASE_URL,
    REDIS_CHAOS_OPT_IN_ENV,
    SECRET_MANAGER_OPT_IN_ENV,
    WEEKLY_ANOMALY_OPT_IN_ENV,
    unique_marker,
)
from e2e_db import RESET_OPT_IN_ENV, reset_spend_logs, run_spend_log_cleanup
from e2e_http import unwrap
from fixture_mode import fixture_mode_collection_error, fixture_report_lines
from fixture_mode import pytest_fixture_setup as pytest_fixture_setup
from idp import Identity, Keycloak, keycloak_from_env
from junit_properties import attach_result_properties
from lifecycle import ProxyClientProvider, ResourceManager
from memory_readings import RssCapture, read_rss_everywhere
from models import TeamNewBody, UserNewBody, UserNewResponse
from provider_cache_routing import LIVE_PROVIDER_REQUIRED
from provider_edge import replay_leftover_error
from proxy_client import ProxyClient, build_proxy_client
from stack_lock import stack_lock

_E2E_TEST_RAN = pytest.StashKey[bool]()
_CALL_PASSED = pytest.StashKey[bool]()
_IDLE_RSS = pytest.StashKey[RssCapture]()

IDLE_RSS_READ_TIMEOUT_SECONDS: Final = 10.0

OPT_IN_MARKERS: Final = MappingProxyType(
    {
        "weekly": WEEKLY_ANOMALY_OPT_IN_ENV,
        "managed_files": MANAGED_FILES_OPT_IN_ENV,
        "prompt_caching_stack": PROMPT_CACHING_OPT_IN_ENV,
        "redis_chaos": REDIS_CHAOS_OPT_IN_ENV,
        "cli_determinism": CLI_DETERMINISM_OPT_IN_ENV,
        "mcp_oauth_live": MCP_OAUTH_LIVE_OPT_IN_ENV,
        "provider_edge_host": PROVIDER_EDGE_HOST_OPT_IN_ENV,
        "owned_gateway": OWNED_GATEWAY_OPT_IN_ENV,
        "otel_v2": OTEL_V2_OPT_IN_ENV,
        "otel_tls": OTEL_TLS_OPT_IN_ENV,
        "secret_manager": SECRET_MANAGER_OPT_IN_ENV,
    }
)


@pytest.fixture(scope="session")
def idp() -> Keycloak:
    return keycloak_from_env()


@pytest.fixture
def jwt_identity(idp: Keycloak, resources: ResourceManager, proxy: ProxyClient) -> Identity:
    marker: Final = unique_marker()
    identity: Final = idp.provision(marker=marker, group=f"e2e-jwt-team-{marker}", defer=resources.defer)
    resources.defer(lambda: proxy.delete_user(identity.user_id))
    # Seed the canonical user before any JWT call populates the auth cache.
    # Group claims grant team access; management membership is added by the test.
    unwrap(
        proxy.transport.post(
            "/user/new",
            headers=proxy.transport.master,
            json=UserNewBody(
                user_id=identity.user_id, user_email=f"{identity.username}@example.com", user_role="internal_user"
            ),
            response_type=UserNewResponse,
        )
    )
    team_id: Final = proxy.create_team(TeamNewBody(team_alias=f"e2e-jwt-{marker}", team_id=identity.group))
    resources.defer(lambda: proxy.delete_team(team_id))
    return identity


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "migration_startup: isolated container startup tests run by the migration CI workflow"
    )
    config.addinivalue_line(
        "markers",
        "provider_live: requires actual provider timing, limits, state, or a response that echoes this"
        " run's own unique value; bypass shared cache",
    )
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
    config.addinivalue_line(
        "markers",
        "prompt_caching_stack: needs a proxy running with router_settings.optional_pre_call_checks including "
        "prompt_caching; deselected unless E2E_PROMPT_CACHING_STACK is set",
    )
    config.addinivalue_line(
        "markers",
        "cli_determinism: drives the real claude CLI for several seconds; deselected unless E2E_CLI_DETERMINISM is set",
    )
    config.addinivalue_line(
        "markers",
        "redis_chaos: load test that pauses the proxy's Redis outright mid-run; needs a proxy booted from "
        "gateway/redis_chaos_ci_config.yml on the same host, and is deselected unless E2E_REDIS_CHAOS is set",
    )
    config.addinivalue_line(
        "markers",
        "quiet_stack: measures the proxy itself, so it runs while no other test on this host is hitting the stack; "
        "every other test waits for it to finish",
    )
    config.addinivalue_line(
        "markers",
        "mcp_oauth_live: real Linear OAuth consent via a captured browser session; deselected unless "
        "E2E_MCP_OAUTH_LIVE is set",
    )
    config.addinivalue_line(
        "markers",
        "provider_edge_host: routes provider traffic through the pytest host's edge in every fixture mode, so the "
        "gateway must reach the pytest host; deselected unless E2E_PROVIDER_EDGE_HOST_REACHABLE is set",
    )
    config.addinivalue_line(
        "markers",
        "owned_gateway: boots its own proxy from source against the stack's Postgres, so it needs DATABASE_URL "
        "on the pytest host; deselected unless E2E_OWNED_GATEWAY is set",
    )
    config.addinivalue_line(
        "markers",
        "otel_v2: needs a proxy running with LITELLM_OTEL_V2=true; deselected unless E2E_OTEL_V2 is set",
    )
    config.addinivalue_line(
        "markers",
        "otel_tls: needs a stack whose gateway exports OTLP over TLS signed by the CA in SSL_CERT_FILE; deselected unless E2E_OTEL_EXPORTER_ENDPOINT is set",
    )
    config.addinivalue_line(
        "markers",
        "secret_manager: needs a proxy booted from gateway/secret_manager_<system>_ci_config.yml against that live "
        "secret manager; deselected unless E2E_SECRET_MANAGER names the backend (see secret_manager/secret_backends.py)",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Abort before collection when E2E_FIXTURE_MODE can never work: an unknown
    mode value, or replay against a missing, unreadable, or stale bundle (the
    stale message names the bundle's age). Live and record modes pass through."""
    reason = fixture_mode_collection_error(FIXTURE_MODE_RAW, FIXTURE_DIR, now=datetime.now(timezone.utc))
    if reason is not None:
        raise pytest.UsageError(reason)


def pytest_report_header(config: pytest.Config) -> list[str]:
    return fixture_report_lines(FIXTURE_MODE_RAW, FIXTURE_DIR, now=datetime.now(timezone.utc))


def _needs_unset_opt_in(item: pytest.Item) -> bool:
    return any(
        item.get_closest_marker(marker) is not None and not os.environ.get(opt_in_env)
        for marker, opt_in_env in OPT_IN_MARKERS.items()
    )


def _reaches_proxy(item: pytest.Item) -> bool:
    """True for a live test that talks to the shared proxy: `e2e`-marked and not a
    `migration_startup` test, which boots its own container instead."""
    return item.get_closest_marker("e2e") is not None and item.get_closest_marker("migration_startup") is None


def _uses_idle_rss(item: pytest.Item) -> bool:
    return isinstance(item, pytest.Function) and "idle_rss" in item.fixturenames


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect every test behind an opt-in marker whose env var is unset (see
    OPT_IN_MARKERS): those tests need a proxy configured differently from the
    default stack, so the coverage collector, which runs over the same collection,
    counts their cells only where they actually run.

    Attach the two custom signals (suite package and covered cell ids) to every
    remaining test's user_properties so the standard JUnit report (`--junitxml`)
    records them as `<property>` entries, on every outcome including skips and
    setup errors. Downstream (Loki/Grafana) reads outcome and duration from the
    standard report and these properties for package rollups and coverage
    drill-down. See junit_properties.py.

    Also sort `load`-marked items last so a whole-tree run drives heavy throughput
    traffic only after the latency-sensitive suites have finished."""
    deselected = [item for item in items if _needs_unset_opt_in(item)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if not _needs_unset_opt_in(item)]
    for item in items:
        attach_result_properties(item)
    if os.environ.get("LITELLM_MIGRATION_TESTS") != "1":
        deselected = [item for item in items if item.get_closest_marker("migration_startup") is not None]
        items[:] = [item for item in items if item.get_closest_marker("migration_startup") is None]
        if deselected:
            deselected[0].config.hook.pytest_deselected(items=deselected)
    items.sort(key=lambda item: item.get_closest_marker("load") is not None)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_finish(session: pytest.Session) -> None:
    """When a selected test asks for the `idle_rss` fixture and this is not a
    `--collect-only` run, read every replica's RSS once, right here at the end of
    collection and before this process sends any traffic. tryfirst keeps the read
    ahead of xdist's own collection-finish report, and the controller schedules no
    test until every worker has reported, so this is the idle footprint of a stack
    that just passed its readiness gate. The fixture hands the capture to the
    idle-budget test in router/test_reliability_memory_e2e.py."""
    if session.config.getoption("collectonly") or not any(_uses_idle_rss(item) for item in session.items):
        return
    session.config.stash[_IDLE_RSS] = read_rss_everywhere(build_proxy_client(), timeout=IDLE_RSS_READ_TIMEOUT_SECONDS)


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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, object, object]:
    with stack_lock(exclusive=item.get_closest_marker("quiet_stack") is not None):
        return (yield)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Hard-fail `e2e`-marked tests unless a proxy answers its liveness probe.
    Unmarked tests (unit coverage of the harness) don't touch the proxy, so they
    run even when none is up. Never skip for a missing proxy. Replay mode needs
    the proxy too: only provider-bound traffic replays from the bundle."""
    LIVE_PROVIDER_REQUIRED.set(item.get_closest_marker("provider_live") is not None)
    if _uses_idle_rss(item):
        item.user_properties.extend(item.config.stash[_IDLE_RSS].junit_properties)
    if not _reaches_proxy(item):
        return
    if isinstance(item, pytest.Function) and "oauth_gateway" in item.fixturenames:
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
    if not _reaches_proxy(item):
        return
    item.session.stash[_E2E_TEST_RAN] = True


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Stash the call-phase outcome so teardown can tell a passed test from a
    failed one without re-deriving it."""
    report = yield
    if item.get_closest_marker("mcp_oauth_live") is not None and call.excinfo is not None:
        # Publish code locations only, never exception messages, source text or locals.
        item.user_properties.append(("oauth_failure_phase", report.when))
        item.user_properties.append(("oauth_exception_type", call.excinfo.type.__name__))
        for entry in call.excinfo.traceback:
            item.user_properties.append(("oauth_frame", f"{Path(entry.path).name}:{entry.lineno + 1}:{entry.name}"))
        report.user_properties = list(item.user_properties)
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
    LIVE_PROVIDER_REQUIRED.set(False)
    if not item.stash.get(_CALL_PASSED, False):
        return result
    reason = replay_leftover_error(mode_raw=FIXTURE_MODE_RAW, bundle_dir=FIXTURE_DIR, test_key=item.nodeid)
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


@pytest.fixture(scope="session")
def idle_rss(request: pytest.FixtureRequest) -> RssCapture:
    """Every replica's RSS as read once at the end of collection, before this process
    sent any traffic (see pytest_collection_finish)."""
    return request.config.stash[_IDLE_RSS]


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
