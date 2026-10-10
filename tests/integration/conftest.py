from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Final

import httpx
import pytest
from redis import Redis

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.generation import LIFECYCLE_SETTINGS
from tests.integration._support.manifest import OWNED_DIRECTORIES
from tests.integration._support.ordering import case_shards, collected_case, fixture_groups
from tests.integration._support.provider import SharedProvider, shared_provider
from tests.integration._support.routing import RoutingPlugin
from tests.integration.run import GITHUB_FILES

COLLECTED: Final = pytest.StashKey[tuple[str, ...]]()
INVENTORY: Final = pytest.StashKey[tuple[str, ...]]()
REPORTS: Final = pytest.StashKey[list[pytest.TestReport]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--integration-order-seed", type=int, default=0)
    parser.addoption("--integration-shard-count", type=int, default=1)
    parser.addoption("--integration-shard-index", type=int, default=0)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: owned real-service integration contracts")
    config.addinivalue_line("markers", "covers(*ids): legacy contract IDs kept for existing tests, not enforced")
    config.stash[REPORTS] = []
    config.pluginmanager.register(IntegrationReportPlugin(config))
    if os.environ.get("INTEGRATION_ROUTING"):
        config.pluginmanager.register(RoutingPlugin(config))


class IntegrationReportPlugin:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.config.stash[REPORTS].append(report)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: object, ids: Sequence[str]) -> None:
        self.config.stash[COLLECTED] = tuple(nodeid for nodeid in ids if _owned(nodeid))


def _owned(nodeid: str) -> bool:
    parts: Final = Path(nodeid.split("::", 1)[0]).parts
    return parts[:2] == ("tests", "integration") and len(parts) > 3 and parts[2] in OWNED_DIRECTORIES


def _digest(seed: int, identity: str) -> bytes:
    return hashlib.sha256(f"{seed}:{identity}".encode()).digest()


def _order_key(seed: int, nodeid: str, group: str = "") -> tuple[bytes, bytes, bytes]:
    return _digest(seed, nodeid.split("::", 1)[0]), _digest(seed, group or nodeid), _digest(seed, nodeid)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    order_seed: Final = config.getoption("integration_order_seed")
    if order_seed:
        groups: Final = fixture_groups(tuple(collected_case(item) for item in items))
        items.sort(key=lambda item: _order_key(order_seed, item.nodeid, groups[item.nodeid]))
    root: Final = Path(__file__).parent
    owned: Final = tuple(
        item
        for item in items
        if item.path.is_relative_to(root) and item.path.relative_to(root).parts[0] in OWNED_DIRECTORIES
    )
    circleci_only: Final = tuple(
        item for item in owned if item.path.relative_to(root.parents[1]).as_posix() not in GITHUB_FILES
    )
    if circleci_only and os.environ.get("GITHUB_ACTIONS") == "true":
        raise pytest.UsageError("Integration contracts are owned by CircleCI")
    for item in owned:
        item.add_marker(pytest.mark.integration)
    config.stash[INVENTORY] = tuple(item.nodeid for item in owned)
    count: Final = config.getoption("integration_shard_count")
    index: Final = config.getoption("integration_shard_index")
    if count < 1 or not 0 <= index < count:
        raise pytest.UsageError("Integration shard index must be within the positive shard count")
    if count > 1:
        shards: Final = case_shards(tuple(collected_case(item) for item in owned), count)
        selected: Final = tuple(item for item in items if item not in owned or item.nodeid in shards[index])
        deselected: Final = tuple(item for item in owned if item.nodeid not in shards[index])
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
    config.stash[COLLECTED] = tuple(item.nodeid for item in items if item in owned)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if hasattr(session.config, "workerinput"):
        return
    destination: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    if destination is None:
        return
    collected: Final = session.config.stash.get(COLLECTED, ())
    reports: Final = tuple(report for report in session.config.stash[REPORTS] if report.nodeid in collected)
    passed: Final = tuple(report.nodeid for report in reports if report.when == "call" and report.passed)
    skipped: Final = tuple(report.nodeid for report in reports if report.skipped)
    complete: Final = (
        exitstatus == 0
        and bool(collected)
        and sorted(collected) == sorted(passed + skipped)
        and not any(report.failed for report in reports)
    )
    output: Final = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    (output / "execution.json").write_text(
        json.dumps(
            {
                "collected": collected,
                "inventory": session.config.stash.get(INVENTORY, collected),
                "shard_count": session.config.getoption("integration_shard_count"),
                "shard_index": session.config.getoption("integration_shard_index"),
                "passed": passed,
                "skipped": skipped,
                "complete": complete,
                "exitstatus": exitstatus,
                "hypothesis_version": version("hypothesis"),
                "hypothesis_seed": session.config.getoption("hypothesis_seed"),
                "order_seed": session.config.getoption("integration_order_seed"),
                "generation": {
                    "max_examples": LIFECYCLE_SETTINGS.max_examples,
                    "stateful_step_count": LIFECYCLE_SETTINGS.stateful_step_count,
                    "database": str(LIFECYCLE_SETTINGS.database),
                    "phases": [phase.name for phase in LIFECYCLE_SETTINGS.phases],
                },
            },
            indent=2,
        )
        + "\n"
    )
    if not complete and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def gateway() -> Iterator[Gateway]:
    with gateway_from_environment() as value:
        yield value


@pytest.fixture(scope="session")
def shared_provider_server() -> Iterator[SharedProvider]:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("the shared fake provider needs tests to run one at a time; this group runs under pytest-xdist")
    with shared_provider() as server:
        yield server
        late: Final = server.received()
        assert late == (), (
            f"the shared fake provider got {[item.target for item in late]} after {server.last_test} finished"
        )


@pytest.fixture
def provider(shared_provider_server: SharedProvider, request: pytest.FixtureRequest) -> Iterator[SharedProvider]:
    stray: Final = shared_provider_server.received()
    shared_provider_server.replies.clear()
    assert stray == (), (
        f"the shared fake provider got {[item.target for item in stray]} "
        f"after {shared_provider_server.last_test} finished"
    )
    yield shared_provider_server
    shared_provider_server.last_test = request.node.nodeid
    unused: Final = len(shared_provider_server.replies)
    unread: Final = shared_provider_server.received()
    shared_provider_server.replies.clear()
    assert unused == 0, f"{unused} queued provider replies were never requested"
    assert unread == (), f"the test never read the provider requests {[item.target for item in unread]}"


@pytest.fixture
def peer(gateway: Gateway) -> Iterator[Gateway]:
    url: Final = os.environ["INTEGRATION_PEER_URL"]
    assert url.rstrip("/") != str(gateway.client.base_url).rstrip("/")
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        eventually(lambda: cache.pubsub_numsub("litellm_proxy.auth_cache_invalidation")[0][1], lambda count: count >= 2)
    with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
        yield Gateway(client, gateway.key, gateway.upstream_url)
