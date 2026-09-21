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
from tests.integration._support.manifest import OWNED_DIRECTORIES, contracts

COLLECTED: Final = pytest.StashKey[tuple[str, ...]]()
REPORTS: Final = pytest.StashKey[list[pytest.TestReport]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--integration-order-seed", type=int, default=0)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: owned real-service integration contracts")
    config.addinivalue_line("markers", "covers(*ids): independently asserted behavior contracts")
    config.stash[REPORTS] = []
    config.pluginmanager.register(IntegrationReportPlugin(config))


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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    order_seed: Final = config.getoption("integration_order_seed")
    if order_seed:
        # rebind-ok: pytest requires this hook to reorder its shared collection list in place.
        items.sort(key=lambda item: hashlib.sha256(f"{order_seed}:{item.nodeid}".encode()).digest())
    manifest: Final = contracts()
    root: Final = Path(__file__).parent
    owned: Final = tuple(
        item
        for item in items
        if item.path.is_relative_to(root) and item.path.relative_to(root).parts[0] in OWNED_DIRECTORIES
    )
    if owned and os.environ.get("GITHUB_ACTIONS") == "true":
        raise pytest.UsageError("Integration contracts are owned by CircleCI")
    for item in owned:
        if item.nodeid not in manifest:
            raise pytest.UsageError(f"Integration node missing from manifest: {item.nodeid}")
        item.add_marker(pytest.mark.integration)
        declared: Final = tuple(value for mark in item.iter_markers("covers") for value in mark.args)
        if set(declared) != set(manifest[item.nodeid]):
            raise pytest.UsageError(f"Contract mapping differs for {item.nodeid}")
    config.stash[COLLECTED] = tuple(item.nodeid for item in owned)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if hasattr(session.config, "workerinput"):
        return
    destination: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    if destination is None:
        return
    collected: Final = session.config.stash.get(COLLECTED, ())
    reports: Final = tuple(report for report in session.config.stash[REPORTS] if report.nodeid in collected)
    passed: Final = tuple(report.nodeid for report in reports if report.when == "call" and report.passed)
    complete: Final = (
        exitstatus == 0
        and bool(collected)
        and sorted(collected) == sorted(passed)
        and all(report.passed for report in reports)
    )
    output: Final = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    (output / "execution.json").write_text(
        json.dumps({
            "collected": collected, "passed": passed, "complete": complete, "exitstatus": exitstatus,
            "hypothesis_version": version("hypothesis"),
            "hypothesis_seed": session.config.getoption("hypothesis_seed"),
            "order_seed": session.config.getoption("integration_order_seed"),
            "generation": {
                "max_examples": LIFECYCLE_SETTINGS.max_examples,
                "stateful_step_count": LIFECYCLE_SETTINGS.stateful_step_count,
                "database": str(LIFECYCLE_SETTINGS.database),
                "phases": [phase.name for phase in LIFECYCLE_SETTINGS.phases],
            },
        }, indent=2)
        + "\n"
    )
    if not complete and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def gateway() -> Iterator[Gateway]:
    with gateway_from_environment() as value:
        yield value


@pytest.fixture
def peer(gateway: Gateway) -> Iterator[Gateway]:
    url: Final = os.environ["INTEGRATION_PEER_URL"]
    assert url.rstrip("/") != str(gateway.client.base_url).rstrip("/")
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        eventually(lambda: cache.pubsub_numsub("litellm_proxy.auth_cache_invalidation")[0][1], lambda count: count >= 2)
    with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
        yield Gateway(client, gateway.key, gateway.upstream_url)
