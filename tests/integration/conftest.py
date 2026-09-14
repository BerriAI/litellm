from __future__ import annotations

import json
import os
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Final

import pytest

from integration._support.client import Gateway, gateway_from_environment
from integration._support.manifest import OWNED_DIRECTORIES, contracts

COLLECTED: Final = pytest.StashKey[tuple[str, ...]]()
REPORTS: Final = pytest.StashKey[list[pytest.TestReport]]()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: owned real-service integration contracts")
    config.addinivalue_line("markers", "covers(*ids): independently asserted behavior contracts")
    config.stash[REPORTS] = []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report: Final = yield
    item.config.stash[REPORTS].append(report)
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
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
        json.dumps({"collected": collected, "passed": passed, "complete": complete, "exitstatus": exitstatus}, indent=2)
        + "\n"
    )
    if not complete and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def gateway() -> Iterator[Gateway]:
    with gateway_from_environment() as value:
        yield value
