from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import Final

import pytest

from .models import CaseResult, HarnessCase, HarnessRun, RunStatus

UpdateCallback = Callable[[HarnessRun], None]


def selector_matches_node(selector: str, nodeid: str) -> bool:
    normalized_selector = selector.replace("\\", "/")
    normalized_nodeid = nodeid.replace("\\", "/")
    if normalized_selector.endswith("/"):
        return normalized_nodeid.startswith(normalized_selector)
    if "::" in normalized_selector:
        return normalized_nodeid == normalized_selector or normalized_nodeid.startswith(
            f"{normalized_selector}["
        )
    return normalized_nodeid == normalized_selector or normalized_nodeid.startswith(
        f"{normalized_selector}::"
    )


def selector_path(selector: str) -> Path:
    return Path(selector.split("::", 1)[0])


def runnable_selectors(
    cases: Sequence[HarnessCase], repo_root: Path
) -> tuple[str, ...]:
    selectors = {
        selector
        for case in cases
        for selector in case.selectors
        if (repo_root / selector_path(selector)).exists()
    }
    return tuple(sorted(selectors))


class HarnessPytestPlugin:
    def __init__(self, run: HarnessRun, on_update: UpdateCallback) -> None:
        self.run = run
        self.on_update = on_update
        self.node_to_results: dict[str, list[CaseResult]] = {}

    def _notify(self) -> None:
        self.on_update(self.run)

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        for item in items:
            matched_results: list[CaseResult] = []
            for result in self.run.results.values():
                if any(
                    selector_matches_node(selector, item.nodeid)
                    for selector in result.case.selectors
                ):
                    result.collected.add(item.nodeid)
                    matched_results.append(result)
            if matched_results:
                self.node_to_results[item.nodeid] = matched_results
        for result in self.run.results.values():
            if result.status is RunStatus.QUEUED and not result.collected:
                result.status = RunStatus.MISSING
        self._notify()

    def pytest_runtest_logstart(
        self, nodeid: str, location: tuple[str, int | None, str]
    ) -> None:
        del location
        self.run.current_nodeid = nodeid
        for result in self.node_to_results.get(nodeid, []):
            if result.status not in {RunStatus.FAILED, RunStatus.ERROR}:
                result.status = RunStatus.RUNNING
        self._notify()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when not in {"setup", "call", "teardown"}:
            return
        results = self.node_to_results.get(report.nodeid, [])
        if not results:
            return

        terminal = report.when == "call" or report.failed or report.skipped
        if not terminal:
            for result in results:
                result.durations[report.nodeid] = (
                    result.durations.get(report.nodeid, 0.0) + report.duration
                )
            return
        for result in results:
            if report.when == "teardown" and not report.failed:
                result.durations[report.nodeid] = (
                    result.durations.get(report.nodeid, 0.0) + report.duration
                )
                continue
            if report.skipped:
                status = RunStatus.SKIPPED
            elif report.failed and report.when in {"setup", "teardown"}:
                status = RunStatus.ERROR
            elif report.failed:
                status = RunStatus.FAILED
            else:
                status = RunStatus.PASSED
            result.record(report.nodeid, status, report.duration)
        if report.failed:
            failure = (report.nodeid, str(report.longrepr))
            if failure not in self.run.failures:
                self.run.failures.append(failure)
        self._notify()

    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: int | pytest.ExitCode
    ) -> None:
        del session, exitstatus
        self.run.current_nodeid = None
        self.run.finished_at = monotonic()
        for result in self.run.results.values():
            result.finalize()
        self._notify()


def run_pytest(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    pytest_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    run = HarnessRun.from_cases(cases)
    selectors = runnable_selectors(cases, repo_root)
    if not selectors:
        for result in run.results.values():
            result.finalize()
        run.finished_at = monotonic()
        on_update(run)
        has_missing_test = any(
            result.status is RunStatus.MISSING for result in run.results.values()
        )
        exit_code = (
            int(pytest.ExitCode.TESTS_FAILED)
            if has_missing_test
            else int(pytest.ExitCode.OK)
        )
        return exit_code, run

    plugin = HarnessPytestPlugin(run=run, on_update=on_update)
    args: Final = (*selectors, "-q", "--tb=no", "--no-summary", "-o", "consider_namespace_packages=true", *pytest_args)
    previous_directory = Path.cwd()
    try:
        os.chdir(repo_root)
        exit_code = int(pytest.main(list(args), plugins=[plugin]))
    finally:
        os.chdir(previous_directory)
    for result in run.results.values():
        missing = tuple(
            selector for selector in result.case.selectors
            if not any(selector_matches_node(selector, node) for node in result.collected)
        )
        if missing:
            result.status = RunStatus.MISSING
            run.failures.extend((selector, "Configured selector collected no tests") for selector in missing)
    on_update(run)
    if exit_code == 0 and any(
        result.status is RunStatus.MISSING for result in run.results.values()
    ):
        exit_code = int(pytest.ExitCode.TESTS_FAILED)
    return exit_code, run
