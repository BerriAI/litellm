from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Iterator

import pytest

from .models import CaseResult, HarnessCase, HarnessRun, RunStatus

UpdateCallback = Callable[[HarnessRun], None]


def _env_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    """Apply environment overrides for the duration of the block, then restore.

    `LITELLM_USE_RUST_OCR` is read once at `litellm.rust_bridge.ocr` import
    time, so mutating `os.environ` alone has no effect once that module is
    already loaded in this process (as it will be across repeated in-process
    `pytest.main()` calls). Re-apply it through `use_litellm_rust()`, the
    module's own runtime toggle, so an OCR variant actually switches paths.
    """
    if not overrides:
        yield
        return
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    rust_ocr_module = None
    previous_rust_ocr_enabled = False
    if "LITELLM_USE_RUST_OCR" in overrides:
        from litellm.rust_bridge import ocr as rust_ocr_module

        previous_rust_ocr_enabled = rust_ocr_module.rust_ocr_enabled()
        rust_ocr_module.use_litellm_rust(
            enabled=_env_truthy(overrides["LITELLM_USE_RUST_OCR"])
        )
    try:
        yield
    finally:
        if rust_ocr_module is not None:
            rust_ocr_module.use_litellm_rust(enabled=previous_rust_ocr_enabled)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def selector_matches_node(selector: str, nodeid: str) -> bool:
    normalized_selector = selector.replace("\\", "/")
    normalized_nodeid = nodeid.replace("\\", "/")
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
    env: dict[str, str] | None = None,
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
    args = [*selectors, "-p", "no:terminal", *pytest_args]
    previous_directory = Path.cwd()
    try:
        os.chdir(repo_root)
        with _temporary_env(env or {}):
            exit_code = int(pytest.main(args, plugins=[plugin]))
    finally:
        os.chdir(previous_directory)
    if exit_code == 0 and any(
        result.status is RunStatus.MISSING for result in run.results.values()
    ):
        exit_code = int(pytest.ExitCode.TESTS_FAILED)
    return exit_code, run


def run_pytest_grouped(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    strategy_env: dict[str, dict[str, str]],
    pytest_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    """Run `cases` once per distinct owning-strategy environment.

    A strategy that declares `variants` (for example Python vs. Rust via
    `LITELLM_USE_RUST_OCR`) produces one `HarnessCase` per variant, each
    carrying its own `strategy_id`. This groups cases by their strategy's
    environment overrides, runs pytest once per group under that
    environment, and merges every group's results into one combined run so
    a variant that fails is never silently overwritten by one that passes.
    """
    groups: dict[tuple[tuple[str, str], ...], list[HarnessCase]] = {}
    for case in cases:
        env = strategy_env.get(case.strategy_id, {})
        groups.setdefault(tuple(sorted(env.items())), []).append(case)

    combined = HarnessRun.from_cases(cases)

    def merge_update(sub_run: HarnessRun) -> None:
        combined.results.update(sub_run.results)
        combined.current_nodeid = sub_run.current_nodeid
        for failure in sub_run.failures:
            if failure not in combined.failures:
                combined.failures.append(failure)
        on_update(combined)

    exit_code = 0
    for env_items, group_cases in groups.items():
        sub_exit_code, _ = run_pytest(
            group_cases,
            repo_root,
            merge_update,
            pytest_args,
            env=dict(env_items),
        )
        exit_code = exit_code or sub_exit_code
    combined.finished_at = monotonic()
    on_update(combined)
    return exit_code, combined
