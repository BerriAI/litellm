from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from ...shared.reporting.models import HarnessCase, HarnessRun, ResultArtifact, RunStatus, SdkFunction
from ...shared.reporting.strategy import SuiteCaseSpec, UpdateCallback
from .mapping_report import mapping_report_lines
from .mapping_validator import MappingReport, MappingSuite, audit_mapping
from .mappings import MAPPING_SUITES

MAPPING_REPORT_ARTIFACT: Final = "mapping_report"


def _audit_problems(report: MappingReport) -> tuple[str, ...]:
    return (
        *(f"mapped Python test does not exist: {nodeid}" for nodeid in report.missing_python_tests),
        *(f"mapped Rust test does not exist: {nodeid}" for nodeid in report.missing_rust_tests),
        *(f"Python test has multiple mappings: {nodeid}" for nodeid in report.duplicate_python_mappings),
        *(f"unit parity exclusion does not exist: {nodeid}" for nodeid in report.invalid_unit_parity_exclusions),
    )


def run_suite(suite: MappingSuite, repo_root: Path, pytest_args: Sequence[str] = ()) -> tuple[str, ...]:
    del pytest_args
    return _audit_problems(audit_mapping(suite, repo_root))


def _run_mapping_case(
    case: HarnessCase,
    repo_root: Path,
    report: HarnessRun,
    on_update: UpdateCallback,
    suites: Mapping[SdkFunction, MappingSuite],
) -> None:
    result: Final = report.results[case.key]
    spec: Final = case.spec
    if not isinstance(spec, SuiteCaseSpec):
        return
    nodeid: Final = f"suite:{case.strategy_id}:{case.sdk_function}:{spec.suite}"
    result.collected.add(nodeid)
    result.status = RunStatus.RUNNING
    on_update(report)
    suite: Final = suites.get(case.sdk_function)
    if suite is None:
        result.record(nodeid, RunStatus.ERROR)
        report.failures.append((nodeid, f"no mapping suite registered for {case.sdk_function}"))
        return
    try:
        audit: Final = audit_mapping(suite, repo_root)
    except (OSError, ValueError) as error:
        result.record(nodeid, RunStatus.ERROR)
        report.failures.append((nodeid, str(error)))
        return
    problems: Final = _audit_problems(audit)
    artifact: Final = ResultArtifact(MAPPING_REPORT_ARTIFACT, "\n".join(mapping_report_lines(audit)))
    result.record(nodeid, RunStatus.FAILED if problems else RunStatus.PASSED, artifacts=(artifact,))
    report.failures.extend((nodeid, problem) for problem in problems)
    on_update(report)


def run_mapping_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
    *,
    suites: Mapping[SdkFunction, MappingSuite] = MAPPING_SUITES,
) -> tuple[int, HarnessRun]:
    del runner_args
    report: Final = HarnessRun.from_cases(cases)
    for case in cases:
        _run_mapping_case(case, repo_root, report, on_update, suites)
    report.finished_at = monotonic()
    on_update(report)
    failed: Final = any(
        result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING} for result in report.results.values()
    )
    return int(failed), report
