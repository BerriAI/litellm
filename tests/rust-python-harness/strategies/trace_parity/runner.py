from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from ...shared.reporting.models import CaseResult, HarnessCase, HarnessRun, ResultArtifact, RunStatus, Surface
from ...shared.reporting.strategy import ModuleCaseSpec, UpdateCallback
from .reporting import TRACE_COMPARISON_ARTIFACT
from .sdk.execution import TraceCase, TraceExecutionFailure, TraceMode, execute_trace


def _load_case(reference: str) -> TraceCase | TraceExecutionFailure:
    try:
        module: Final = importlib.import_module(reference)
    except Exception as error:
        return TraceExecutionFailure("harness", f"cannot import {reference}: {type(error).__name__}: {error}")
    case: Final = getattr(module, "TRACE_CASE", None)
    if not isinstance(case, TraceCase):
        return TraceExecutionFailure("harness", f"{reference} must export TRACE_CASE: TraceCase")
    return case


def _record_load_failure(
    run: HarnessRun,
    case: HarnessCase,
    failure: TraceExecutionFailure,
) -> None:
    result: Final = run.results[case.key]
    nodeid: Final = f"trace:{case.surface}:{case.sdk_function}:load"
    result.collected.add(nodeid)
    result.record(nodeid, RunStatus.ERROR)
    run.failures.append((nodeid, failure.message))


def _run_mode(
    run: HarnessRun,
    result: CaseResult,
    trace_case: TraceCase,
    mode: TraceMode,
    surface: Surface,
    nodeid: str,
    on_update: UpdateCallback,
) -> None:
    started_at: Final = monotonic()
    comparison: Final = execute_trace(trace_case, mode, surface)
    duration: Final = monotonic() - started_at
    if isinstance(comparison, TraceExecutionFailure):
        result.record(nodeid, RunStatus.ERROR, duration)
        run.failures.append((nodeid, f"{comparison.engine}: {comparison.message}"))
    else:
        artifact: Final = ResultArtifact(TRACE_COMPARISON_ARTIFACT, comparison.model_dump_json())
        status: Final = RunStatus.PASSED if comparison.contract_matches() else RunStatus.FAILED
        result.record(nodeid, status, duration, (artifact,))
        if status is RunStatus.FAILED:
            run.failures.append((nodeid, "trace contract mismatch; see the rendered comparison"))
    on_update(run)


def _run_case(run: HarnessRun, harness_case: HarnessCase, on_update: UpdateCallback) -> None:
    result: Final = run.results[harness_case.key]
    spec: Final = harness_case.spec
    if not isinstance(spec, ModuleCaseSpec):
        return
    trace_case: Final = _load_case(spec.module)
    if isinstance(trace_case, TraceExecutionFailure):
        _record_load_failure(run, harness_case, trace_case)
        on_update(run)
        return
    nodeids: Final[tuple[tuple[TraceMode, str], ...]] = tuple(
        (mode, f"trace:{harness_case.surface}:{harness_case.sdk_function}:{mode}") for mode in trace_case.modes
    )
    result.collected.update(nodeid for _, nodeid in nodeids)
    result.status = RunStatus.RUNNING
    on_update(run)
    for mode, nodeid in nodeids:
        _run_mode(run, result, trace_case, mode, harness_case.surface, nodeid, on_update)


def run_trace_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    del repo_root, runner_args
    run: Final = HarnessRun.from_cases(cases)
    for harness_case in cases:
        _run_case(run, harness_case, on_update)
    run.finished_at = monotonic()
    on_update(run)
    failed: Final = any(
        result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING} for result in run.results.values()
    )
    return int(failed), run
