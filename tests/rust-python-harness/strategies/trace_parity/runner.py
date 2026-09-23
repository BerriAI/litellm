from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from ...shared.reporting.models import CaseResult, HarnessCase, HarnessRun, ResultArtifact, RunStatus, Surface
from ...shared.reporting.strategy import ModuleCaseSpec, UpdateCallback
from .models import (
    TraceExecutionFailure,
    TraceScenario,
    TraceSuite,
)
from .reporting import TRACE_ARTIFACT, TraceArtifact
from .sdk.execution import execute_trace


def _load_case(reference: str, harness_case: HarnessCase) -> TraceSuite | TraceExecutionFailure:
    try:
        module: Final = importlib.import_module(reference)
    except Exception as error:
        return TraceExecutionFailure("harness", f"cannot import {reference}: {type(error).__name__}: {error}")
    suite: Final = getattr(module, "TRACE_SUITE", None)
    if not isinstance(suite, TraceSuite):
        return TraceExecutionFailure("harness", f"{reference} must export TRACE_SUITE: TraceSuite")
    validation_error: Final = validate_trace_suite(suite, harness_case)
    if validation_error is not None:
        return TraceExecutionFailure("harness", f"{reference} {validation_error}")
    return suite


def validate_trace_suite(suite: TraceSuite, harness_case: HarnessCase) -> str | None:
    names: Final = tuple(scenario.name for scenario in suite.scenarios)
    if not names or len(names) != len(set(names)) or any(not name or ":" in name for name in names):
        return "scenario names must be non-empty, unique, and colon-free"
    invalid_names: Final = tuple(
        scenario.name
        for scenario in suite.scenarios
        if not scenario.name.startswith("async-" if scenario.asynchronous else "sync-")
    )
    if invalid_names:
        return f"scenario names must start with sync- or async-: {', '.join(invalid_names)}"
    surface: Final = harness_case.surface
    if surface != "sdk":
        return "requires the sdk surface"
    if suite.route.route != harness_case.sdk_function:
        return f"route {suite.route.route} does not match case function {harness_case.sdk_function}"
    return None


def scenario_nodeids(
    trace_suite: TraceSuite,
    harness_case: HarnessCase,
    selected_scenarios: frozenset[str] = frozenset(),
) -> tuple[tuple[TraceScenario, str], ...]:
    surface: Final = harness_case.surface
    if surface is None:
        return ()
    return tuple(
        (scenario, f"trace:{surface}:{harness_case.sdk_function}:{scenario.name}")
        for scenario in trace_suite.scenarios
        if not selected_scenarios or scenario.name in selected_scenarios
    )


def _record_setup_failure(run: HarnessRun, case: HarnessCase, message: str, stage: str) -> None:
    result: Final = run.results[case.key]
    nodeid: Final = f"trace:{case.surface}:{case.sdk_function}:{stage}"
    result.collected.add(nodeid)
    result.record(nodeid, RunStatus.ERROR)
    run.failures.append((nodeid, message))


def run_trace_scenario(
    run: HarnessRun,
    result: CaseResult,
    trace_suite: TraceSuite,
    scenario: TraceScenario,
    surface: Surface,
    nodeid: str,
    on_update: UpdateCallback,
) -> None:
    started_at: Final = monotonic()
    trace: Final = _execute_scenario(trace_suite, scenario, surface)
    duration: Final = monotonic() - started_at
    if isinstance(trace, TraceExecutionFailure):
        result.record(nodeid, RunStatus.ERROR, duration)
        run.failures.append((nodeid, trace.message))
        on_update(run)
        return
    artifact: Final = ResultArtifact(TRACE_ARTIFACT, trace.model_dump_json())
    if trace.has_errors():
        result.record(nodeid, RunStatus.ERROR, duration, (artifact,))
        run.failures.append((nodeid, trace.python_error or ""))
    else:
        result.record(nodeid, RunStatus.PASSED, duration, (artifact,))
    on_update(run)


def _execute_scenario(
    trace_suite: TraceSuite,
    scenario: TraceScenario,
    surface: Surface,
) -> TraceArtifact | TraceExecutionFailure:
    if surface != "sdk":
        return TraceExecutionFailure("harness", "trace scenarios only run on the sdk surface")
    return execute_trace(trace_suite.route, scenario, surface)


def _run_case(
    run: HarnessRun,
    harness_case: HarnessCase,
    selected_scenarios: frozenset[str],
    on_update: UpdateCallback,
) -> None:
    result: Final = run.results[harness_case.key]
    spec: Final = harness_case.spec
    if not isinstance(spec, ModuleCaseSpec):
        return
    surface: Final = harness_case.surface
    if surface is None:
        return
    trace_suite: Final = _load_case(spec.module, harness_case)
    if isinstance(trace_suite, TraceExecutionFailure):
        _record_setup_failure(run, harness_case, trace_suite.message, "load")
        on_update(run)
        return
    nodeids: Final = scenario_nodeids(trace_suite, harness_case, selected_scenarios)
    result.collected.update(nodeid for _, nodeid in nodeids)
    if not nodeids:
        result.status = RunStatus.SKIPPED
        on_update(run)
        return
    result.status = RunStatus.RUNNING
    on_update(run)
    for scenario, nodeid in nodeids:
        run_trace_scenario(run, result, trace_suite, scenario, surface, nodeid, on_update)


def run_trace_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    del repo_root
    selected_scenarios: Final = frozenset(runner_args)
    run: Final = HarnessRun.from_cases(cases)
    for harness_case in cases:
        _run_case(run, harness_case, selected_scenarios, on_update)
    run.finished_at = monotonic()
    on_update(run)
    failed: Final = any(
        result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING} for result in run.results.values()
    )
    return int(failed), run
