from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from ...shared.reporting.models import CaseResult, HarnessCase, HarnessRun, ResultArtifact, RunStatus, Surface
from ...shared.reporting.strategy import ModuleCaseSpec, UpdateCallback
from ...shared.native_build import ensure_trace_bridge
from .models import GatewayRouteSpec, RouteSpec, TraceExecutionFailure, TraceMode, TraceScenario, TraceSuite
from .reporting import TRACE_COMPARISON_ARTIFACT, TraceComparisonArtifact
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
    invalid_modes: Final = tuple(
        scenario.name
        for scenario in suite.scenarios
        if not scenario.modes
        or len(scenario.modes) != len(set(scenario.modes))
        or any(mode not in {"sync", "async"} for mode in scenario.modes)
    )
    if invalid_modes:
        return f"scenarios must use non-empty, unique sync/async modes: {', '.join(invalid_modes)}"
    surface: Final = harness_case.surface
    if surface == "sdk" and not isinstance(suite.route, RouteSpec):
        return "must use RouteSpec for the sdk surface"
    if surface == "gateway" and not isinstance(suite.route, GatewayRouteSpec):
        return "must use GatewayRouteSpec for the gateway surface"
    if surface is None:
        return "requires an sdk or gateway surface"
    if suite.route.route != harness_case.sdk_function:
        return f"route {suite.route.route} does not match case function {harness_case.sdk_function}"
    return None


def scenario_nodeids(
    trace_suite: TraceSuite,
    harness_case: HarnessCase,
    selected_scenarios: frozenset[str] = frozenset(),
) -> tuple[tuple[TraceScenario, TraceMode, str], ...]:
    surface: Final = harness_case.surface
    if surface is None:
        return ()
    return tuple(
        (scenario, mode, f"trace:{surface}:{harness_case.sdk_function}:{scenario.name}:{mode}")
        for scenario in trace_suite.scenarios
        if not selected_scenarios or scenario.name in selected_scenarios
        for mode in scenario.modes
    )


def _record_setup_failure(run: HarnessRun, case: HarnessCase, message: str, stage: str) -> None:
    result: Final = run.results[case.key]
    nodeid: Final = f"trace:{case.surface}:{case.sdk_function}:{stage}"
    result.collected.add(nodeid)
    result.record(nodeid, RunStatus.ERROR)
    run.failures.append((nodeid, message))


def run_trace_mode(
    run: HarnessRun,
    result: CaseResult,
    trace_suite: TraceSuite,
    scenario: TraceScenario,
    mode: TraceMode,
    surface: Surface,
    nodeid: str,
    on_update: UpdateCallback,
) -> None:
    started_at: Final = monotonic()
    comparison: Final = _execute_mode(trace_suite, scenario, mode, surface)
    duration: Final = monotonic() - started_at
    if isinstance(comparison, TraceExecutionFailure):
        result.record(nodeid, RunStatus.ERROR, duration)
        run.failures.append((nodeid, comparison.message))
        on_update(run)
        return
    artifact: Final = ResultArtifact(TRACE_COMPARISON_ARTIFACT, comparison.model_dump_json())
    if comparison.has_errors():
        result.record(nodeid, RunStatus.ERROR, duration, (artifact,))
        run.failures.append(
            (nodeid, "\n".join(error for error in (comparison.python_error, comparison.rust_error) if error))
        )
    else:
        status: Final = RunStatus.PASSED if comparison.contract_matches() else RunStatus.FAILED
        result.record(nodeid, status, duration, (artifact,))
        if status is RunStatus.FAILED:
            run.failures.append((nodeid, "trace contract mismatch; see the rendered comparison"))
    on_update(run)


def _execute_mode(
    trace_suite: TraceSuite,
    scenario: TraceScenario,
    mode: TraceMode,
    surface: Surface,
) -> TraceComparisonArtifact | TraceExecutionFailure:
    route: Final = trace_suite.route
    if isinstance(route, GatewayRouteSpec):
        if surface != "gateway":
            return TraceExecutionFailure("harness", "gateway route cannot run on the sdk surface")
        from .gateway.execution import execute_gateway_trace

        return execute_gateway_trace(route, scenario, mode)
    if surface != "sdk":
        return TraceExecutionFailure("harness", "sdk route cannot run on the gateway surface")
    return execute_trace(route, scenario, mode, surface)


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
    result.collected.update(nodeid for _, _, nodeid in nodeids)
    if not nodeids:
        result.status = RunStatus.SKIPPED
        on_update(run)
        return
    result.status = RunStatus.RUNNING
    on_update(run)
    for scenario, mode, nodeid in nodeids:
        run_trace_mode(run, result, trace_suite, scenario, mode, surface, nodeid, on_update)


def run_trace_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    selected_scenarios: Final = frozenset(runner_args)
    run: Final = HarnessRun.from_cases(cases)
    runnable_cases: Final = tuple(case for case in cases if isinstance(case.spec, ModuleCaseSpec))
    bridge_error: Final = ensure_trace_bridge(repo_root) if runnable_cases else None
    if bridge_error is not None:
        for harness_case in runnable_cases:
            _record_setup_failure(run, harness_case, bridge_error, "bridge")
        run.finished_at = monotonic()
        on_update(run)
        return 1, run
    for harness_case in cases:
        _run_case(run, harness_case, selected_scenarios, on_update)
    run.finished_at = monotonic()
    on_update(run)
    failed: Final = any(
        result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING} for result in run.results.values()
    )
    return int(failed), run
