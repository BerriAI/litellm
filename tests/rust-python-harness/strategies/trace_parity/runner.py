from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final, cast

from ...shared.native_build import ensure_trace_bridge
from ...shared.reporting.models import CaseResult, HarnessCase, HarnessRun, ResultArtifact, RunStatus, Surface
from ...shared.reporting.strategy import ModuleCaseSpec, UpdateCallback
from .models import (
    GatewayRouteSpec,
    RouteSpec,
    TraceEngine,
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
    engine: TraceEngine = "both",
) -> None:
    started_at: Final = monotonic()
    trace: Final = _execute_scenario(trace_suite, scenario, surface, engine)
    duration: Final = monotonic() - started_at
    if isinstance(trace, TraceExecutionFailure):
        result.record(nodeid, RunStatus.ERROR, duration)
        run.failures.append((nodeid, trace.message))
        on_update(run)
        return
    artifact: Final = ResultArtifact(TRACE_ARTIFACT, trace.model_dump_json())
    if trace.has_errors():
        result.record(nodeid, RunStatus.ERROR, duration, (artifact,))
        run.failures.append((nodeid, "\n".join(error for error in (trace.python_error, trace.rust_error) if error)))
    else:
        result.record(nodeid, RunStatus.PASSED, duration, (artifact,))
    on_update(run)


def _execute_scenario(
    trace_suite: TraceSuite,
    scenario: TraceScenario,
    surface: Surface,
    engine: TraceEngine,
) -> TraceArtifact | TraceExecutionFailure:
    route: Final = trace_suite.route
    if isinstance(route, GatewayRouteSpec):
        if surface != "gateway":
            return TraceExecutionFailure("harness", "gateway route cannot run on the sdk surface")
        from .gateway.execution import execute_gateway_trace

        return execute_gateway_trace(route, scenario, engine)
    if surface != "sdk":
        return TraceExecutionFailure("harness", "sdk route cannot run on the gateway surface")
    return execute_trace(route, scenario, surface, engine)


def _run_case(
    run: HarnessRun,
    harness_case: HarnessCase,
    selected_scenarios: frozenset[str],
    on_update: UpdateCallback,
    engine: TraceEngine,
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
        run_trace_scenario(run, result, trace_suite, scenario, surface, nodeid, on_update, engine)


def runner_selection(runner_args: Sequence[str]) -> tuple[frozenset[str], TraceEngine]:
    engine: TraceEngine = "both"
    scenarios: list[str] = []
    for argument in runner_args:
        if argument.startswith("--engine="):
            value = argument.removeprefix("--engine=")
            if value not in {"python", "rust"}:
                raise ValueError(f"invalid trace engine: {value}")
            engine = cast(TraceEngine, value)
        else:
            scenarios.append(argument)
    return frozenset(scenarios), engine


def run_trace_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    selected_scenarios, engine = runner_selection(runner_args)
    run: Final = HarnessRun.from_cases(cases)
    runnable_cases: Final = tuple(case for case in cases if isinstance(case.spec, ModuleCaseSpec))
    bridge_error: Final = ensure_trace_bridge(repo_root) if runnable_cases and engine != "python" else None
    if bridge_error is not None:
        for harness_case in runnable_cases:
            _record_setup_failure(run, harness_case, bridge_error, "bridge")
        run.finished_at = monotonic()
        on_update(run)
        return 1, run
    for harness_case in cases:
        _run_case(run, harness_case, selected_scenarios, on_update, engine)
    run.finished_at = monotonic()
    on_update(run)
    failed: Final = any(
        result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING} for result in run.results.values()
    )
    return int(failed), run
