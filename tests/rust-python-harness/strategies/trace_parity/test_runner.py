from __future__ import annotations

from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase, HarnessRun, RunStatus, SdkFunction, Surface
from ...shared.reporting.strategy import ModuleCaseSpec
from ...shared.tracing.steps import Engine
from .models import GatewayRouteSpec, RouteFixture, RouteSpec, TraceScenario, TraceSuite
from .runner import run_trace_mode, scenario_nodeids, validate_trace_suite


def _fixture(_engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(kwargs={}, provider_responses=())


def _case(*, surface: Surface = "sdk", function: SdkFunction = "ocr") -> HarnessCase:
    return HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function=function,
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface=surface,
    )


def test_scenario_filtering_and_occurrence_node_ids() -> None:
    suite: Final = TraceSuite(
        route=RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture),
        scenarios=(
            TraceScenario("one", _fixture, (), modes=("sync", "async")),
            TraceScenario("two", _fixture, (), modes=("async",)),
        ),
    )
    case: Final = _case()

    nodes: Final = scenario_nodeids(suite, case, frozenset({"two"}))

    assert tuple(nodeid for _, _, nodeid in nodes) == ("trace:sdk:ocr:two:async",)


def test_scenario_validation_rejects_duplicate_and_unsafe_names() -> None:
    route: Final = RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture)
    duplicate: Final = TraceSuite(
        route=route,
        scenarios=(TraceScenario("same", _fixture, ()), TraceScenario("same", _fixture, ())),
    )
    unsafe: Final = TraceSuite(route=route, scenarios=(TraceScenario("bad:name", _fixture, ()),))
    case: Final = _case()

    assert validate_trace_suite(duplicate, case) is not None
    assert validate_trace_suite(unsafe, case) is not None


def test_scenario_validation_rejects_invalid_modes_and_route_registration() -> None:
    invalid_modes: Final = TraceSuite(
        route=RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture),
        scenarios=(TraceScenario("invalid", _fixture, (), modes=("sync", "sync")),),
    )
    wrong_function: Final = TraceSuite(
        route=RouteSpec("messages", ("create", "acreate"), ("messages", "amessages"), _fixture),
        scenarios=(TraceScenario("one", _fixture, ()),),
    )
    wrong_surface: Final = TraceSuite(
        route=GatewayRouteSpec("ocr"),
        scenarios=(TraceScenario("one", _fixture, ()),),
    )
    case: Final = _case()

    assert "unique sync/async modes" in (validate_trace_suite(invalid_modes, case) or "")
    assert "does not match case function" in (validate_trace_suite(wrong_function, case) or "")
    assert "must use RouteSpec" in (validate_trace_suite(wrong_surface, case) or "")


def test_invalid_route_dispatch_records_harness_error() -> None:
    case: Final = _case()
    run: Final = HarnessRun.from_cases((case,))
    result: Final = run.results[case.key]
    suite: Final = TraceSuite(
        route=GatewayRouteSpec("ocr"),
        scenarios=(TraceScenario("one", _fixture, (), modes=("sync",)),),
    )
    nodeid: Final = "trace:sdk:ocr:one:sync"

    run_trace_mode(run, result, suite, suite.scenarios[0], "sync", "sdk", nodeid, lambda _: None)

    assert result.outcomes[nodeid] is RunStatus.ERROR
    assert run.failures == [(nodeid, "gateway route cannot run on the sdk surface")]
