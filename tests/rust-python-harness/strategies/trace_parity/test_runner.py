from __future__ import annotations

from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase, HarnessRun, RunStatus, SdkFunction, Surface
from ...shared.reporting.strategy import ModuleCaseSpec
from ...shared.tracing.profiler import FunctionTraceEvent
from ...shared.tracing.steps import Engine, PipelineStep, mapping, trace_diff
from .models import GatewayRouteSpec, RouteFixture, RouteSpec, TraceScenario, TraceSuite
from .runner import run_trace_mode, scenario_nodeids, validate_trace_suite
from .sdk.execution import _combined_rust_projection


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


def test_public_scenario_requires_wrapper_mappings() -> None:
    suite: Final = TraceSuite(
        route=RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture),
        scenarios=(TraceScenario("public", _fixture, (), boundary="public_sdk"),),
    )

    assert "require Rust wrapper mappings" in (validate_trace_suite(suite, _case()) or "")


def test_public_projection_attaches_native_trace_to_rust_dispatch() -> None:
    mappings: Final = (
        mapping(rust_span="ocr", python_frame=r" main_ocr$"),
        mapping(rust_span="prepare", python_frame=r"prepare$"),
        mapping(rust_span="public_sdk_entrypoint"),
        mapping(rust_span="rust_bridge_dispatch"),
    )
    scenario: Final = TraceScenario(
        "public",
        _fixture,
        mappings,
        boundary="public_sdk",
        rust_wrapper_mappings=(
            mapping(span="public_sdk_entrypoint", python_frame=r" main_ocr$"),
            mapping(span="rust_bridge_dispatch", python_frame=r"_run_rust_ocr$"),
        ),
    )
    projection: Final = _combined_rust_projection(
        (
            FunctionTraceEvent(0, None, "ocr/main.py:1 main_ocr"),
            FunctionTraceEvent(1, 0, "ocr/main.py:2 ignored"),
            FunctionTraceEvent(2, 1, "ocr/main.py:3 _run_rust_ocr"),
        ),
        (
            FunctionTraceEvent(0, None, "ocr"),
            FunctionTraceEvent(1, 0, "prepare"),
        ),
        scenario,
        "sync",
    )

    assert tuple((step.span, step.parent_id) for step in projection.steps) == (
        ("public_sdk_entrypoint", None),
        ("rust_bridge_dispatch", 0),
        ("ocr", 1),
        ("prepare", 2),
    )
    python: Final = (PipelineStep(0, None, "ocr", "ocr"), PipelineStep(1, 0, "prepare", "prepare"))
    assert trace_diff(python, projection.steps, mappings).matches
    native_only: Final = (
        PipelineStep(0, None, "ocr", "ocr"),
        PipelineStep(1, 0, "prepare", "prepare"),
    )
    assert trace_diff(python, native_only, mappings).missing_mappings == (
        "public_sdk_entrypoint",
        "rust_bridge_dispatch",
    )


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
