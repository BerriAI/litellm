from __future__ import annotations

from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase
from ...shared.tracing.steps import Engine
from ...shared.reporting.strategy import ModuleCaseSpec
from .runner import scenario_nodeids, validate_trace_suite
from .sdk.execution import RouteFixture, RouteSpec, TraceScenario, TraceSuite


def _fixture(_engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(kwargs={}, provider_responses=())


def test_scenario_filtering_and_occurrence_node_ids() -> None:
    suite: Final = TraceSuite(
        route=RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture),
        scenarios=(
            TraceScenario("one", _fixture, (), modes=("sync", "async")),
            TraceScenario("two", _fixture, (), modes=("async",)),
        ),
    )
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function="ocr",
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface="sdk",
    )

    nodes: Final = scenario_nodeids(suite, case, frozenset({"two"}))

    assert tuple(nodeid for _, _, nodeid in nodes) == ("trace:sdk:ocr:two:async",)


def test_scenario_validation_rejects_duplicate_and_unsafe_names() -> None:
    route: Final = RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture)
    duplicate: Final = TraceSuite(
        route=route,
        scenarios=(TraceScenario("same", _fixture, ()), TraceScenario("same", _fixture, ())),
    )
    unsafe: Final = TraceSuite(route=route, scenarios=(TraceScenario("bad:name", _fixture, ()),))

    assert validate_trace_suite(duplicate) is not None
    assert validate_trace_suite(unsafe) is not None
