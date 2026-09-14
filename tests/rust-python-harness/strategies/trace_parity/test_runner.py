from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

import pytest

import litellm

from ...shared.reporting.models import Coverage, HarnessCase, HarnessRun, RunStatus, SdkFunction, Surface
from ...shared.reporting.strategy import ModuleCaseSpec
from ...shared.tracing.profiler import FunctionTraceEvent
from ...shared.tracing.steps import Engine, PipelineStep
from .models import GatewayRouteSpec, RouteFixture, RouteSpec, TraceScenario, TraceSuite
from .reporting import TraceArtifact
from .runner import run_trace_cases, run_trace_scenario, runner_selection, scenario_nodeids, validate_trace_suite
from .sdk.execution import SdkCall, collect_trace, execute_trace


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
            TraceScenario("sync-one", _fixture, (), asynchronous=False),
            TraceScenario("async-one", _fixture, (), asynchronous=True),
            TraceScenario("async-two", _fixture, (), asynchronous=True),
        ),
    )
    case: Final = _case()

    nodes: Final = scenario_nodeids(suite, case, frozenset({"async-two"}))

    assert tuple(nodeid for _, nodeid in nodes) == ("trace:sdk:ocr:async-two",)


def test_python_engine_is_separate_from_scenario_selection() -> None:
    assert runner_selection(("mistral", "--engine=python")) == (frozenset({"mistral"}), "python")


def test_python_engine_skips_native_bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner: Final = importlib.import_module("tests.rust-python-harness.strategies.trace_parity.runner")
    case: Final = _case()
    selected: list[tuple[frozenset[str], str]] = []

    def reject_bridge(_repo_root: Path) -> str | None:
        raise AssertionError("Python-only tracing must not inspect or build the native bridge")

    def capture_case(
        _run: HarnessRun,
        _case: HarnessCase,
        scenarios: frozenset[str],
        _on_update: object,
        engine: str,
    ) -> None:
        selected.append((scenarios, engine))

    monkeypatch.setattr(runner, "ensure_trace_bridge", reject_bridge)
    monkeypatch.setattr(runner, "_run_case", capture_case)

    exit_code, _ = run_trace_cases((case,), tmp_path, lambda _: None, ("mistral", "--engine=python"))

    assert exit_code == 0
    assert selected == [(frozenset({"mistral"}), "python")]


def test_python_trace_controls_native_ocr_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    execution: Final = importlib.import_module("tests.rust-python-harness.strategies.trace_parity.sdk.execution")
    route: Final = RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture)
    observed: list[str | None] = []

    def collect(
        _function: SdkCall,
        _fixture: RouteFixture,
        _engine: Engine,
        *,
        asynchronous: bool,
    ) -> SimpleNamespace:
        observed.append(os.environ.get("LITELLM_RUST"))
        return SimpleNamespace(
            events=(FunctionTraceEvent(0, None, "aocr" if asynchronous else "ocr"),),
            error=None,
        )

    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(execution, "_collect", collect)

    collect_trace(route, "python", asynchronous=False)
    collect_trace(route, "python", asynchronous=True, python_rust_enabled=True)

    assert observed == ["0", "1"]
    assert os.environ["LITELLM_RUST"] == "1"


def test_expected_provider_failure_omits_feedback_banner(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: Final = importlib.import_module("tests.rust-python-harness.strategies.trace_parity.sdk.responses.case")
    suite: Final = cast(TraceSuite, loaded.TRACE_SUITE)
    scenario: Final = next(item for item in suite.scenarios if item.name == "async-openai-provider-error")
    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    assert isinstance(suite.route, RouteSpec)

    result: Final = execute_trace(suite.route, scenario, "sdk", engine="python")

    assert result.python_error is None
    assert "Give Feedback / Get Help" not in capsys.readouterr().out
    assert litellm.suppress_debug_info is False


def test_scenario_validation_rejects_duplicate_and_unsafe_names() -> None:
    route: Final = RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture)
    duplicate: Final = TraceSuite(
        route=route,
        scenarios=(
            TraceScenario("sync-same", _fixture, (), asynchronous=False),
            TraceScenario("sync-same", _fixture, (), asynchronous=False),
        ),
    )
    unsafe: Final = TraceSuite(
        route=route, scenarios=(TraceScenario("sync-bad:name", _fixture, (), asynchronous=False),)
    )
    case: Final = _case()

    assert validate_trace_suite(duplicate, case) is not None
    assert validate_trace_suite(unsafe, case) is not None


def test_scenario_validation_rejects_invalid_names_and_route_registration() -> None:
    invalid_name: Final = TraceSuite(
        route=RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture),
        scenarios=(TraceScenario("bedrock", _fixture, (), asynchronous=True),),
    )
    wrong_function: Final = TraceSuite(
        route=RouteSpec("messages", ("create", "acreate"), ("messages", "amessages"), _fixture),
        scenarios=(TraceScenario("sync-one", _fixture, (), asynchronous=False),),
    )
    wrong_surface: Final = TraceSuite(
        route=GatewayRouteSpec("ocr"),
        scenarios=(TraceScenario("sync-one", _fixture, (), asynchronous=False),),
    )
    case: Final = _case()

    assert "start with sync- or async-" in (validate_trace_suite(invalid_name, case) or "")
    assert "does not match case function" in (validate_trace_suite(wrong_function, case) or "")
    assert "must use RouteSpec" in (validate_trace_suite(wrong_surface, case) or "")


def test_invalid_route_dispatch_records_harness_error() -> None:
    case: Final = _case()
    run: Final = HarnessRun.from_cases((case,))
    result: Final = run.results[case.key]
    suite: Final = TraceSuite(
        route=GatewayRouteSpec("ocr"),
        scenarios=(TraceScenario("sync-one", _fixture, (), asynchronous=False),),
    )
    nodeid: Final = "trace:sdk:ocr:sync-one"

    run_trace_scenario(run, result, suite, suite.scenarios[0], "sdk", nodeid, lambda _: None)

    assert result.outcomes[nodeid] is RunStatus.ERROR
    assert run.failures == [(nodeid, "gateway route cannot run on the sdk surface")]


def test_different_python_and_rust_traces_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    runner: Final = importlib.import_module("tests.rust-python-harness.strategies.trace_parity.runner")
    case: Final = _case()
    run: Final = HarnessRun.from_cases((case,))
    result: Final = run.results[case.key]
    suite: Final = TraceSuite(
        route=RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture),
        scenarios=(TraceScenario("sync-one", _fixture, (), asynchronous=False),),
    )
    trace: Final = TraceArtifact.from_traces(
        surface="sdk",
        sdk_function="ocr",
        scenario="sync-one",
        python=(PipelineStep(0, None, "python_step", "python.py:1 python_step"),),
        rust=(PipelineStep(0, None, "rust_step", "rust_step"),),
    )
    monkeypatch.setattr(runner, "_execute_scenario", lambda *_args: trace)

    run_trace_scenario(run, result, suite, suite.scenarios[0], "sdk", "trace:sdk:ocr:sync-one", lambda _: None)

    assert result.outcomes["trace:sdk:ocr:sync-one"] is RunStatus.PASSED
    assert run.failures == []
