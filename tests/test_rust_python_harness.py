from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import pytest

models = importlib.import_module("tests.rust-python-harness.shared.reporting.models")
strategy_module = importlib.import_module("tests.rust-python-harness.shared.reporting.strategy")
ui = importlib.import_module("tests.rust-python-harness.shared.reporting.ui")
mapping_validator = importlib.import_module("tests.rust-python-harness.strategies.unit_tests_mapping.mapping_validator")
mappings = importlib.import_module("tests.rust-python-harness.strategies.unit_tests_mapping.mappings")
ocr_mapping = importlib.import_module("tests.rust-python-harness.strategies.unit_tests_mapping.cases.ocr")
cli = importlib.import_module("tests.rust-python-harness.cli")
native_build = importlib.import_module("tests.rust-python-harness.shared.native_build")

audit_mapping = mapping_validator.audit_mapping
UNIT_TEST_CONTRACTS = mappings.UNIT_TEST_CONTRACTS
OCR_CONTRACT = ocr_mapping.OCR_CONTRACT
REPO_ROOT = Path(__file__).resolve().parents[1]
CaseResult = models.CaseResult
Coverage = models.Coverage
HarnessCase = models.HarnessCase
HarnessRun = models.HarnessRun
RunStatus = models.RunStatus
ModuleCaseSpec = strategy_module.ModuleCaseSpec
NotImplementedCaseSpec = strategy_module.NotImplementedCaseSpec
SkippedCaseSpec = strategy_module.SkippedCaseSpec
_format_duration = ui._format_duration
_summary = ui._summary


def _case(module: str = "tests.example") -> HarnessCase:
    return HarnessCase(
        strategy_id="example",
        strategy_label="Example",
        sdk_function="messages",
        spec=ModuleCaseSpec(coverage=Coverage.COMPLETE, module=module),
    )


@pytest.mark.parametrize(
    "module",
    [
        "tests.rust-python-harness.strategies.e2e_parity.sdk.ocr.test_sdk_parity",
        "tests.rust-python-harness.strategies.trace_parity.sdk.ocr.case",
        "tests.rust-python-harness.strategies.trace_parity.sdk.messages.case",
        "tests.rust-python-harness.strategies.trace_parity.sdk.chat_completions.case",
        "tests.rust-python-harness.strategies.trace_parity.sdk.transcription.case",
        "tests.rust-python-harness.strategies.trace_parity.gateway.messages.case",
    ],
)
def test_implemented_namespace_case_modules_remain_importable(module: str) -> None:
    assert importlib.import_module(module)


def test_should_mark_not_implemented_and_skipped_cases_without_running() -> None:
    not_implemented: Final = CaseResult(
        case=HarnessCase(
            strategy_id="example",
            strategy_label="Example",
            sdk_function="messages",
            spec=NotImplementedCaseSpec(reason="No case is registered."),
        )
    )
    skipped: Final = CaseResult(
        case=HarnessCase(
            strategy_id="example",
            strategy_label="Example",
            sdk_function="messages",
            spec=SkippedCaseSpec(reason="The surface does not apply."),
        )
    )

    not_implemented.set_initial_status()
    skipped.set_initial_status()

    assert not_implemented.status is RunStatus.NOT_IMPLEMENTED
    assert skipped.status is RunStatus.SKIPPED


def test_should_finalize_a_fully_passing_case() -> None:
    result = CaseResult(case=_case())
    result.set_initial_status()
    result.collected.update({"one", "two"})
    result.completed.update({"one", "two"})
    result.passed = 2

    result.finalize()

    assert result.status is RunStatus.PASSED


def test_should_replace_a_pass_with_a_teardown_error() -> None:
    result = CaseResult(case=_case())
    result.set_initial_status()
    result.collected.add("one")

    result.record("one", RunStatus.PASSED, 0.1)
    result.record("one", RunStatus.ERROR, 0.2)

    assert result.status is RunStatus.ERROR
    assert result.passed == 0
    assert result.errors == 1
    assert result.duration == pytest.approx(0.3)


def test_should_format_developer_facing_run_context() -> None:
    run = HarnessRun.from_cases((_case(),))
    result = next(iter(run.results.values()))
    result.collected.add("tests/test_parity.py::test_one")
    result.record("tests/test_parity.py::test_one", RunStatus.PASSED, 1.25)

    assert _summary(run) == (1, 0, 0, 0)
    assert _format_duration(1.25) == "1.2s"


def test_should_leave_functions_without_mapping_contracts_unimplemented() -> None:
    assert "messages" not in UNIT_TEST_CONTRACTS


def test_should_report_a_bridge_that_cannot_be_imported() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native_build, "get_native_bridge", lambda: None)
        message: Final = native_build.trace_bridge_error()

    assert message is not None
    assert "not importable" in message


def test_should_report_a_bridge_built_without_the_trace_feature() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native_build, "get_native_bridge", lambda: SimpleNamespace(_trace=None))
        message: Final = native_build.trace_bridge_error()

    assert message is not None
    assert native_build.BRIDGE_FEATURE in message


def test_should_accept_a_bridge_built_with_the_trace_feature() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native_build, "get_native_bridge", lambda: SimpleNamespace(_trace=object()))

        assert native_build.trace_bridge_error() is None


def test_should_not_rebuild_the_bridge_while_reporting_its_state() -> None:
    def forbidden_rebuild(repo_root: object) -> tuple[bool, str]:
        raise AssertionError("trace_bridge_error must not rebuild the native bridge")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native_build, "_rebuild", forbidden_rebuild)
        patch.setattr(native_build, "get_native_bridge", lambda: None)

        assert native_build.trace_bridge_error() is not None


def test_should_derive_ocr_mapping_status_from_live_tests() -> None:
    bridge_error: Final = native_build.trace_bridge_error()
    if bridge_error is not None:
        pytest.skip(bridge_error)

    report = audit_mapping(OCR_CONTRACT, repo_root=REPO_ROOT)

    assert report.is_valid, (
        f"Missing Python tests: {list(report.missing_python_tests)}\n"
        f"Missing Rust tests: {list(report.missing_rust_tests)}\n"
        f"Duplicate Python mappings: {list(report.duplicate_python_mappings)}\n"
        f"Invalid mapping exclusions: {list(report.invalid_mapping_exclusions)}\n"
        f"Invalid parity exclusions: {list(report.invalid_unit_parity_exclusions)}"
    )
    assert report.mapped_count == len(OCR_CONTRACT.mapping.mappings)
    assert report.total_count == (
        report.mapped_count + len(report.excluded_python_tests) + len(report.unmapped_python_tests)
    )


def test_strategy_subcommand_accepts_function_filter(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code: Final = cli.main(["run", "unit_tests_mapping", "--function", "messages"])

    captured: Final = capsys.readouterr()
    assert exit_code == 0
    assert "- messages: not_implemented" in captured.out
    assert "unit_tests_mapping:messages: not_implemented" not in captured.out
