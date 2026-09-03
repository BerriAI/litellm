from __future__ import annotations

import importlib
from typing import Final

import pytest

models = importlib.import_module("tests.rust-python-harness.shared.reporting.models")
strategy_module = importlib.import_module("tests.rust-python-harness.shared.reporting.strategy")
ui = importlib.import_module("tests.rust-python-harness.shared.reporting.ui")
ledger_module = importlib.import_module("tests.rust-python-harness.shared.parity.ledger")
mapping_validator = importlib.import_module(
    "tests.rust-python-harness.strategies.unit_tests_mapping.mapping_validator"
)

load_ledger = ledger_module.load_ledger
ledger_path_for = mapping_validator.ledger_path_for
REPO_ROOT = mapping_validator.REPO_ROOT
audit_ledger = mapping_validator.audit_ledger
build_function_report = mapping_validator.build_function_report
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
        "tests.rust-python-harness.strategies.trace_parity.sdk.ocr.test_trace_parity",
        "tests.rust-python-harness.strategies.trace_parity.sdk.messages.test_trace_parity",
        "tests.rust-python-harness.strategies.trace_parity.sdk.chat_completions.test_trace_parity",
        "tests.rust-python-harness.strategies.trace_parity.sdk.transcription.test_trace_parity",
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


def test_should_report_no_ledger_for_a_function_without_one() -> None:
    report = build_function_report("messages", repo_root=REPO_ROOT)

    assert report.has_ledger is False
    assert report.is_clean is True


def test_should_report_ocr_ledger_stats_and_a_clean_audit() -> None:
    ledger = load_ledger(ledger_path_for("ocr"))

    report = build_function_report("ocr", repo_root=REPO_ROOT)

    assert report.has_ledger is True
    assert report.ledger.mapped_count == ledger.mapped_count
    assert report.ledger.total_count == ledger.total_count
    assert report.is_clean is True


def test_should_have_every_python_and_rust_ocr_test_accounted_for_in_the_ledger() -> None:
    ledger = load_ledger(ledger_path_for("ocr"))

    report = audit_ledger(ledger, repo_root=REPO_ROOT)

    assert report.is_clean, (
        "\nOCR test-parity ledger is out of sync with the live test files.\n"
        f"Ledger references a Python test that no longer exists: {list(report.missing_python_tests)}\n"
        f"Python test exists but is not tracked in the ledger: {list(report.stale_python_tests)}\n"
        f"Ledger references a Rust test that no longer exists: {list(report.missing_rust_tests)}\n"
        f"Rust test exists but is not tracked in the ledger: {list(report.stale_rust_tests)}\n"
    )
