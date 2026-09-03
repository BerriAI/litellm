from __future__ import annotations

from pathlib import Path
from typing import Final

from .models import Coverage, HarnessCase, RunStatus
from .orchestration import run_strategies
from .pytest_runner import run_pytest


def test_combines_independent_strategy_reports_and_keeps_failures(tmp_path: Path) -> None:
    (tmp_path / "test_first.py").write_text("def test_first():\n    assert 1 == 2\n")
    (tmp_path / "test_second.py").write_text("def test_second():\n    assert True\n")
    cases: Final = tuple(
        HarnessCase(
            strategy_id=name,
            strategy_label=name,
            sdk_function="ocr",
            coverage=Coverage.COMPLETE,
            selectors=(f"test_{name}.py",),
        )
        for name in ("first", "second")
    )
    code, report = run_strategies(cases, tmp_path, lambda _: None, (), lambda _: run_pytest)
    assert code == 1
    assert report.results["first:ocr"].status is RunStatus.FAILED
    assert report.results["second:ocr"].status is RunStatus.PASSED
    assert report.completed_tests == 2
    assert len(report.failures) == 1
    assert "assert 1 == 2" in report.failures[0][1]
    assert "terminalreporter" not in report.failures[0][1]


def test_missing_selector_cannot_hide_behind_a_passing_surface(tmp_path: Path) -> None:
    (tmp_path / "test_present.py").write_text("def test_present():\n    assert True\n")
    case: Final = HarnessCase(
        strategy_id="e2e_parity",
        strategy_label="End-to-end parity",
        sdk_function="ocr",
        surface="gateway",
        coverage=Coverage.PARTIAL,
        selectors=("test_present.py", "test_missing.py"),
    )
    code, report = run_pytest((case,), tmp_path, lambda _: None)
    assert code == 1
    assert report.results["e2e_parity:gateway:ocr"].status is RunStatus.MISSING
    assert ("test_missing.py", "Configured selector collected no tests") in report.failures
