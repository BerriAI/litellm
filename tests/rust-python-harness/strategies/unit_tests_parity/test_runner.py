from __future__ import annotations

from pathlib import Path
from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase, HarnessRun, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from ...shared.unit_runners.suite_runner import run_suites
from .runner import UnitParityExclusion, UnitParitySuite, run_suite


def _write_tests(tmp_path: Path, *, mismatch: bool = False, failing: bool = False) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_api.py").write_text(
        "import os\n"
        "def test_decode():\n    assert int('42') == 42\n"
        + ("def test_backend():\n    assert os.environ['LITELLM_RUST'] == '0'\n" if mismatch else "")
        + ("def test_fails():\n    assert False\n" if failing else "")
    )


def _case() -> HarnessCase:
    return HarnessCase(
        strategy_id="unit_tests_parity",
        strategy_label="Unit test parity",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )


def _run(case: HarnessCase, tmp_path: Path, suite: UnitParitySuite) -> tuple[int, HarnessRun]:
    return run_suites((case,), tmp_path, lambda _: None, suites={"ocr": suite}, execute=run_suite)


def test_passes_when_both_backends_agree(tmp_path: Path) -> None:
    _write_tests(tmp_path)
    case: Final = _case()

    code, report = _run(case, tmp_path, UnitParitySuite(python_selectors=("test_api.py",)))

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED


def test_passes_when_both_backends_fail_identically(tmp_path: Path) -> None:
    _write_tests(tmp_path, failing=True)
    case: Final = _case()

    code, report = _run(case, tmp_path, UnitParitySuite(python_selectors=("test_api.py",)))

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED


def test_fails_when_backend_outcomes_differ(tmp_path: Path) -> None:
    _write_tests(tmp_path, mismatch=True)
    case: Final = _case()

    code, report = _run(case, tmp_path, UnitParitySuite(python_selectors=("test_api.py",)))

    assert code == 1
    assert report.results[case.key].status is RunStatus.FAILED
    assert any("Python/Rust test outcomes differ" in detail for _, detail in report.failures)
    assert any("Python only: test_api.py::test_backend [call] passed" in detail for _, detail in report.failures)
    assert any("Rust only: test_api.py::test_backend [call] failed" in detail for _, detail in report.failures)


def test_excludes_tests_whose_contract_is_the_backend_flag(tmp_path: Path) -> None:
    _write_tests(tmp_path, mismatch=True)
    suite: Final = UnitParitySuite(
        python_selectors=("test_api.py",),
        exclusions=(
            UnitParityExclusion(
                nodeid="test_api.py::test_backend",
                reason="The test intentionally asserts which backend is selected.",
            ),
        ),
    )
    case: Final = _case()

    code, report = _run(case, tmp_path, suite)

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED
