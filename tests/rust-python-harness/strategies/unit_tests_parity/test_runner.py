from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from . import STRATEGY

HARNESS_ROOT: Final = Path(__file__).resolve().parents[4]


def _write_tests(tmp_path: Path, *, mismatch: bool = False, failing: bool = False) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_api.py").write_text(
        "import os\n"
        "def test_decode():\n    assert int('42') == 42\n"
        + ("def test_backend():\n    assert os.environ['LITELLM_RUST'] == '0'\n" if mismatch else "")
        + ("def test_fails():\n    assert False\n" if failing else "")
    )
    (tmp_path / "suite.json").write_text(json.dumps({"python_selectors": ("test_api.py",)}))


def _case() -> HarnessCase:
    return HarnessCase(
        strategy_id="unit_tests_parity",
        strategy_label="Unit test parity",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="suite.json"),
    )


def test_passes_when_both_backends_agree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(HARNESS_ROOT))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    _write_tests(tmp_path)
    case: Final = _case()

    code, report = STRATEGY.run((case,), tmp_path, lambda _: None)

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED


def test_passes_when_both_backends_fail_identically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(HARNESS_ROOT))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    _write_tests(tmp_path, failing=True)
    case: Final = _case()

    code, report = STRATEGY.run((case,), tmp_path, lambda _: None)

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED


def test_fails_when_backend_outcomes_differ(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(HARNESS_ROOT))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    _write_tests(tmp_path, mismatch=True)
    case: Final = _case()

    code, report = STRATEGY.run((case,), tmp_path, lambda _: None)

    assert code == 1
    assert report.results[case.key].status is RunStatus.FAILED
    assert any("Python/Rust test outcomes differ" in detail for _, detail in report.failures)
