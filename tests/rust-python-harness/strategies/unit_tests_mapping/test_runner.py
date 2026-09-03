from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Final

import pytest

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from .runner import run


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is required for the mapping unit strategy")
def test_validates_mappings_without_running_the_selected_tests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[4]))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "backend_probe.py").write_text(
        "import os\ndef selected():\n    return 'rust' if os.environ['TEST_USE_RUST'] == '1' else 'python'\n"
    )
    (tmp_path / "test_api.py").write_text("def test_decode():\n    assert int('42') == 42\n")
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "mapping-check"\nversion = "0.1.0"\nedition = "2021"\n[workspace]\n'
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/lib.rs").write_text('#[test] fn test_decode() { assert_eq!("42".parse::<u8>().unwrap(), 42); }\n')
    suite: Final = {
        "python_selectors": ("test_api.py",),
        "cargo_manifest": "Cargo.toml",
        "cargo_package": "mapping-check",
        "cargo_filter": "test_decode",
        "backend": {"environment_variable": "TEST_USE_RUST", "probe": "backend_probe:selected"},
    }
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    case: Final = HarnessCase(
        strategy_id="unit_tests_mapping",
        strategy_label="Unit test mapping",
        sdk_function="ocr",
        coverage=Coverage.COMPLETE,
        selectors=(),
        unit_suite="suite.json",
    )

    (tmp_path / "test_api.py").write_text("def test_decode():\n    assert False\n")
    passing_code, passing_report = run((case,), tmp_path, lambda _: None)

    assert passing_code == 0, passing_report.failures
    assert passing_report.results[case.key].status is RunStatus.PASSED

    (tmp_path / "suite.json").write_text(
        json.dumps({**suite, "mappings": [{"python": "test_api.py::test_decode", "rust": "removed"}]})
    )
    failed_code, failed_report = run((case,), tmp_path, lambda _: None)

    assert failed_code == 1
    assert failed_report.results[case.key].status is RunStatus.FAILED
    assert any("missing Rust counterpart: removed" in detail for _, detail in failed_report.failures)
