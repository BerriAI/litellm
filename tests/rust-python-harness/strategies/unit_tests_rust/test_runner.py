from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Final

import pytest

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from . import STRATEGY


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is required for the native unit strategy")
def test_runs_cargo_tests_and_propagates_ignored_or_failing_tests(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "rust-unit-check"\nversion = "0.1.0"\nedition = "2021"\n[workspace]\n'
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/lib.rs").write_text('#[test] fn test_decode() { assert_eq!("42".parse::<u8>().unwrap(), 42); }\n')
    suite: Final = {"cargo_manifest": "Cargo.toml", "cargo_package": "rust-unit-check", "cargo_filter": "test_decode"}
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    case: Final = HarnessCase(
        strategy_id="unit_tests_rust",
        strategy_label="Unit test Rust",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="suite.json"),
    )

    code, report = STRATEGY.run((case,), tmp_path, lambda _: None)

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED

    (tmp_path / "src/lib.rs").write_text("#[test] #[ignore] fn test_decode() {}\n")
    ignored_code, ignored_report = STRATEGY.run((case,), tmp_path, lambda _: None)

    assert ignored_code == 1
    assert any("native Rust tests did not all pass" in detail for _, detail in ignored_report.failures)

    (tmp_path / "src/lib.rs").write_text('#[test] fn test_decode() { assert_eq!(2 + 2, 5); }\n')
    failed_code, failed_report = STRATEGY.run((case,), tmp_path, lambda _: None)

    assert failed_code == 1
    assert failed_report.results[case.key].status is RunStatus.FAILED
    assert any("test_decode" in detail for _, detail in failed_report.failures)
