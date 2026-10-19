from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from ...shared.unit_runners.suite_runner import run_suites
from .runner import RustSuite, run_suite


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is required for the native unit strategy")
def test_runs_cargo_tests_and_propagates_ignored_or_failing_tests(
    tmp_path: Path,
    cargo_project: Callable[[str, str], Path],
) -> None:
    cargo_project("rust-unit-check", '#[test] fn test_decode() { assert_eq!("42".parse::<u8>().unwrap(), 42); }\n')
    rust_root: Final = tmp_path / "litellm-rust"
    rust_root.mkdir()
    (tmp_path / "Cargo.toml").rename(rust_root / "Cargo.toml")
    (tmp_path / "src").rename(rust_root / "src")
    suite: Final = RustSuite(
        cargo_manifest="litellm-rust/Cargo.toml",
        cargo_filter="test_decode",
    )
    case: Final = HarnessCase(
        strategy_id="unit_tests_rust",
        strategy_label="Unit test Rust",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )

    code, report = run_suites((case,), tmp_path, lambda _: None, suites={"ocr": suite}, execute=run_suite)

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED

    (rust_root / "src/lib.rs").write_text("#[test] #[ignore] fn test_decode() {}\n")
    ignored_code, ignored_report = run_suites(
        (case,), tmp_path, lambda _: None, suites={"ocr": suite}, execute=run_suite
    )

    assert ignored_code == 1
    assert any("native Rust tests did not all pass" in detail for _, detail in ignored_report.failures)

    (rust_root / "src/lib.rs").write_text("#[test] fn test_decode() { assert_eq!(2 + 2, 5); }\n")
    failed_code, failed_report = run_suites((case,), tmp_path, lambda _: None, suites={"ocr": suite}, execute=run_suite)

    assert failed_code == 1
    assert failed_report.results[case.key].status is RunStatus.FAILED
    assert any("test_decode" in detail for _, detail in failed_report.failures)
