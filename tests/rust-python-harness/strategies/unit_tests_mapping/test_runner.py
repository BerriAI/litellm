from __future__ import annotations

from pathlib import Path
from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from .mapping_validator import MappingSuite, TestMapping as MappingPair
from .runner import MAPPING_REPORT_ARTIFACT, run_mapping_cases


def _suite(mapping: MappingPair) -> MappingSuite:
    return MappingSuite(
        python_scope=("test_api.py",),
        unit_parity_scope=("test_api.py",),
        rust_scope=("lib.rs",),
        cargo_manifest="Cargo.toml",
        cargo_filter="api",
        mappings=(mapping,),
    )


def test_reports_derived_mapping_status_without_running_tests(tmp_path: Path) -> None:
    (tmp_path / "test_api.py").write_text(
        "def test_decode():\n    raise AssertionError\n\ndef test_unmapped():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "lib.rs").write_text(
        "#[test]\nfn decodes() { panic!(); }\n\n#[test]\nfn rust_only() {}\n",
        encoding="utf-8",
    )
    suite: Final = _suite(MappingPair(python="test_api.py::test_decode", rust="lib.rs::decodes"))
    case: Final = HarnessCase(
        strategy_id="unit_tests_mapping",
        strategy_label="Unit test mapping",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )

    code, report = run_mapping_cases((case,), tmp_path, lambda _: None, suites={"ocr": suite})

    assert code == 0, report.failures
    assert report.results[case.key].status is RunStatus.PASSED
    rendered: Final = "\n".join(
        artifact.body
        for artifacts in report.results[case.key].artifacts.values()
        for artifact in artifacts
        if artifact.kind == MAPPING_REPORT_ARTIFACT
    )
    assert "1/2 python tests mapped to rust (50.0%)" in rendered
    assert "1 rust-only tests with no python counterpart" in rendered
    assert "unmapped python test: test_api.py::test_unmapped" in rendered
    assert "rust-only test: lib.rs::rust_only" in rendered
    assert "mapping contract is valid" in rendered


def test_fails_when_a_mapping_target_is_missing(tmp_path: Path) -> None:
    (tmp_path / "test_api.py").write_text("def test_decode():\n    pass\n", encoding="utf-8")
    (tmp_path / "lib.rs").write_text("#[test]\nfn removed() {}\n", encoding="utf-8")
    suite: Final = _suite(MappingPair(python="test_api.py::test_decode", rust="lib.rs::decodes"))
    case: Final = HarnessCase(
        strategy_id="unit_tests_mapping",
        strategy_label="Unit test mapping",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )

    code, report = run_mapping_cases((case,), tmp_path, lambda _: None, suites={"ocr": suite})

    assert code == 1
    assert report.results[case.key].status is RunStatus.FAILED
    assert any("mapped Rust test does not exist" in detail for _, detail in report.failures)
