from __future__ import annotations

from typing import Final

from ...shared.reporting.models import CaseResult, Coverage, HarnessCase, ResultArtifact, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from .mapping_report import MappingReportArtifact
from .mapping_validator import MappingReport
from .reporting import render_mapping_results
from .runner import MAPPING_REPORT_ARTIFACT


def _report(*, invalid: bool = False, excluded: bool = False) -> MappingReport:
    return MappingReport(
        python_tests=("test_api.py::test_decode", "test_api.py::test_unmapped"),
        rust_tests=("example/lib/example::api::tests::decodes", "example/lib/example::api::tests::rust_only"),
        mapped_python_tests=("test_api.py::test_decode",),
        excluded_python_tests=(("test_api.py::test_unmapped",) if excluded else ()),
        unmapped_python_tests=(() if excluded else ("test_api.py::test_unmapped",)),
        rust_only_tests=("example/lib/example::api::tests::rust_only",),
        missing_python_tests=("test_api.py::removed",) if invalid else (),
        missing_rust_tests=(),
        duplicate_python_mappings=(),
        duplicate_rust_mappings=(),
        invalid_mapping_exclusions=(),
        mapped_and_excluded_python_tests=(),
        invalid_unit_parity_exclusions=(),
    )


def _result(body: str) -> CaseResult:
    case: Final = HarnessCase(
        strategy_id="unit_tests_mapping",
        strategy_label="Unit test mapping",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )
    result: Final = CaseResult(case=case)
    result.record(
        "suite:unit_tests_mapping:ocr:ocr",
        RunStatus.PASSED,
        artifacts=(ResultArtifact(MAPPING_REPORT_ARTIFACT, body),),
    )
    return result


def test_renderer_preserves_summary_and_detailed_output() -> None:
    summary: Final = MappingReportArtifact(report=_report()).model_dump_json()
    detailed: Final = MappingReportArtifact(report=_report(), detailed=True).model_dump_json()

    summary_text: Final = "\n".join(render_mapping_results((_result(summary),))[0].blocks)
    detailed_text: Final = "\n".join(render_mapping_results((_result(detailed),))[0].blocks)

    assert "Mapped       1 / 2 (50.0%)" in summary_text
    assert "Unmapped Python test details" not in summary_text
    assert "Unmapped Python test details\n  test_api.py\n    test_unmapped" in detailed_text
    assert "Rust-only test details" in detailed_text


def test_renderer_shows_contract_errors() -> None:
    body: Final = MappingReportArtifact(report=_report(invalid=True)).model_dump_json()
    rendered: Final = "\n".join(render_mapping_results((_result(body),))[0].blocks)

    assert "Contract: FAIL" in rendered
    assert "Missing Python test: test_api.py::removed" in rendered


def test_renderer_distinguishes_excluded_python_tests() -> None:
    body: Final = MappingReportArtifact(report=_report(excluded=True), detailed=True).model_dump_json()
    rendered: Final = "\n".join(render_mapping_results((_result(body),))[0].blocks)

    assert "Excluded     1 / 2 (50.0%)" in rendered
    assert "Unmapped     0 / 2 (0.0%)" in rendered
    assert "Excluded Python test details\n  test_api.py\n    test_unmapped" in rendered


def test_renderer_handles_empty_inventory_and_malformed_artifact() -> None:
    empty: Final = MappingReport(
        python_tests=(),
        rust_tests=(),
        mapped_python_tests=(),
        excluded_python_tests=(),
        unmapped_python_tests=(),
        rust_only_tests=(),
        missing_python_tests=(),
        missing_rust_tests=(),
        duplicate_python_mappings=(),
        duplicate_rust_mappings=(),
        invalid_mapping_exclusions=(),
        mapped_and_excluded_python_tests=(),
        invalid_unit_parity_exclusions=(),
    )
    empty_text: Final = "\n".join(
        render_mapping_results((_result(MappingReportArtifact(report=empty).model_dump_json()),))[0].blocks
    )
    invalid_text: Final = "\n".join(render_mapping_results((_result("not-json"),))[0].blocks)

    assert "Mapped       0 / 0 (0.0%)" in empty_text
    assert "Mapping report artifact is invalid:" in invalid_text
