from __future__ import annotations

from pathlib import Path
from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from ...shared.unit_runners.rust_runner import RustTarget, RustTestIdentity, RustTestScope
from .mapping_validator import MappingSuite, TestMapping as MappingPair
from .runner import MAPPING_REPORT_ARTIFACT, run_mapping_cases

_TARGET: Final = RustTarget(package="example", name="example", kind="lib")
_RUST_TEST: Final = RustTestIdentity(target=_TARGET, name="api::tests::decodes")
_RUST_ONLY: Final = RustTestIdentity(target=_TARGET, name="api::tests::rust_only")


def _python_inventory(*_: object) -> frozenset[str]:
    return frozenset(("test_api.py::test_decode", "test_api.py::test_unmapped"))


def _rust_inventory(*_: object) -> frozenset[RustTestIdentity]:
    return frozenset((_RUST_TEST, _RUST_ONLY))


def _suite(mapping: MappingPair) -> MappingSuite:
    return MappingSuite(
        python_selectors=("test_api.py",),
        unit_parity_selectors=("test_api.py",),
        rust_scope=(RustTestScope(target=_TARGET, modules=("api::tests",)),),
        cargo_manifest="Cargo.toml",
        cargo_filter="api",
        mappings=(mapping,),
    )


def test_reports_derived_mapping_status_without_running_tests(tmp_path: Path) -> None:
    suite: Final = _suite(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST.key))
    case: Final = HarnessCase(
        strategy_id="unit_tests_mapping",
        strategy_label="Unit test mapping",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )

    code, report = run_mapping_cases(
        (case,),
        tmp_path,
        lambda _: None,
        suites={"ocr": suite},
        python_inventory=_python_inventory,
        rust_inventory=_rust_inventory,
    )

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
    assert f"rust-only test: {_RUST_ONLY.key}" in rendered
    assert "mapping contract is valid" in rendered


def test_fails_when_a_mapping_target_is_missing(tmp_path: Path) -> None:
    missing: Final = f"{_TARGET.key}::api::tests::missing"
    suite: Final = _suite(MappingPair(python="test_api.py::test_decode", rust=missing))
    case: Final = HarnessCase(
        strategy_id="unit_tests_mapping",
        strategy_label="Unit test mapping",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )

    code, report = run_mapping_cases(
        (case,),
        tmp_path,
        lambda _: None,
        suites={"ocr": suite},
        python_inventory=_python_inventory,
        rust_inventory=_rust_inventory,
    )

    assert code == 1
    assert report.results[case.key].status is RunStatus.FAILED
    assert any("mapped Rust test does not exist" in detail for _, detail in report.failures)
