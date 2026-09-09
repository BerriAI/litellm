from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from ...shared.reporting.strategy import SuiteCaseSpec
from ...shared.unit_runners.rust_runner import RustTarget, RustTestIdentity, RustTestScope
from ...shared.unit_runners.suite_runner import run_suites
from .contracts import (
    MappingExclusionSpec,
    MappingSpec,
    RustUnitSpec,
    TestMapping as MappingPair,
    UnitParitySpec,
    UnitTestContract,
)
from .mapping_report import MappingReportArtifact
from .runner import MAPPING_REPORT_ARTIFACT, run_suite

_TARGET: Final = RustTarget(package="example", name="example", kind="lib")
_RUST_TEST: Final = RustTestIdentity(target=_TARGET, name="api::tests::decodes")
_RUST_ONLY: Final = RustTestIdentity(target=_TARGET, name="api::tests::rust_only")


def _python_inventory(*_: object) -> frozenset[str]:
    return frozenset(("test_api.py::test_decode", "test_api.py::test_unmapped"))


def _rust_inventory(*_: object) -> frozenset[RustTestIdentity]:
    return frozenset((_RUST_TEST, _RUST_ONLY))


def _contract(mapping: MappingPair) -> UnitTestContract:
    return UnitTestContract(
        mapping=MappingSpec(
            python_selectors=("test_api.py",),
            rust_scope=(RustTestScope(target=_TARGET, modules=("api::tests",)),),
            mappings=(mapping,),
        ),
        unit_parity=UnitParitySpec(python_selectors=("test_api.py",)),
        rust=RustUnitSpec(cargo_manifest="Cargo.toml", cargo_filter="api"),
    )


def _case() -> HarnessCase:
    return HarnessCase(
        strategy_id="unit_tests_mapping",
        strategy_label="Unit test mapping",
        sdk_function="ocr",
        spec=SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"),
    )


def test_reports_structured_mapping_status_without_running_tests(tmp_path: Path) -> None:
    contract: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST))
    case: Final = _case()

    code, report = run_suites(
        (case,),
        tmp_path,
        lambda _: None,
        suites={"ocr": contract},
        execute=partial(
            run_suite,
            python_inventory=_python_inventory,
            rust_inventory=_rust_inventory,
        ),
    )

    result: Final = report.results[case.key]
    artifacts: Final = tuple(
        artifact
        for values in result.artifacts.values()
        for artifact in values
        if artifact.kind == MAPPING_REPORT_ARTIFACT
    )
    parsed: Final = MappingReportArtifact.model_validate_json(artifacts[0].body)
    assert code == 0, report.failures
    assert result.status is RunStatus.PASSED
    assert parsed.report.mapped_count == 1
    assert parsed.report.total_count == 2
    assert not parsed.detailed


def test_fails_when_a_mapping_target_is_missing(tmp_path: Path) -> None:
    missing: Final = RustTestIdentity(target=_TARGET, name="api::tests::missing")
    contract: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=missing))
    case: Final = _case()

    code, report = run_suites(
        (case,),
        tmp_path,
        lambda _: None,
        suites={"ocr": contract},
        execute=partial(
            run_suite,
            python_inventory=_python_inventory,
            rust_inventory=_rust_inventory,
        ),
    )

    assert code == 1
    assert report.results[case.key].status is RunStatus.FAILED
    assert any("mapped Rust test does not exist" in detail for _, detail in report.failures)


def test_required_complete_mapping_fails_for_unmapped_python_test(tmp_path: Path) -> None:
    partial: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST))
    contract: Final = partial.model_copy(
        update={"mapping": partial.mapping.model_copy(update={"require_complete": True})}
    )

    execution: Final = run_suite(
        contract,
        tmp_path,
        python_inventory=_python_inventory,
        rust_inventory=_rust_inventory,
    )

    assert execution.problems == ("Python test has no Rust mapping: test_api.py::test_unmapped",)


def test_required_complete_mapping_accepts_host_only_exclusion(tmp_path: Path) -> None:
    partial: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST))
    contract: Final = partial.model_copy(
        update={
            "mapping": partial.mapping.model_copy(
                update={
                    "require_complete": True,
                    "exclusions": (
                        MappingExclusionSpec(
                            nodeid="test_api.py::test_unmapped",
                            reason="Python bridge availability is host-only",
                        ),
                    ),
                }
            )
        }
    )

    execution: Final = run_suite(
        contract,
        tmp_path,
        python_inventory=_python_inventory,
        rust_inventory=_rust_inventory,
    )
    artifact: Final = MappingReportArtifact.model_validate_json(execution.artifacts[0].body)

    assert execution.problems == ()
    assert artifact.report.excluded_python_tests == ("test_api.py::test_unmapped",)


def test_detail_argument_is_stored_in_artifact(tmp_path: Path) -> None:
    contract: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST))
    execution: Final = run_suite(
        contract,
        tmp_path,
        ("full",),
        python_inventory=_python_inventory,
        rust_inventory=_rust_inventory,
    )
    artifact: Final = MappingReportArtifact.model_validate_json(execution.artifacts[0].body)

    assert artifact.detailed
