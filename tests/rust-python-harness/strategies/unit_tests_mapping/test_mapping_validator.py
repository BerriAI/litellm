from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from ...shared.unit_runners.rust_runner import RustTarget, RustTestIdentity, RustTestScope
from .contracts import (
    MappingExclusionSpec,
    MappingSpec,
    RustTestFamily,
    RustUnitSpec,
    UnitParityExclusionSpec,
    UnitParitySpec,
    UnitTestContract,
)
from .contracts import TestMapping as MappingPair
from .mapping_validator import audit_mapping

_TARGET: Final = RustTarget(package="example", name="example", kind="lib")
_SCOPE: Final = RustTestScope(target=_TARGET, modules=("api::tests",))
_PYTHON_TESTS: Final = frozenset(("test_api.py::test_decode", "test_api.py::test_unmapped"))
_RUST_TEST: Final = RustTestIdentity(target=_TARGET, name="api::tests::decodes")
_RUST_ONLY: Final = RustTestIdentity(target=_TARGET, name="api::tests::rust_only")
_RUST_TESTS: Final = frozenset((_RUST_TEST, _RUST_ONLY))


def _python_inventory(*_: object) -> frozenset[str]:
    return _PYTHON_TESTS


def _rust_inventory(*_: object) -> frozenset[RustTestIdentity]:
    return _RUST_TESTS


def _contract(*mappings: MappingPair, exclusions: tuple[UnitParityExclusionSpec, ...] = ()) -> UnitTestContract:
    return UnitTestContract(
        mapping=MappingSpec(
            python_selectors=("test_api.py",),
            rust_scope=(_SCOPE,),
            mappings=mappings,
        ),
        unit_parity=UnitParitySpec(python_selectors=("test_api.py",), exclusions=exclusions),
        rust=RustUnitSpec(cargo_manifest="Cargo.toml", cargo_filter="api"),
    )


def _mapping_exclusion(nodeid: str) -> MappingExclusionSpec:
    return MappingExclusionSpec(nodeid=nodeid, reason="Python bridge availability is host-only")


def test_derives_mapping_status_from_live_inventories(tmp_path: Path) -> None:
    contract: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST))

    report: Final = audit_mapping(
        contract, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory
    )

    assert report.is_valid
    assert report.mapped_python_tests == ("test_api.py::test_decode",)
    assert report.unmapped_python_tests == ("test_api.py::test_unmapped",)
    assert report.rust_only_tests == (_RUST_ONLY.key,)
    assert report.percentage == 50.0


def test_reports_stale_and_duplicate_mappings(tmp_path: Path) -> None:
    removed: Final = RustTestIdentity(target=_TARGET, name="api::tests::removed")
    contract: Final = _contract(
        MappingPair(python="test_api.py::removed", rust=removed),
        MappingPair(python="test_api.py::removed", rust=_RUST_TEST),
    )

    report: Final = audit_mapping(
        contract, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory
    )

    assert not report.is_valid
    assert report.missing_python_tests == ("test_api.py::removed",)
    assert report.missing_rust_tests == (removed.key,)
    assert report.duplicate_python_mappings == ("test_api.py::removed",)


def test_reports_duplicate_rust_mapping_and_invalid_exclusion(tmp_path: Path) -> None:
    contract: Final = _contract(
        MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST),
        MappingPair(python="test_api.py::test_unmapped", rust=_RUST_TEST),
        exclusions=(UnitParityExclusionSpec(nodeid="test_api.py::removed", reason="Removed test"),),
    )

    report: Final = audit_mapping(
        contract, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory
    )

    assert not report.is_valid
    assert report.duplicate_rust_mappings == (_RUST_TEST.key,)
    assert report.invalid_unit_parity_exclusions == ("test_api.py::removed",)


def test_excludes_host_only_python_test_from_unmapped_inventory(tmp_path: Path) -> None:
    partial: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST))
    contract: Final = partial.model_copy(
        update={
            "mapping": partial.mapping.model_copy(
                update={"exclusions": (_mapping_exclusion("test_api.py::test_unmapped"),)}
            )
        }
    )

    report: Final = audit_mapping(
        contract, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory
    )

    assert report.is_valid
    assert report.excluded_python_tests == ("test_api.py::test_unmapped",)
    assert report.unmapped_python_tests == ()


def test_reports_missing_and_mapped_mapping_exclusions(tmp_path: Path) -> None:
    partial: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST))
    contract: Final = partial.model_copy(
        update={
            "mapping": partial.mapping.model_copy(
                update={
                    "exclusions": (
                        _mapping_exclusion("test_api.py::test_decode"),
                        _mapping_exclusion("test_api.py::removed"),
                    )
                }
            )
        }
    )

    report: Final = audit_mapping(
        contract, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory
    )

    assert not report.is_valid
    assert report.invalid_mapping_exclusions == ("test_api.py::removed",)
    assert report.mapped_and_excluded_python_tests == ("test_api.py::test_decode",)


def test_resolves_rstest_family_to_generated_cases(tmp_path: Path) -> None:
    first_case: Final = RustTestIdentity(target=_TARGET, name="api::tests::decodes::case_1_png")
    second_case: Final = RustTestIdentity(target=_TARGET, name="api::tests::decodes::case_2_pdf")
    family: Final = RustTestFamily(target=_TARGET, name="api::tests::decodes")
    contract: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=family))

    report: Final = audit_mapping(
        contract,
        tmp_path,
        python_inventory=_python_inventory,
        rust_inventory=lambda *_: frozenset((first_case, second_case)),
    )

    assert report.is_valid
    assert report.mapped_python_tests == ("test_api.py::test_decode",)
    assert report.missing_rust_tests == ()


def test_reports_missing_rstest_family(tmp_path: Path) -> None:
    family: Final = RustTestFamily(target=_TARGET, name="api::tests::decodes")
    contract: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=family))

    report: Final = audit_mapping(
        contract, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory
    )

    assert not report.is_valid
    assert report.missing_rust_tests == (family.key,)


def test_reports_concrete_test_owned_by_exact_and_family_mappings(tmp_path: Path) -> None:
    generated: Final = RustTestIdentity(target=_TARGET, name="api::tests::decodes::case_1_png")
    family: Final = RustTestFamily(target=_TARGET, name="api::tests::decodes")
    contract: Final = _contract(
        MappingPair(python="test_api.py::test_decode", rust=family),
        MappingPair(python="test_api.py::test_unmapped", rust=generated),
    )

    report: Final = audit_mapping(
        contract,
        tmp_path,
        python_inventory=_python_inventory,
        rust_inventory=lambda *_: frozenset((generated,)),
    )

    assert not report.is_valid
    assert report.duplicate_rust_mappings == (generated.key,)


def test_rstest_family_cases_are_not_rust_only(tmp_path: Path) -> None:
    generated: Final = RustTestIdentity(target=_TARGET, name="api::tests::decodes::case_1_png")
    unrelated: Final = RustTestIdentity(target=_TARGET, name="api::tests::rust_only")
    family: Final = RustTestFamily(target=_TARGET, name="api::tests::decodes")
    contract: Final = _contract(MappingPair(python="test_api.py::test_decode", rust=family))

    report: Final = audit_mapping(
        contract,
        tmp_path,
        python_inventory=_python_inventory,
        rust_inventory=lambda *_: frozenset((generated, unrelated)),
    )

    assert report.rust_only_tests == (unrelated.key,)


def test_merges_configured_and_colocated_rust_scopes(tmp_path: Path) -> None:
    support_test: Final = RustTestIdentity(target=_TARGET, name="support::tests::rust_only")
    configured_scope: Final = RustTestScope(
        target=_TARGET,
        modules=("support::tests",),
        features=("mock",),
        default_features=False,
    )
    expected_scope: Final = RustTestScope(
        target=_TARGET,
        modules=("api::tests", "support::tests"),
        features=("mock",),
        default_features=False,
    )
    contract: Final = UnitTestContract(
        mapping=MappingSpec(
            python_selectors=("test_api.py",),
            rust_scope=(configured_scope,),
            mappings=(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST),),
        ),
        unit_parity=UnitParitySpec(python_selectors=("test_api.py",)),
        rust=RustUnitSpec(cargo_manifest="Cargo.toml", cargo_filter="api"),
    )

    def assert_merged_scope(_: Path, scopes: tuple[RustTestScope, ...]) -> frozenset[RustTestIdentity]:
        assert scopes == (expected_scope,)
        return frozenset((_RUST_TEST, support_test))

    report: Final = audit_mapping(
        contract,
        tmp_path,
        python_inventory=_python_inventory,
        rust_inventory=assert_merged_scope,
    )

    assert report.is_valid
    assert report.rust_only_tests == (support_test.key,)


def test_merged_rust_scope_removes_modules_contained_by_parent(tmp_path: Path) -> None:
    expected_scope: Final = RustTestScope(target=_TARGET, modules=("api",))
    contract: Final = UnitTestContract(
        mapping=MappingSpec(
            python_selectors=("test_api.py",),
            rust_scope=(expected_scope,),
            mappings=(MappingPair(python="test_api.py::test_decode", rust=_RUST_TEST),),
        ),
        unit_parity=UnitParitySpec(python_selectors=("test_api.py",)),
        rust=RustUnitSpec(cargo_manifest="Cargo.toml", cargo_filter="api"),
    )

    def assert_parent_scope(_: Path, scopes: tuple[RustTestScope, ...]) -> frozenset[RustTestIdentity]:
        assert scopes == (expected_scope,)
        return frozenset((_RUST_TEST,))

    report: Final = audit_mapping(
        contract,
        tmp_path,
        python_inventory=_python_inventory,
        rust_inventory=assert_parent_scope,
    )

    assert report.is_valid


def test_accepts_descendant_unit_parity_selector() -> None:
    contract: Final = UnitTestContract(
        mapping=MappingSpec(python_selectors=("tests/api",), rust_scope=(_SCOPE,), mappings=()),
        unit_parity=UnitParitySpec(python_selectors=("tests/api/test_ocr.py",)),
        rust=RustUnitSpec(cargo_manifest="Cargo.toml", cargo_filter="api"),
    )

    assert contract.unit_parity.python_selectors == ("tests/api/test_ocr.py",)


@pytest.mark.parametrize(
    "mapping_selectors,parity_selectors",
    (((), ("tests/api",)), (("tests/api", "tests/api"), ("tests/api",)), (("tests/api",), ("tests/chat",))),
)
def test_rejects_invalid_selector_contracts(
    mapping_selectors: tuple[str, ...], parity_selectors: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError):
        UnitTestContract(
            mapping=MappingSpec(python_selectors=mapping_selectors, rust_scope=(_SCOPE,), mappings=()),
            unit_parity=UnitParitySpec(python_selectors=parity_selectors),
            rust=RustUnitSpec(cargo_manifest="Cargo.toml", cargo_filter="api"),
        )


def test_rejects_duplicate_scopes_and_exclusions() -> None:
    exclusion: Final = UnitParityExclusionSpec(nodeid="test_api.py::test_skip", reason="Backend assertion")
    with pytest.raises(ValidationError, match="duplicate targets"):
        MappingSpec(python_selectors=("test_api.py",), rust_scope=(_SCOPE, _SCOPE), mappings=())
    with pytest.raises(ValidationError, match="duplicate nodeids"):
        UnitParitySpec(python_selectors=("test_api.py",), exclusions=(exclusion, exclusion))
    mapping_exclusion: Final = _mapping_exclusion("test_api.py::test_skip")
    with pytest.raises(ValidationError, match="mapping exclusions contain duplicate nodeids"):
        MappingSpec(
            python_selectors=("test_api.py",),
            rust_scope=(_SCOPE,),
            mappings=(),
            exclusions=(mapping_exclusion, mapping_exclusion),
        )
    with pytest.raises(ValidationError, match="must be a non-empty string"):
        MappingExclusionSpec(nodeid="test_api.py::test_skip", reason=" ")
