from __future__ import annotations

from pathlib import Path
from typing import Final

from ...shared.unit_runners.rust_runner import RustTarget, RustTestIdentity, RustTestScope
from .mapping_validator import MappingSuite, TestMapping as MappingPair, audit_mapping

_TARGET: Final = RustTarget(package="example", name="example", kind="lib")
_SCOPE: Final = RustTestScope(target=_TARGET, modules=("api::tests",))
_PYTHON_TESTS: Final = frozenset(("test_api.py::test_decode", "test_api.py::test_unmapped"))
_RUST_TESTS: Final = frozenset(
    (
        RustTestIdentity(target=_TARGET, name="api::tests::decodes"),
        RustTestIdentity(target=_TARGET, name="api::tests::rust_only"),
    )
)


def _python_inventory(*_: object) -> frozenset[str]:
    return _PYTHON_TESTS


def _rust_inventory(*_: object) -> frozenset[RustTestIdentity]:
    return _RUST_TESTS


def _suite(*mappings: MappingPair) -> MappingSuite:
    return MappingSuite(
        python_selectors=("test_api.py",),
        unit_parity_selectors=("test_api.py",),
        rust_scope=(_SCOPE,),
        cargo_manifest="Cargo.toml",
        cargo_filter="api",
        mappings=mappings,
    )


def test_derives_mapping_status_from_live_inventories(tmp_path: Path) -> None:
    suite = _suite(MappingPair(python="test_api.py::test_decode", rust=f"{_TARGET.key}::api::tests::decodes"))

    report = audit_mapping(suite, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory)

    assert report.is_valid
    assert report.mapped_python_tests == ("test_api.py::test_decode",)
    assert report.unmapped_python_tests == ("test_api.py::test_unmapped",)
    assert report.rust_only_tests == (f"{_TARGET.key}::api::tests::rust_only",)
    assert report.percentage == 50.0


def test_reports_stale_and_duplicate_mappings(tmp_path: Path) -> None:
    suite = _suite(
        MappingPair(python="test_api.py::removed", rust=f"{_TARGET.key}::api::tests::removed"),
        MappingPair(python="test_api.py::removed", rust=f"{_TARGET.key}::api::tests::decodes"),
    )

    report = audit_mapping(suite, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory)

    assert not report.is_valid
    assert report.missing_python_tests == ("test_api.py::removed",)
    assert report.missing_rust_tests == (f"{_TARGET.key}::api::tests::removed",)
    assert report.duplicate_python_mappings == ("test_api.py::removed",)


def test_rejects_multiple_python_tests_mapped_to_one_rust_test(tmp_path: Path) -> None:
    rust: Final = f"{_TARGET.key}::api::tests::decodes"
    suite = _suite(
        MappingPair(python="test_api.py::test_decode", rust=rust),
        MappingPair(python="test_api.py::test_unmapped", rust=rust),
    )

    report = audit_mapping(suite, tmp_path, python_inventory=_python_inventory, rust_inventory=_rust_inventory)

    assert not report.is_valid
    assert report.mapped_count == 2
    assert report.unmapped_python_tests == ()
    assert report.duplicate_rust_mappings == (rust,)
