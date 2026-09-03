from __future__ import annotations

from pathlib import Path

from .mapping_validator import MappingSuite, TestMapping as MappingPair, audit_mapping


def _write_inventory(tmp_path: Path) -> None:
    (tmp_path / "test_api.py").write_text(
        "def test_decode():\n    pass\n\ndef test_unmapped():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "lib.rs").write_text(
        "#[test]\nfn decodes() {}\n\n#[test]\nfn rust_only() {}\n",
        encoding="utf-8",
    )


def _suite(*mappings: MappingPair) -> MappingSuite:
    return MappingSuite(
        python_scope=("test_api.py",),
        unit_parity_scope=("test_api.py",),
        rust_scope=("lib.rs",),
        cargo_manifest="Cargo.toml",
        cargo_filter="api",
        mappings=mappings,
    )


def test_derives_mapping_status_from_live_inventories(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    suite = _suite(MappingPair(python="test_api.py::test_decode", rust="lib.rs::decodes"))

    report = audit_mapping(suite, tmp_path)

    assert report.is_valid
    assert report.mapped_python_tests == ("test_api.py::test_decode",)
    assert report.unmapped_python_tests == ("test_api.py::test_unmapped",)
    assert report.rust_only_tests == ("lib.rs::rust_only",)
    assert report.percentage == 50.0


def test_reports_stale_and_duplicate_mappings(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    suite = _suite(
        MappingPair(python="test_api.py::removed", rust="lib.rs::removed"),
        MappingPair(python="test_api.py::removed", rust="lib.rs::decodes"),
    )

    report = audit_mapping(suite, tmp_path)

    assert not report.is_valid
    assert report.missing_python_tests == ("test_api.py::removed",)
    assert report.missing_rust_tests == ("lib.rs::removed",)
    assert report.duplicate_python_mappings == ("test_api.py::removed",)


def test_allows_multiple_python_tests_to_map_to_one_rust_test(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    suite = _suite(
        MappingPair(python="test_api.py::test_decode", rust="lib.rs::decodes"),
        MappingPair(python="test_api.py::test_unmapped", rust="lib.rs::decodes"),
    )

    report = audit_mapping(suite, tmp_path)

    assert report.is_valid
    assert report.mapped_count == 2
    assert report.unmapped_python_tests == ()
