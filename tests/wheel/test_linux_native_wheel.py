from __future__ import annotations

from pytest_ci_report import CIMetric
from verify_linux_native_wheel import (
    EXPECTED_ABI_TAG,
    EXPECTED_PLATFORM_TAG,
    EXPECTED_PYTHON_TAG,
    WheelFacts,
)


def test_python_tag_is_cp310(facts: WheelFacts) -> None:
    assert facts.python_tag == EXPECTED_PYTHON_TAG, f"unexpected Python tag: {facts.python_tag}"


def test_abi_tag_is_abi3(facts: WheelFacts) -> None:
    assert facts.abi_tag == EXPECTED_ABI_TAG, f"unexpected ABI tag: {facts.abi_tag}"


def test_platform_tag_is_linux_x86_64(facts: WheelFacts) -> None:
    assert facts.platform_tag == EXPECTED_PLATFORM_TAG, f"unexpected platform tag: {facts.platform_tag}"


def test_dist_info_directory_matches_filename(facts: WheelFacts) -> None:
    assert facts.dist_info_directories == frozenset((facts.expected_dist_info_directory,)), (
        f"unexpected dist-info directories: {facts.dist_info_directories}"
    )


def test_required_dist_info_files_are_present_once(facts: WheelFacts) -> None:
    assert all(count == 1 for count in facts.dist_info_file_counts.values()), (
        f"required dist-info file counts are invalid: {facts.dist_info_file_counts}"
    )


def test_metadata_tags_match_filename(facts: WheelFacts) -> None:
    assert len(facts.wheel_metadata_tags) == len(facts.expanded_filename_tags) and (
        frozenset(facts.wheel_metadata_tags) == facts.expanded_filename_tags
    ), f"WHEEL tags do not match filename: {facts.wheel_metadata_tags}"


def test_debug_sections_are_absent(facts: WheelFacts) -> None:
    assert not facts.debug_sections, f"native extension contains debug sections: {facts.debug_sections}"


def test_static_symbol_table_is_absent(facts: WheelFacts) -> None:
    assert facts.static_symbol_table_absent, "native extension contains a static symbol table"


def test_python_extension_entry_point_is_present(facts: WheelFacts) -> None:
    assert facts.extension_entry_point_present, "native extension does not export PyInit__native"


def test_native_module_loads(facts: WheelFacts) -> None:
    assert facts.native_module_loads, "native extension cannot be loaded"


def test_production_module_omits_panic_hook(facts: WheelFacts) -> None:
    assert facts.panic_test_hook_absent, "production native extension exposes _panic_for_test"


def test_native_extension_does_not_exceed_48_mb(facts: WheelFacts) -> None:
    assert facts.native_size <= 48_000_000, (
        f"native extension exceeds 48 MB: {facts.native_size / 1_000_000:.2f} MB"
    )


def test_tokenizer_vocabularies_are_not_duplicated(facts: WheelFacts) -> None:
    assert not facts.duplicated_vocabularies, (
        f"tokenizer vocabularies are duplicated in the native extension: {facts.duplicated_vocabularies}"
    )


def test_wheel_contents_are_valid(facts: WheelFacts) -> None:
    assert not facts.unexpected_members, f"wheel contains unexpected build artifacts: {facts.unexpected_members}"


def test_build_report(facts: WheelFacts, ci_metric: CIMetric) -> None:
    assert 0 < facts.native_size <= facts.uncompressed_size
    ci_metric("Compressed wheel", round(facts.compressed_size / 1_000_000, 2), "MB")
    ci_metric("Uncompressed wheel", round(facts.uncompressed_size / 1_000_000, 2), "MB")
    ci_metric("Native extension", round(facts.native_size / 1_000_000, 2), "MB")
    ci_metric("Native share", round(facts.native_size / facts.uncompressed_size * 100, 2), "%")
    ci_metric("Commit", facts.commit_sha)
    ci_metric("Platform", facts.platform_tag)
    ci_metric("Python ABI", f"{facts.python_tag}-{facts.abi_tag}")
    ci_metric("Rust compiler", facts.rustc_version)
    ci_metric("Maturin", facts.maturin_version)
    ci_metric("Cargo profile", "release")
