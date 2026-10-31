from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final, Protocol, cast

import pytest


class _CommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]: ...


class _WheelFacts(Protocol):
    python_tag: str
    abi_tag: str
    platform_tag: str
    expected_dist_info_directory: str
    dist_info_directories: frozenset[str]
    dist_info_file_counts: Mapping[str, int]
    wheel_metadata_tags: tuple[str, ...]
    expanded_filename_tags: frozenset[str]
    compressed_size: int
    uncompressed_size: int
    native_size: int
    debug_sections: tuple[str, ...]
    static_symbol_table_absent: bool
    extension_entry_point_present: bool
    native_module_loads: bool
    panic_test_hook_absent: bool
    duplicated_vocabularies: tuple[str, ...]
    unexpected_members: tuple[str, ...]


class _VerifierModule(Protocol):
    def inspect_wheel(
        self,
        wheel: Path,
        environment: Mapping[str, str] | None = None,
        load_native_module: Callable[[Path], ModuleType | None] = ...,
        run_command: _CommandRunner = ...,
    ) -> _WheelFacts: ...


_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "verify_linux_native_wheel.py"
_SPEC: Final = importlib.util.spec_from_file_location("verify_linux_native_wheel", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOADED_VERIFIER: Final = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LOADED_VERIFIER
_SPEC.loader.exec_module(_LOADED_VERIFIER)
verifier: Final = cast(_VerifierModule, _LOADED_VERIFIER)

_EXPECTED_TAG: Final = "cp310-abi3-linux_x86_64"
_NATIVE_MEMBER: Final = "litellm/rust_bridge/_native.abi3.so"
_DIST_INFO: Final = "litellm-1.100.0.dist-info"


def _write_wheel(
    tmp_path: Path,
    *,
    filename_tag: str,
    metadata_tags: tuple[str, ...] | None = (_EXPECTED_TAG,),
    dist_info: str = _DIST_INFO,
    duplicate_wheel: bool = False,
    native_bytes: bytes = b"synthetic native extension",
) -> Path:
    wheel: Final = tmp_path / f"litellm-1.100.0-{filename_tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_NATIVE_MEMBER, native_bytes)
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: litellm\nVersion: 1.100.0\n",
        )
        archive.writestr(
            f"{dist_info}/RECORD",
            f"{_NATIVE_MEMBER},,\n{dist_info}/WHEEL,,\n",
        )
        if metadata_tags is not None:
            wheel_metadata: Final = (
                "Wheel-Version: 1.0\nGenerator: regression-test\nRoot-Is-Purelib: false\n"
                + "".join(f"Tag: {tag}\n" for tag in metadata_tags)
            )
            archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
            if duplicate_wheel:
                archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
    return wheel


_STRIPPED_SECTIONS: Final = "[ 1] .text PROGBITS\n"
_UNSTRIPPED_SECTIONS: Final = "[ 1] .text PROGBITS\n[ 2] .debug_info PROGBITS\n[ 3] .symtab SYMTAB\n"


def _fake_runner(sections: str = _STRIPPED_SECTIONS) -> _CommandRunner:
    def run(
        command: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text
        if command == ("rustc", "--version"):
            return subprocess.CompletedProcess(command, 0, stdout="rustc 1.98.0 (regression-test)\n", stderr="")
        if "--sections" in command:
            return subprocess.CompletedProcess(command, 0, stdout=sections, stderr="")
        if "--dyn-syms" in command:
            return subprocess.CompletedProcess(command, 0, stdout="PyInit__native\n", stderr="")
        raise AssertionError(f"unexpected subprocess command: {command}")

    return run


class _NativeModuleWithPanicHook(ModuleType):
    def _panic_for_test(self) -> None:
        return None


def _inspect(
    wheel: Path,
    *,
    exposes_panic: bool = False,
    sections: str = _STRIPPED_SECTIONS,
) -> _WheelFacts:
    native_module: Final = (
        _NativeModuleWithPanicHook("litellm.rust_bridge._native")
        if exposes_panic
        else ModuleType("litellm.rust_bridge._native")
    )

    def _fake_load_native_module(_: Path) -> ModuleType:
        return native_module

    environment: Final = MappingProxyType({"RELEASE_WHEEL_COMMIT_SHA": "0" * 40})
    return verifier.inspect_wheel(
        wheel,
        environment,
        _fake_load_native_module,
        _fake_runner(sections),
    )


def test_reports_expected_release_wheel_tags(tmp_path: Path) -> None:
    native_bytes: Final = b"synthetic native extension"
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG, native_bytes=native_bytes)

    facts: Final = _inspect(wheel)

    assert (facts.python_tag, facts.abi_tag, facts.platform_tag) == ("cp310", "abi3", "linux_x86_64")
    assert facts.dist_info_directories == frozenset((facts.expected_dist_info_directory,))
    assert facts.expected_dist_info_directory == _DIST_INFO
    assert dict(facts.dist_info_file_counts) == {"METADATA": 1, "RECORD": 1, "WHEEL": 1}
    assert frozenset(facts.wheel_metadata_tags) == facts.expanded_filename_tags
    assert len(facts.wheel_metadata_tags) == len(facts.expanded_filename_tags)
    assert facts.native_size == len(native_bytes)
    assert facts.compressed_size == wheel.stat().st_size
    assert facts.uncompressed_size >= facts.native_size > 0
    assert not facts.duplicated_vocabularies
    assert not facts.unexpected_members
    assert facts.extension_entry_point_present and facts.native_module_loads
    assert facts.panic_test_hook_absent


def test_reports_debug_sections_and_static_symbol_table(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    facts: Final = _inspect(wheel, sections=_UNSTRIPPED_SECTIONS)

    assert facts.debug_sections == (".debug_",)
    assert not facts.static_symbol_table_absent


def test_reports_cp312_version_specific_tags(tmp_path: Path) -> None:
    tag: Final = "cp312-cp312-linux_x86_64"
    wheel: Final = _write_wheel(tmp_path, filename_tag=tag, metadata_tags=(tag,))

    facts: Final = _inspect(wheel)

    assert (facts.python_tag, facts.abi_tag) == ("cp312", "cp312")


def test_reports_non_linux_platform_tag(tmp_path: Path) -> None:
    tag: Final = "cp310-abi3-win_amd64"
    wheel: Final = _write_wheel(tmp_path, filename_tag=tag, metadata_tags=(tag,))

    facts: Final = _inspect(wheel)

    assert facts.platform_tag == "win_amd64"


@pytest.mark.parametrize(
    "metadata_tags",
    (None, ("cp312-cp312-linux_x86_64",)),
    ids=("missing", "mismatched"),
)
def test_reports_missing_or_mismatched_wheel_metadata_tag(
    tmp_path: Path,
    metadata_tags: tuple[str, ...] | None,
) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG, metadata_tags=metadata_tags)

    facts: Final = _inspect(wheel)

    assert frozenset(facts.wheel_metadata_tags) != facts.expanded_filename_tags


def test_reports_wheel_metadata_from_wrong_dist_info_directory(
    tmp_path: Path,
) -> None:
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        dist_info="decoy-1.0.0.dist-info",
    )

    facts: Final = _inspect(wheel)

    assert facts.dist_info_directories == frozenset(("decoy-1.0.0.dist-info",))
    assert facts.dist_info_directories != frozenset((facts.expected_dist_info_directory,))
    assert dict(facts.dist_info_file_counts) == {"METADATA": 0, "RECORD": 0, "WHEEL": 0}


def test_reports_duplicate_wheel_metadata_tags(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        metadata_tags=(_EXPECTED_TAG, _EXPECTED_TAG),
    )

    facts: Final = _inspect(wheel)

    assert facts.wheel_metadata_tags == (_EXPECTED_TAG, _EXPECTED_TAG)
    assert len(facts.wheel_metadata_tags) != len(facts.expanded_filename_tags)


def test_reports_duplicate_wheel_metadata_file(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        wheel: Final = _write_wheel(
            tmp_path,
            filename_tag=_EXPECTED_TAG,
            duplicate_wheel=True,
        )

    facts: Final = _inspect(wheel)

    assert facts.dist_info_file_counts["WHEEL"] == 2


def test_reports_production_module_exposing_panic_hook(tmp_path: Path) -> None:
    wheel: Final = _write_wheel(tmp_path, filename_tag=_EXPECTED_TAG)

    facts: Final = _inspect(wheel, exposes_panic=True)

    assert not facts.panic_test_hook_absent


@pytest.mark.parametrize("embedded", (False, True))
def test_vocabulary_is_packaged_once(tmp_path: Path, embedded: bool) -> None:
    ranks: Final = b"AA== 0\nAQ== 1\nAg== 2\n"
    wheel: Final = _write_wheel(
        tmp_path,
        filename_tag=_EXPECTED_TAG,
        native_bytes=b"native engine" + (ranks if embedded else b""),
    )
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("litellm/litellm_core_utils/tokenizers/" + "a" * 40, ranks)

    facts: Final = _inspect(wheel)

    assert bool(facts.duplicated_vocabularies) == embedded
