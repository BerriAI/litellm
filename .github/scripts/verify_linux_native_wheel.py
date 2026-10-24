from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Final, Protocol

EXPECTED_PYTHON_TAG: Final = "cp310"
EXPECTED_ABI_TAG: Final = "abi3"
EXPECTED_PLATFORM_TAG: Final = "linux_x86_64"


@dataclass(frozen=True, slots=True)
class WheelInspectionError:
    message: str


@dataclass(frozen=True, slots=True)
class WheelFacts:
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
    commit_sha: str
    rustc_version: str
    maturin_version: str


class CommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]: ...


def _run_command(
    command: tuple[str, ...],
    *,
    check: bool,
    capture_output: bool,
    text: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=capture_output, text=text)


def _dist_info_directory(member: zipfile.ZipInfo) -> str | None:
    parts: Final = PurePosixPath(member.filename).parts
    if not parts or not parts[0].endswith(".dist-info"):
        return None
    return parts[0]


def _wheel_metadata_tags(archive: zipfile.ZipFile, members: tuple[zipfile.ZipInfo, ...]) -> tuple[str, ...]:
    if len(members) != 1:
        return ()
    lines: Final = archive.read(members[0]).splitlines()
    return tuple(line.removeprefix(b"Tag:").strip().decode("ascii") for line in lines if line.startswith(b"Tag:"))


def _load_native_module(native_path: Path) -> ModuleType | None:
    module_spec: Final = importlib.util.spec_from_file_location("litellm.rust_bridge._native", native_path)
    if module_spec is None or module_spec.loader is None:
        return None
    try:
        native_module: Final = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(native_module)
    except Exception as error:  # noqa: BLE001  # native module initialization can raise arbitrary exceptions
        sys.stderr.write(f"native module load failed: {error}\n")
        return None
    return native_module


def inspect_wheel(
    wheel: Path,
    environment: Mapping[str, str] | None = None,
    load_native_module: Callable[[Path], ModuleType | None] = _load_native_module,
    run_command: CommandRunner = _run_command,
) -> WheelFacts | WheelInspectionError:
    resolved_environment: Final = os.environ if environment is None else environment
    wheel_tags: Final = wheel.stem.rsplit("-", maxsplit=3)
    if len(wheel_tags) != 4:
        return WheelInspectionError(f"cannot parse wheel tags from {wheel.name}")

    wheel_identity: Final = wheel_tags[0].split("-")
    if len(wheel_identity) != 2 or wheel_identity[0] != "litellm" or not wheel_identity[1]:
        return WheelInspectionError(f"unexpected wheel identity: {wheel_tags[0]}")

    expected_dist_info_directory: Final = f"{wheel_tags[0]}.dist-info"
    python_tag: Final = wheel_tags[1]
    abi_tag: Final = wheel_tags[2]
    platform_tag: Final = wheel_tags[3]
    expanded_filename_tags: Final = frozenset(
        "-".join(tag) for tag in product(python_tag.split("."), abi_tag.split("."), platform_tag.split("."))
    )

    with zipfile.ZipFile(wheel) as archive:
        wheel_members: Final = archive.infolist()
        dist_info_directories: Final = frozenset(
            directory for member in wheel_members if (directory := _dist_info_directory(member)) is not None
        )
        required_dist_info_files: Final = ("METADATA", "RECORD", "WHEEL")
        dist_info_file_counts: Final = MappingProxyType(
            {
                filename: sum(
                    member.filename == f"{expected_dist_info_directory}/{filename}" for member in wheel_members
                )
                for filename in required_dist_info_files
            }
        )
        wheel_metadata_members: Final = tuple(
            member for member in wheel_members if member.filename == f"{expected_dist_info_directory}/WHEEL"
        )
        wheel_metadata_tags: Final = _wheel_metadata_tags(archive, wheel_metadata_members)
        native_members: Final = tuple(
            member
            for member in wheel_members
            if member.filename.startswith("litellm/rust_bridge/_native.") and member.filename.endswith(".so")
        )
        if len(native_members) != 1:
            return WheelInspectionError(f"expected one native extension, found {len(native_members)}")

        unexpected_members: Final = tuple(
            member.filename
            for member in wheel_members
            if member.filename.endswith((".pdb", ".dwp", ".rlib", ".rmeta", "Cargo.toml", "Cargo.lock"))
            or any(part.endswith(".dSYM") for part in PurePosixPath(member.filename).parts)
        )
        native_member: Final = native_members[0]
        uncompressed_wheel_size: Final = sum(member.file_size for member in wheel_members)
        native_path: Final = wheel.parent / "native" / Path(native_member.filename).name
        native_path.parent.mkdir(parents=True, exist_ok=True)
        native_bytes: Final = archive.read(native_member)
        native_path.write_bytes(native_bytes)
        duplicated_vocabularies: Final = tuple(
            member.filename
            for member in wheel_members
            if member.filename.startswith("litellm/litellm_core_utils/tokenizers/")
            and re.fullmatch(r"[0-9a-f]{40}", PurePosixPath(member.filename).name)
            and member.file_size > 0
            and archive.read(member) in native_bytes
        )

    commit_sha: Final = resolved_environment.get(
        "RELEASE_WHEEL_COMMIT_SHA", resolved_environment.get("GITHUB_SHA", "unknown")
    )
    rustc_version: Final = run_command(
        ("rustc", "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pyproject: Final = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    maturin_match: Final = re.search(r'"maturin==([^";]+)', pyproject)
    if maturin_match is None:
        return WheelInspectionError("build-system does not pin an exact Maturin version")

    maturin_version: Final = maturin_match.group(1)
    sections: Final = run_command(
        ("readelf", "--sections", "--wide", str(native_path)),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    debug_sections: Final = tuple(section for section in (".debug_", ".zdebug_") if section in sections)
    static_symbol_table_absent: Final = ".symtab" not in sections

    dynamic_symbols: Final = run_command(
        ("readelf", "--dyn-syms", "--wide", str(native_path)),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    extension_entry_point_present: Final = "PyInit__native" in dynamic_symbols
    native_module: Final = load_native_module(native_path)
    native_module_loads: Final = native_module is not None
    panic_test_hook_absent: Final = native_module is not None and not hasattr(native_module, "_panic_for_test")
    return WheelFacts(
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        expected_dist_info_directory=expected_dist_info_directory,
        dist_info_directories=dist_info_directories,
        dist_info_file_counts=dist_info_file_counts,
        wheel_metadata_tags=wheel_metadata_tags,
        expanded_filename_tags=expanded_filename_tags,
        compressed_size=wheel.stat().st_size,
        uncompressed_size=uncompressed_wheel_size,
        native_size=native_member.file_size,
        debug_sections=debug_sections,
        static_symbol_table_absent=static_symbol_table_absent,
        extension_entry_point_present=extension_entry_point_present,
        native_module_loads=native_module_loads,
        panic_test_hook_absent=panic_test_hook_absent,
        duplicated_vocabularies=duplicated_vocabularies,
        unexpected_members=unexpected_members,
        commit_sha=commit_sha,
        rustc_version=rustc_version,
        maturin_version=maturin_version,
    )
