from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Final, Protocol

EXPECTED_PYTHON_TAG: Final = "cp310"
EXPECTED_ABI_TAG: Final = "abi3"
EXPECTED_PLATFORM_TAG: Final = "linux_x86_64"


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


def main(
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    load_native_module: Callable[[Path], ModuleType | None] = _load_native_module,
    run_command: CommandRunner = _run_command,
) -> int:
    arguments: Final = tuple(sys.argv if argv is None else argv)
    resolved_environment: Final = os.environ if environment is None else environment
    if len(arguments) != 2:
        sys.stderr.write(f"usage: {Path(arguments[0]).name} WHEEL\n")
        return 2

    wheel: Final = Path(arguments[1])
    wheel_tags: Final = wheel.stem.rsplit("-", maxsplit=3)
    if len(wheel_tags) != 4:
        sys.stderr.write(f"cannot parse wheel tags from {wheel.name}\n")
        return 1

    wheel_identity: Final = wheel_tags[0].split("-")
    if len(wheel_identity) != 2 or wheel_identity[0] != "litellm" or not wheel_identity[1]:
        sys.stderr.write(f"unexpected wheel identity: {wheel_tags[0]}\n")
        return 1

    expected_dist_info_directory: Final = f"{wheel_tags[0]}.dist-info"
    expected_dist_info_directories: Final = frozenset((expected_dist_info_directory,))
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
            sys.stderr.write(f"expected one native extension, found {len(native_members)}\n")
            return 1

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
        native_path.write_bytes(archive.read(native_member))

    wheel_metadata_tags_match: Final = (
        len(wheel_metadata_tags) == len(expanded_filename_tags)
        and frozenset(wheel_metadata_tags) == expanded_filename_tags
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
        sys.stderr.write("build-system does not pin an exact Maturin version\n")
        return 1

    maturin_version: Final = maturin_match.group(1)
    native_percentage: Final = native_member.file_size / uncompressed_wheel_size * 100
    size_report: Final = "\n".join(
        (
            "## Native wheel build report",
            "",
            "| Build | Value |",
            "| --- | --- |",
            f"| Commit | `{commit_sha}` |",
            f"| Platform | `{platform_tag}` |",
            f"| Python ABI | `{python_tag}-{abi_tag}` |",
            f"| Rust compiler | `{rustc_version}` |",
            f"| Maturin | `{maturin_version}` |",
            "| Cargo profile | `release` |",
            "",
            "| Artifact | Size |",
            "| --- | ---: |",
            f"| Compressed wheel | {wheel.stat().st_size / 1_000_000:.2f} MB |",
            f"| Uncompressed wheel | {uncompressed_wheel_size / 1_000_000:.2f} MB |",
            f"| Native extension | {native_member.file_size / 1_000_000:.2f} MB |",
            f"| Native share | {native_percentage:.2f}% |",
            "",
        )
    )
    summary_path: Final = resolved_environment.get("GITHUB_STEP_SUMMARY")
    if summary_path is None:
        sys.stdout.write(size_report)
    else:
        Path(summary_path).write_text(size_report)

    sections: Final = run_command(
        ("readelf", "--sections", "--wide", str(native_path)),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    debug_sections: Final = tuple(section for section in (".debug_", ".zdebug_") if section in sections)
    debug_sections_absent: Final = not debug_sections
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
    native_size_limit: Final = 20_000_000
    native_size_within_limit: Final = native_member.file_size <= native_size_limit
    validations: Final = (
        (f"Python tag is {EXPECTED_PYTHON_TAG}", python_tag == EXPECTED_PYTHON_TAG),
        (f"ABI tag is {EXPECTED_ABI_TAG}", abi_tag == EXPECTED_ABI_TAG),
        (f"Platform tag is {EXPECTED_PLATFORM_TAG}", platform_tag == EXPECTED_PLATFORM_TAG),
        ("Wheel dist-info directory matches the filename", dist_info_directories == expected_dist_info_directories),
        (
            "Required dist-info files are present exactly once",
            all(count == 1 for count in dist_info_file_counts.values()),
        ),
        ("Wheel metadata tags match the filename", wheel_metadata_tags_match),
        ("Debug sections are absent", debug_sections_absent),
        ("Static symbol table is absent", static_symbol_table_absent),
        ("Python extension entry point is present", extension_entry_point_present),
        ("Native module loads", native_module_loads),
        ("Production module omits the panic test hook", panic_test_hook_absent),
        ("Native extension does not exceed 20 MB", native_size_within_limit),
        ("Wheel contents are valid", not unexpected_members),
    )

    verified_report: Final = size_report + "\n".join(
        ("", "| Validation | Expected | Result |", "| --- | --- | :---: |")
        + tuple(f"| {label} | Yes | {'O' if passed else 'X'} |" for label, passed in validations)
        + ("",)
    )
    if summary_path is not None:
        Path(summary_path).write_text(verified_report)

    invalid_dist_info_files: Final = any(count != 1 for count in dist_info_file_counts.values())
    validation_errors: Final = tuple(
        message
        for failed, message in (
            (bool(debug_sections), f"{native_member.filename} contains debug sections: {', '.join(debug_sections)}"),
            (not static_symbol_table_absent, f"{native_member.filename} contains a static symbol table"),
            (not extension_entry_point_present, "native extension does not export PyInit__native"),
            (
                python_tag != EXPECTED_PYTHON_TAG,
                f"unexpected Python tag: expected {EXPECTED_PYTHON_TAG}, found {python_tag}",
            ),
            (abi_tag != EXPECTED_ABI_TAG, f"unexpected ABI tag: expected {EXPECTED_ABI_TAG}, found {abi_tag}"),
            (
                platform_tag != EXPECTED_PLATFORM_TAG,
                f"unexpected platform tag: expected {EXPECTED_PLATFORM_TAG}, found {platform_tag}",
            ),
            (
                dist_info_directories != expected_dist_info_directories,
                f"unexpected dist-info directories: expected {expected_dist_info_directory}, "
                f"found {', '.join(sorted(dist_info_directories))}",
            ),
            (invalid_dist_info_files, f"required dist-info file counts are invalid: {dist_info_file_counts}"),
            (
                not invalid_dist_info_files and not wheel_metadata_tags_match,
                f"WHEEL tags do not match filename: expected {', '.join(sorted(expanded_filename_tags))}, "
                f"found {', '.join(sorted(wheel_metadata_tags))}",
            ),
            (
                native_module is not None and not panic_test_hook_absent,
                "production native module exposes _panic_for_test",
            ),
            (
                not native_size_within_limit,
                f"native extension exceeds 20 MB: {native_member.file_size / 1_000_000:.2f} MB",
            ),
            (bool(unexpected_members), f"wheel contains unexpected build artifacts: {', '.join(unexpected_members)}"),
        )
        if failed
    )
    sys.stderr.write("".join(f"{message}\n" for message in validation_errors))

    return 0 if all(passed for _, passed in validations) else 1


if __name__ == "__main__":
    sys.exit(main())
