from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Final, Protocol

EXPECTED_PLATFORM_TAG: Final = "linux_x86_64"
REPO_ROOT: Final = Path(__file__).parents[2]


@dataclass(frozen=True, slots=True)
class WheelProfile:
    distribution: str
    python_tag: str
    abi_tag: str
    project_directory: Path
    label: str
    report_title: str
    size_limit: int
    is_payload: Callable[[str], bool]
    is_extension: bool


PROFILES: Final = MappingProxyType(
    {
        "extension": WheelProfile(
            distribution="litellm",
            python_tag="cp310",
            abi_tag="abi3",
            project_directory=REPO_ROOT,
            label="Native extension",
            report_title="Native wheel build report",
            size_limit=48_000_000,
            is_payload=lambda name: name.startswith("litellm/rust_bridge/_native.") and name.endswith(".so"),
            is_extension=True,
        ),
        "gateway": WheelProfile(
            distribution="litellm_gateway",
            python_tag="py3",
            abi_tag="none",
            project_directory=REPO_ROOT / "litellm-rust" / "crates" / "gateway",
            label="Gateway binary",
            report_title="Gateway wheel build report",
            size_limit=37_000_000,
            is_payload=lambda name: name.endswith(".data/scripts/litellm-gateway"),
            is_extension=False,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Validation:
    label: str
    passed: bool
    failure: str


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


def _extension_validations(
    wheel: Path,
    native_path: Path,
    native_bytes: bytes,
    load_native_module: Callable[[Path], ModuleType | None],
    run_command: CommandRunner,
) -> tuple[Validation, ...]:
    with zipfile.ZipFile(wheel) as archive:
        duplicated_vocabularies: Final = tuple(
            member.filename
            for member in archive.infolist()
            if member.filename.startswith("litellm/litellm_core_utils/tokenizers/")
            and re.fullmatch(r"[0-9a-f]{40}", PurePosixPath(member.filename).name)
            and member.file_size > 0
            and archive.read(member) in native_bytes
        )
    dynamic_symbols: Final = run_command(
        ("readelf", "--dyn-syms", "--wide", str(native_path)),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    native_module: Final = load_native_module(native_path)
    return (
        Validation(
            "Tokenizer vocabularies are not duplicated in the native extension",
            not duplicated_vocabularies,
            f"native extension duplicates tokenizer vocabularies: {', '.join(duplicated_vocabularies)}",
        ),
        Validation(
            "Python extension entry point is present",
            "PyInit__native" in dynamic_symbols,
            "native extension does not export PyInit__native",
        ),
        Validation("Native module loads", native_module is not None, "native module did not load"),
        Validation(
            "Production module omits the panic test hook",
            native_module is None or not hasattr(native_module, "_panic_for_test"),
            "production native module exposes _panic_for_test",
        ),
    )


def _write_report(report: str, summary_path: str | None) -> None:
    if summary_path is None:
        sys.stdout.write(report + "\n")
        return
    with Path(summary_path).open("a") as summary:
        summary.write(report + "\n")


def main(
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    load_native_module: Callable[[Path], ModuleType | None] = _load_native_module,
    run_command: CommandRunner = _run_command,
) -> int:
    arguments: Final = tuple(sys.argv if argv is None else argv)
    resolved_environment: Final = os.environ if environment is None else environment
    if len(arguments) != 3 or arguments[1] not in PROFILES:
        sys.stderr.write("usage: verify_linux_native_wheel.py {extension|gateway} WHEEL\n")
        return 2

    profile: Final = PROFILES[arguments[1]]
    wheel: Final = Path(arguments[2])
    wheel_tags: Final = wheel.stem.rsplit("-", maxsplit=3)
    if len(wheel_tags) != 4:
        sys.stderr.write(f"cannot parse wheel tags from {wheel.name}\n")
        return 1

    wheel_identity: Final = wheel_tags[0].split("-")
    if len(wheel_identity) != 2 or wheel_identity[0] != profile.distribution or not wheel_identity[1]:
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
        native_members: Final = tuple(member for member in wheel_members if profile.is_payload(member.filename))
        if len(native_members) != 1:
            sys.stderr.write(f"expected one {profile.label.lower()}, found {len(native_members)}\n")
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
        native_bytes: Final = archive.read(native_member)
        native_path.write_bytes(native_bytes)

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
    pyproject: Final = (profile.project_directory / "pyproject.toml").read_text()
    maturin_match: Final = re.search(r'"maturin==([^";]+)', pyproject)
    if maturin_match is None:
        sys.stderr.write("build-system does not pin an exact Maturin version\n")
        return 1

    maturin_version: Final = maturin_match.group(1)
    native_percentage: Final = native_member.file_size / uncompressed_wheel_size * 100
    size_report: Final = "\n".join(
        (
            f"## {profile.report_title}",
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
            f"| {profile.label} | {native_member.file_size / 1_000_000:.2f} MB |",
            f"| Native share | {native_percentage:.2f}% |",
            "",
        )
    )
    summary_path: Final = resolved_environment.get("GITHUB_STEP_SUMMARY")
    _write_report(size_report, summary_path)

    sections: Final = run_command(
        ("readelf", "--sections", "--wide", str(native_path)),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    debug_sections: Final = tuple(section for section in (".debug_", ".zdebug_") if section in sections)
    validations: Final = (
        Validation(
            f"Python tag is {profile.python_tag}",
            python_tag == profile.python_tag,
            f"unexpected Python tag: expected {profile.python_tag}, found {python_tag}",
        ),
        Validation(
            f"ABI tag is {profile.abi_tag}",
            abi_tag == profile.abi_tag,
            f"unexpected ABI tag: expected {profile.abi_tag}, found {abi_tag}",
        ),
        Validation(
            f"Platform tag is {EXPECTED_PLATFORM_TAG}",
            platform_tag == EXPECTED_PLATFORM_TAG,
            f"unexpected platform tag: expected {EXPECTED_PLATFORM_TAG}, found {platform_tag}",
        ),
        Validation(
            "Wheel dist-info directory matches the filename",
            dist_info_directories == expected_dist_info_directories,
            f"unexpected dist-info directories: expected {expected_dist_info_directory}, "
            f"found {', '.join(sorted(dist_info_directories))}",
        ),
        Validation(
            "Required dist-info files are present exactly once",
            all(count == 1 for count in dist_info_file_counts.values()),
            f"required dist-info file counts are invalid: {dist_info_file_counts}",
        ),
        Validation(
            "Wheel metadata tags match the filename",
            wheel_metadata_tags_match,
            f"WHEEL tags do not match filename: expected {', '.join(sorted(expanded_filename_tags))}, "
            f"found {', '.join(sorted(wheel_metadata_tags))}",
        ),
        Validation(
            "Debug sections are absent",
            not debug_sections,
            f"{native_member.filename} contains debug sections: {', '.join(debug_sections)}",
        ),
        Validation(
            "Static symbol table is absent",
            ".symtab" not in sections,
            f"{native_member.filename} contains a static symbol table",
        ),
        Validation(
            f"{profile.label} does not exceed {profile.size_limit / 1_000_000:.0f} MB",
            native_member.file_size <= profile.size_limit,
            f"{profile.label.lower()} exceeds {profile.size_limit / 1_000_000:.0f} MB: "
            f"{native_member.file_size / 1_000_000:.2f} MB",
        ),
        Validation(
            "Wheel contents are valid",
            not unexpected_members,
            f"wheel contains unexpected build artifacts: {', '.join(unexpected_members)}",
        ),
    ) + (
        _extension_validations(wheel, native_path, native_bytes, load_native_module, run_command)
        if profile.is_extension
        else ()
    )

    verified_report: Final = "\n".join(
        ("", "| Validation | Expected | Result |", "| --- | --- | :---: |")
        + tuple(f"| {row.label} | Yes | {'O' if row.passed else 'X'} |" for row in validations)
        + ("",)
    )
    _write_report(verified_report, summary_path)

    validation_errors: Final = tuple(row.failure for row in validations if not row.passed)
    sys.stderr.write("".join(f"{message}\n" for message in validation_errors))

    return 0 if not validation_errors else 1


if __name__ == "__main__":
    sys.exit(main())
