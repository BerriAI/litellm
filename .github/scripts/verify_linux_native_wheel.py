from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final


def _loads_native_module(native_path: Path) -> bool:
    module_spec: Final = importlib.util.spec_from_file_location("litellm.rust_bridge._native", native_path)
    if module_spec is None or module_spec.loader is None:
        return False
    try:
        native_module: Final = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(native_module)
    except Exception as error:
        sys.stderr.write(f"native module load failed: {error}\n")
        return False
    return True


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {Path(sys.argv[0]).name} WHEEL\n")
        return 2

    wheel: Final = Path(sys.argv[1])
    with zipfile.ZipFile(wheel) as archive:
        wheel_members: Final = archive.infolist()
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

    wheel_tags: Final = wheel.stem.rsplit("-", maxsplit=3)
    if len(wheel_tags) != 4:
        sys.stderr.write(f"cannot parse wheel tags from {wheel.name}\n")
        return 1

    python_tag: Final = wheel_tags[1]
    abi_tag: Final = wheel_tags[2]
    platform_tag: Final = wheel_tags[3]
    commit_sha: Final = os.environ.get("RELEASE_WHEEL_COMMIT_SHA", os.environ.get("GITHUB_SHA", "unknown"))
    rustc_version: Final = subprocess.run(
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
    summary_path: Final = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path is None:
        sys.stdout.write(size_report)
    else:
        Path(summary_path).write_text(size_report)

    sections: Final = subprocess.run(
        ("readelf", "--sections", "--wide", native_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    debug_sections: Final = tuple(section for section in (".debug_", ".zdebug_") if section in sections)
    debug_sections_absent: Final = not debug_sections
    static_symbol_table_absent: Final = ".symtab" not in sections

    dynamic_symbols: Final = subprocess.run(
        ("readelf", "--dyn-syms", "--wide", native_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    extension_entry_point_present: Final = "PyInit__native" in dynamic_symbols
    native_module_loads: Final = _loads_native_module(native_path)
    native_size_limit: Final = 20_000_000
    native_size_within_limit: Final = native_member.file_size <= native_size_limit
    validations: Final = (
        ("Debug sections are absent", debug_sections_absent),
        ("Static symbol table is absent", static_symbol_table_absent),
        ("Python extension entry point is present", extension_entry_point_present),
        ("Native module loads", native_module_loads),
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

    if debug_sections:
        sys.stderr.write(f"{native_member.filename} contains debug sections: {', '.join(debug_sections)}\n")
    if not static_symbol_table_absent:
        sys.stderr.write(f"{native_member.filename} contains a static symbol table\n")
    if not extension_entry_point_present:
        sys.stderr.write("native extension does not export PyInit__native\n")
    if not native_size_within_limit:
        sys.stderr.write(f"native extension exceeds 20 MB: {native_member.file_size / 1_000_000:.2f} MB\n")
    if unexpected_members:
        sys.stderr.write(f"wheel contains unexpected build artifacts: {', '.join(unexpected_members)}\n")

    return 0 if all(passed for _, passed in validations) else 1


if __name__ == "__main__":
    sys.exit(main())
