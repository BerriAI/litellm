"""Build the independent core distribution from shared repository sources."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
SOURCES: Final = ("litellm", "litellm-rust", ".cargo", "rust-toolchain.toml", "README.md", "LICENSE")


def stage_core_distribution(source: Path, destination: Path) -> None:
    version: Final = subprocess.run(
        ["uv", "version", "--short"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    ignored: Final = frozenset(
        source / name
        for name in subprocess.run(
            [
                "git",
                "ls-files",
                "--ignored",
                "--cached",
                "--others",
                "--exclude-standard",
                "--directory",
                "-z",
                "--",
                *SOURCES,
            ],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split("\0")
        if name
    )

    def ignored_sources(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if Path(directory) / name in ignored} | shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".ruff_cache", "target", ".git", "*.so", "*.pyd"
        )(directory, names)

    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        path: Final = source / name
        if path.is_dir():
            shutil.copytree(
                path,
                destination / name,
                ignore=ignored_sources,
            )
        else:
            shutil.copy2(path, destination / name)
    shutil.copy2(source / "packaging/litellm-core/pyproject.toml", destination / "pyproject.toml")
    subprocess.run(["uv", "version", version, "--frozen"], cwd=destination, check=True)


def build_core_distribution(output: Path, *, sdist_only: bool = False) -> None:
    output_path: Final = output.resolve()
    with tempfile.TemporaryDirectory(prefix="litellm-core-") as temporary:
        stage: Final = Path(temporary)
        stage_core_distribution(ROOT, stage)
        subprocess.run(
            ["uv", "build", "--python", sys.executable, "--out-dir", str(output_path)]
            + (["--sdist"] if sdist_only else []),
            cwd=stage,
            check=True,
        )


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist/core")
    parser.add_argument(
        "--sdist-only", action="store_true", help="Build only the source archive, without compiling a wheel"
    )
    args: Final = parser.parse_args()
    build_core_distribution(Path(args.out_dir), sdist_only=args.sdist_only)


if __name__ == "__main__":
    main()
