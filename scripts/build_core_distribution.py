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
    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        path: Final = source / name
        if path.is_dir():
            shutil.copytree(
                path,
                destination / name,
                ignore=shutil.ignore_patterns(
                    "__pycache__", ".pytest_cache", ".ruff_cache", "target", ".git", "*.so", "*.pyd"
                ),
            )
        else:
            shutil.copy2(path, destination / name)
    shutil.copy2(source / "packaging/litellm-core/pyproject.toml", destination / "pyproject.toml")
    subprocess.run(["uv", "version", version, "--frozen"], cwd=destination, check=True)


def build_core_distribution(output: Path) -> None:
    output_path: Final = output.resolve()
    with tempfile.TemporaryDirectory(prefix="litellm-core-") as temporary:
        stage: Final = Path(temporary)
        stage_core_distribution(ROOT, stage)
        subprocess.run(
            ["uv", "build", "--python", sys.executable, "--out-dir", str(output_path)],
            cwd=stage,
            check=True,
        )


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist/core")
    args: Final = parser.parse_args()
    build_core_distribution(Path(args.out_dir))


if __name__ == "__main__":
    main()
