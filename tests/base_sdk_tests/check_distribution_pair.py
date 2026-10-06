"""Check independently built distributions share implementation, not dependencies."""

import argparse
import sys
from email.parser import Parser
from pathlib import Path
from typing import Final
from zipfile import ZipFile


def check_distribution_pair(directory: Path) -> None:
    legacy_wheels: Final = tuple(directory.glob("litellm-[0-9]*.whl"))
    core_wheels: Final = tuple(directory.glob("litellm_core-*.whl"))
    assert legacy_wheels and core_wheels, "Build both independent SDK wheels"
    for legacy_path in legacy_wheels:
        with ZipFile(legacy_path) as legacy:
            legacy_files: Final = frozenset(legacy.namelist())
            legacy_metadata_path: Final = next(name for name in legacy_files if name.endswith(".dist-info/METADATA"))
            legacy_metadata: Final = Parser().parsestr(legacy.read(legacy_metadata_path).decode())
            for core_path in core_wheels:
                with ZipFile(core_path) as core:
                    core_files: Final = frozenset(core.namelist())
                    core_metadata_path: Final = next(
                        name for name in core_files if name.endswith(".dist-info/METADATA")
                    )
                    core_metadata: Final = Parser().parsestr(core.read(core_metadata_path).decode())
                    assert legacy_metadata["Version"] == core_metadata["Version"]
                    for archive, names, metadata, other in (
                        (legacy, legacy_files, legacy_metadata, "litellm-core"),
                        (core, core_files, core_metadata, "litellm"),
                    ):
                        assert "litellm/__init__.py" in names
                        assert any(
                            name.startswith("litellm/rust_bridge/_native.") and name.endswith((".so", ".pyd"))
                            for name in names
                        ), "Each SDK retains the native extension"
                        has_ui: Final = "litellm/proxy/_experimental/out/index.html" in names
                        assert has_ui == (metadata["Name"] == "litellm"), "Only legacy bundles its dashboard"
                        if metadata["Name"] == "litellm-core":
                            assert not any(name.startswith("litellm/proxy/_experimental/out/") for name in names)
                        assert not any(
                            value.lower().startswith((other + "[", other + " ", other + "="))
                            for value in metadata.get_all("Requires-Dist", ())
                        ), "Neither SDK may install the other distribution"
                    shared: Final = frozenset(
                        name for name in legacy_files if name.startswith("litellm/") and name.endswith(".py")
                    )
                    assert shared == frozenset(
                        name for name in core_files if name.startswith("litellm/") and name.endswith(".py")
                    )
                    assert all(legacy.read(name) == core.read(name) for name in shared)
                    assert legacy.read(legacy_metadata_path.replace("METADATA", "entry_points.txt")) == core.read(
                        core_metadata_path.replace("METADATA", "entry_points.txt")
                    )
                    sys.stdout.write(f"Verified {len(shared)} identical Python files and independent metadata\n")


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check_distribution_pair(parser.parse_args().directory)
