from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Final

CHILD_SCRIPT: Final = """
from importlib.resources import files
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])

import litellm.rust_bridge._native as module

before = module.gil_stats()
if not isinstance(before.get("releases"), int):
    raise AssertionError(f"unexpected gil_stats result: {before!r}")

try:
    module._panic_for_test()
except BaseException as error:
    if type(error).__name__ != "PanicException":
        raise AssertionError(f"expected PanicException, got {type(error).__name__}") from error
else:
    raise AssertionError("Rust panic returned without raising")

after = module.gil_stats()
if not isinstance(after.get("releases"), int):
    raise AssertionError(f"native module unusable after panic: {after!r}")

wheel_cost_map = files("litellm").joinpath("model_prices_and_context_window_backup.json").read_bytes()
if module.bundled_model_cost_map() != wheel_cost_map:
    raise AssertionError("the native cost map does not match the wheel package resource")
"""


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {Path(sys.argv[0]).name} WHEEL\n")
        return 2

    wheel: Final = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as temporary_directory, zipfile.ZipFile(wheel) as archive:
        native_members: Final = tuple(
            member
            for member in archive.infolist()
            if member.filename.startswith("litellm/rust_bridge/_native.") and member.filename.endswith(".so")
        )
        if len(native_members) != 1:
            sys.stderr.write(f"expected one native extension, found {len(native_members)}\n")
            return 1

        package_path: Final = Path(temporary_directory) / "litellm"
        bridge_path: Final = package_path / "rust_bridge"
        bridge_path.mkdir(parents=True)
        (package_path / "__init__.py").write_bytes(b"")
        (bridge_path / "__init__.py").write_bytes(b"")
        (bridge_path / Path(native_members[0].filename).name).write_bytes(archive.read(native_members[0]))
        (package_path / "model_prices_and_context_window_backup.json").write_bytes(
            archive.read("litellm/model_prices_and_context_window_backup.json")
        )
        result: Final = subprocess.run(
            (sys.executable, "-c", CHILD_SCRIPT, temporary_directory), check=False
        )

    if result.returncode != 0:
        sys.stderr.write(f"native wheel smoke test exited with status {result.returncode}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
