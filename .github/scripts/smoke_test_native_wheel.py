from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Final

CHILD_SCRIPT: Final = """
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

native_path = Path(sys.argv[1])
spec = spec_from_file_location("litellm.rust_bridge._native", native_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot create native extension import specification")
module = module_from_spec(spec)
spec.loader.exec_module(module)

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

        native_path: Final = Path(temporary_directory) / Path(native_members[0].filename).name
        native_path.write_bytes(archive.read(native_members[0]))
        result: Final = subprocess.run((sys.executable, "-c", CHILD_SCRIPT, str(native_path)), check=False)

    if result.returncode != 0:
        sys.stderr.write(f"native wheel smoke test exited with status {result.returncode}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
