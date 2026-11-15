#!/usr/bin/env python3
"""Swap the generated ``prisma/types.py`` for a stub that defines nothing at import time.

``prisma generate`` writes tens of thousands of TypedDict classes into ``types.py``. Only type
checkers read them, yet importing the client builds every one, which costs each proxy worker
several hundred MB of RSS and grows with every model added to the schema. The full file is kept
beside the stub as ``types.pyi`` so type checkers still see the real definitions. Every name
resolves to ``dict[str, Any]`` because callers build these TypedDicts by calling them.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Final

STUB_SOURCE: Final = """\
from typing import Any


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return dict[str, Any]
"""


def stub_runtime_types(prisma_dir: Path) -> bool:
    runtime: Final = prisma_dir / "types.py"
    if runtime.read_text(encoding="utf-8") == STUB_SOURCE:
        return False
    _ = runtime.replace(prisma_dir / "types.pyi")
    _ = runtime.write_text(STUB_SOURCE, encoding="utf-8")
    return True


def installed_package_dir(package: str) -> Path | None:
    spec: Final = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def main(package: str = "prisma") -> int:
    prisma_dir: Final = installed_package_dir(package)
    if prisma_dir is None:
        _ = sys.stderr.write(f"{package} is not installed in this interpreter\n")
        return 1
    changed: Final = stub_runtime_types(prisma_dir)
    _ = sys.stdout.write(f"{prisma_dir / 'types.py'}: {'stubbed' if changed else 'already stubbed'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
