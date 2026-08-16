#!/usr/bin/env python3
"""Run ``prisma generate`` only when its inputs changed since the last run.

The generated client is a pure function of ``litellm/proxy/schema.prisma`` and
the installed prisma package version, so a stamp of those two written next to
the venv is enough to prove the client is current. The stamp lives under
``sys.prefix`` so recreating the venv discards it, and a missing generated
client (a fresh or reinstalled prisma package) forces a regenerate even when
the stamp matches. The prisma package itself is never imported here: once
generated it re-exports the whole client on import, which costs more than the
generate this script exists to skip.
"""

import hashlib
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = REPO_ROOT / "litellm" / "proxy" / "schema.prisma"
STAMP = Path(sys.prefix) / "litellm-prisma-schema.stamp"


def stamp_value(schema_bytes: bytes, prisma_version: str) -> str:
    return f"{hashlib.sha256(schema_bytes).hexdigest()}:{prisma_version}"


def should_skip(stamp: Path, expected: str, client_generated: bool) -> bool:
    if not client_generated:
        return False
    try:
        return stamp.read_text() == expected
    except OSError:
        return False


def client_is_generated() -> bool:
    spec = importlib.util.find_spec("prisma")
    if spec is None or not spec.submodule_search_locations:
        return False
    return any(
        (Path(location) / "client.py").exists()
        for location in spec.submodule_search_locations
    )


def main() -> int:
    version = importlib.metadata.version("prisma")
    expected = stamp_value(SCHEMA.read_bytes(), version)
    if should_skip(STAMP, expected, client_is_generated()):
        print(
            f"Prisma client already generated for {SCHEMA.relative_to(REPO_ROOT)} "
            f"(prisma {version}); skipping prisma generate"
        )
        return 0
    result = subprocess.run(
        [sys.executable, "-m", "prisma", "generate", "--schema", str(SCHEMA)],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return result.returncode
    STAMP.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
