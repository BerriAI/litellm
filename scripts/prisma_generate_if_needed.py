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

prisma resolves its generator command (``prisma-client-py``) through a plain
PATH lookup, never through the interpreter that invoked ``prisma generate``,
so the generate runs with this interpreter's own bin directory pinned to the
front of PATH; without that pin the client lands in whichever venv the caller
happened to have on PATH (or the generate fails outright when none is).
"""

import hashlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
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


def env_with_own_bin_first(base_env: Mapping[str, str]) -> dict[str, str]:
    bin_dir = str(Path(sys.executable).parent)
    inherited = base_env.get("PATH")
    path = os.pathsep.join((bin_dir, inherited)) if inherited else bin_dir
    return {**base_env, "PATH": path}


def _run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def run_generate(run: Callable[[list[str], Path, dict[str, str]], int] = _run_command) -> int:
    return run(
        [sys.executable, "-m", "prisma", "generate", "--schema", str(SCHEMA)],
        REPO_ROOT,
        env_with_own_bin_first(os.environ),
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
    returncode = run_generate()
    if returncode != 0:
        return returncode
    STAMP.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
