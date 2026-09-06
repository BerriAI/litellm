from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast

from litellm.rust_bridge import get_native_bridge, reset_native_bridge_cache

MATURIN_SPEC: Final = "maturin==1.15.0"
BRIDGE_FEATURE: Final = "trace-parity"
_RUST_ROOT: Final = "litellm-rust"
_LOCKFILE: Final = "Cargo.lock"
_SOURCE_SUFFIXES: Final = frozenset({".rs", ".toml"})
_FAILURE_OUTPUT_LINES: Final = 15
_CAPABILITY_PROBE: Final = """
from litellm.rust_bridge import get_native_bridge
bridge = get_native_bridge()
trace = getattr(bridge, "_trace", None)
if bridge is None:
    raise SystemExit(10)
if trace is None:
    raise SystemExit(11)
if not all(callable(getattr(trace, name, None)) for name in ("start_capture", "finish_capture")):
    raise SystemExit(12)
"""


def needs_rebuild(native_mtime: float | None, newest_source_mtime: float | None) -> bool:
    if native_mtime is None:
        return True
    if newest_source_mtime is None:
        return False
    return newest_source_mtime > native_mtime


def _source_files(rust_root: Path) -> Iterator[Path]:
    for path in rust_root.rglob("*"):
        if not _is_source_file(rust_root, path):
            continue
        yield path


def _is_source_file(rust_root: Path, path: Path) -> bool:
    relative: Final = path.relative_to(rust_root)
    return (
        "target" not in relative.parts
        and path.is_file()
        and (path.name == _LOCKFILE or path.suffix in _SOURCE_SUFFIXES)
    )


def _newest_source_mtime(repo_root: Path) -> float | None:
    rust_root: Final = repo_root / _RUST_ROOT
    if not rust_root.is_dir():
        return None
    return max((path.stat().st_mtime for path in _source_files(rust_root)), default=None)


def _native_module_path() -> Path | None:
    try:
        spec: Final = importlib.util.find_spec("litellm.rust_bridge._native")
    except (ImportError, ValueError):
        return None
    origin: Final = cast(str | None, getattr(spec, "origin", None))
    return Path(origin) if origin is not None else None


def _drop_imported_bridge() -> None:
    reset_native_bridge_cache()
    for name in tuple(sys.modules):
        if name.startswith("litellm.rust_bridge._native"):
            del sys.modules[name]


def _rebuild(repo_root: Path) -> tuple[bool, str]:
    command: Final = ("uvx", "--from", MATURIN_SPEC, "maturin", "develop", "--features", BRIDGE_FEATURE)
    completed: Final = subprocess.run(
        command,
        cwd=repo_root,
        env={**os.environ, "VIRTUAL_ENV": sys.prefix},
        capture_output=True,
        text=True,
        check=False,
    )
    output: Final = f"{completed.stdout}\n{completed.stderr}".strip()
    lines: Final = tuple(output.splitlines())
    return completed.returncode == 0, "\n".join(lines[-_FAILURE_OUTPUT_LINES:])


def trace_bridge_error() -> str | None:
    bridge: Final = cast(object | None, get_native_bridge())
    if bridge is None:
        return "native Rust bridge is not importable"
    trace: Final = cast(object | None, getattr(bridge, "_trace", None))
    if trace is None:
        return f"native Rust bridge does not expose _trace; it must be built with the {BRIDGE_FEATURE} feature"
    if not all(callable(getattr(trace, name, None)) for name in ("start_capture", "finish_capture")):
        return "native Rust trace bridge does not expose public route capture controls"
    return None


def _probe_trace_bridge_error() -> str | None:
    completed: Final = subprocess.run(
        (sys.executable, "-c", _CAPABILITY_PROBE),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        0: None,
        10: "native Rust bridge is not importable",
        11: f"native Rust bridge does not expose _trace; it must be built with the {BRIDGE_FEATURE} feature",
        12: "native Rust trace bridge does not expose public route capture controls",
    }.get(completed.returncode, f"native Rust bridge capability probe failed with exit code {completed.returncode}")


def ensure_trace_bridge(repo_root: Path) -> str | None:
    native_path: Final = _native_module_path()
    native_mtime: Final = native_path.stat().st_mtime if native_path is not None and native_path.exists() else None
    capability_error: Final = _probe_trace_bridge_error()
    if needs_rebuild(native_mtime, _newest_source_mtime(repo_root)) or capability_error is not None:
        print(  # noqa: T201  # CLI users need progress during the native rebuild
            f"Rebuilding native Rust bridge ({BRIDGE_FEATURE} feature)...", flush=True
        )
        rebuilt: Final = _rebuild(repo_root)
        if not rebuilt[0]:
            return f"native Rust bridge rebuild failed:\n{rebuilt[1]}"
        _drop_imported_bridge()
    return trace_bridge_error()
