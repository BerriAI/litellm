#!/usr/bin/env python3
"""Gate: run changed tests/test_litellm files against a mutated cost map.

The provider sync rewrites prices, context limits and deprecation dates in
model_prices_and_context_window.json whenever a vendor changes them. A test that
pins any of those values breaks on the next sync even though no litellm code
changed. This gate applies one combined mutation to every cost-map entry the
same way the audit did (prices x1.37, deprecation_date set, max_* limits +1000),
writes both JSON copies, runs the changed test files, and restores the files
from git afterwards. A red run means a test asserts a vendor fact instead of a
litellm-owned invariant.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Final, NamedTuple

from pydantic import TypeAdapter

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
COST_MAP_PATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)
TERMINATION_SIGNALS: Final = (signal.SIGTERM, signal.SIGHUP)

PRICE_MULTIPLIER: Final = 1.37
DEPRECATION_DATE: Final = "2030-01-01"
LIMIT_BUMP: Final = 1_000
LIMIT_FIELDS: Final = frozenset({"max_tokens", "max_input_tokens", "max_output_tokens"})

_COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, object])
_MODEL_ENTRY_ADAPTER: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST_ADAPTER: Final = TypeAdapter(list[object])


class _Args(NamedTuple):
    base: str | None
    paths: tuple[str, ...]
    pytest_args: tuple[str, ...]


def _exit_on_termination(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def _install_termination_handlers() -> None:
    for termination in TERMINATION_SIGNALS:
        if signal.getsignal(termination) == signal.SIG_DFL:
            signal.signal(termination, _exit_on_termination)


def _run(cmd: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    proc: Final = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def _cost_map_is_dirty() -> bool:
    status: Final = _run(["git", "status", "--porcelain", "--", *COST_MAP_PATHS])
    return bool(status.strip())


def _changed_test_files(base: str) -> tuple[str, ...]:
    out: Final = _run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            base,
            "HEAD",
            "--",
            ":(glob)tests/test_litellm/**/*.py",
        ]
    )
    return tuple(
        line
        for line in out.splitlines()
        if line.startswith("tests/test_litellm/") and line.endswith(".py") and Path(line).name != "conftest.py"
    )


def _mutate_value(key: str, value: object, scale_numbers: bool = False) -> object:
    inside_cost: Final = scale_numbers or "cost" in key
    if isinstance(value, dict):
        mapping: Final = _MODEL_ENTRY_ADAPTER.validate_python(value)
        return {k: _mutate_value(k, v, inside_cost) for k, v in mapping.items()}
    if isinstance(value, list):
        items: Final = _OBJECT_LIST_ADAPTER.validate_python(value)
        return [_mutate_value(key, v, inside_cost) for v in items]
    if inside_cost and isinstance(value, (int, float)) and not isinstance(value, bool):
        return value * PRICE_MULTIPLIER
    return value


def mutate_entry(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (
            value + LIMIT_BUMP
            if key in LIMIT_FIELDS and isinstance(value, int) and not isinstance(value, bool)
            else _mutate_value(key, value)
        )
        for key, value in {**entry, "deprecation_date": DEPRECATION_DATE}.items()
    }


def mutate_cost_map(cost_map: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (
            mutate_entry(_MODEL_ENTRY_ADAPTER.validate_python(value))
            if isinstance(value, dict) and "litellm_provider" in value
            else value
        )
        for key, value in cost_map.items()
    }


def _serialize(cost_map: Mapping[str, object]) -> str:
    return json.dumps(cost_map, indent=4, ensure_ascii=False) + "\n"


def _mutated_text(path: Path) -> str:
    original: Final = path.read_text()
    cost_map: Final = _COST_MAP_ADAPTER.validate_python(json.loads(original))
    return _serialize(mutate_cost_map(cost_map))


def _restore_cost_map_files() -> None:
    subprocess.run(["git", "checkout", "--", *COST_MAP_PATHS], cwd=REPO_ROOT, check=False)


def _pytest_command(files: Sequence[str], extra_args: Sequence[str]) -> list[str]:
    forwarded: Final = tuple(extra_args)
    workers: Final = (
        ()
        if any(arg == "-n" or arg.startswith("-n=") or arg.startswith("-nauto") for arg in forwarded)
        else ("-n", "4")
    )
    return [
        "uv",
        "run",
        "--no-sync",
        "pytest",
        *files,
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        *workers,
        *forwarded,
    ]


def _parse_args(argv: Sequence[str]) -> _Args:
    parser: Final = argparse.ArgumentParser(
        description="Run changed tests/test_litellm files against a mutated cost map",
        epilog="extra arguments after -- are passed to pytest",
    )
    parser.add_argument("--base", help="git ref to diff against for changed-test selection")
    parser.add_argument("paths", nargs="*", help="explicit test paths (overrides --base selection)")
    argv_tuple: Final = tuple(argv)
    before, after = (
        (argv_tuple[: argv_tuple.index("--")], argv_tuple[argv_tuple.index("--") + 1 :])
        if "--" in argv_tuple
        else (argv_tuple, ())
    )
    args: Final = parser.parse_args(before)
    return _Args(
        base=args.base,  # pyright: ignore[reportAny]  # argparse Namespace attributes are untyped
        paths=tuple(args.paths),  # pyright: ignore[reportAny]  # argparse Namespace attributes are untyped
        pytest_args=tuple(after),
    )


def main(argv: Sequence[str] | None = None) -> int:
    _install_termination_handlers()
    args: Final = _parse_args(tuple(argv) if argv is not None else tuple(sys.argv[1:]))

    files: Final = args.paths or (_changed_test_files(args.base) if args.base else ())
    if not files:
        sys.stdout.write("No tests/test_litellm files selected; nothing to gate.\n")
        return 0
    if _cost_map_is_dirty():
        sys.stderr.write(
            "Refusing to run: model_prices_and_context_window.json or its litellm/ backup "
            "has uncommitted changes. Commit or restore them first.\n"
        )
        return 2

    mutated_by_path: Final = tuple((REPO_ROOT / path, _mutated_text(REPO_ROOT / path)) for path in COST_MAP_PATHS)
    for path, text in mutated_by_path:
        path.write_text(text)
    try:
        env: Final = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
        proc: Final = subprocess.run(_pytest_command(files, args.pytest_args), cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            sys.stderr.write(
                "\nCost-map mutation gate failed: the failing assertions pin cost-map values "
                "the provider sync rewrites (prices, limits, deprecation dates). Derive the "
                "expected value from the entry the code selects (litellm.model_cost / "
                "get_model_info) or replace the assertion with an invariant our code owns.\n"
            )
        return proc.returncode
    finally:
        _restore_cost_map_files()


if __name__ == "__main__":
    raise SystemExit(main())
