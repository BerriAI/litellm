#!/usr/bin/env python3
"""Scope a mutmut run to the production code a pull request actually changed.

mutmut reads its configuration only from ``[tool.mutmut]`` in ``pyproject.toml``,
and it filters which mutants to *execute* by fnmatch against mutant names. This
script bridges a git diff to both:

1. It rewrites ``[tool.mutmut]`` so ``paths_to_mutate`` is the changed production
   files and ``tests_dir`` is the test files that mirror them.
2. It emits mutant-name globs for the functions containing the changed lines, to
   be passed as arguments to ``mutmut run``.

mutmut installs its trampolines per top-level function and per class method, so a
changed line's enclosing function is the smallest unit it can execute.

Usage:
    python scripts/mutation_diff_scope.py --base origin/litellm_internal_staging
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
MUTATE_ROOTS: Final = ("litellm/",)
EXCLUDED_PREFIXES: Final = (
    "litellm/proxy/_experimental/",
    "litellm/types/",
)
TEST_MIRROR_ROOT: Final = "tests/test_litellm"
# mutmut runs the selected tests from a `mutants/` sandbox. It copies tests/ itself, but a
# test that reaches back out to repo tooling (scripts/, .github/) fails to import there.
SANDBOX_COPIES: Final = ("litellm/", "scripts/", ".github/")
CLASS_NAME_SEPARATOR: Final = "ǁ"
HUNK_HEADER: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
CONFIG_SECTION: Final = re.compile(r"^\[tool\.mutmut\]$.*?(?=^\[)", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class ChangedFile:
    path: str
    lines: frozenset[int]


@dataclass(frozen=True, slots=True)
class TrampolineUnit:
    """A function or method mutmut can address by name, and the lines it spans."""

    mangled_name: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class Scope:
    files: tuple[ChangedFile, ...]
    tests: tuple[str, ...]
    globs: tuple[str, ...]
    dropped_globs: tuple[str, ...]

    @property
    def is_runnable(self) -> bool:
        return bool(self.globs and self.tests)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(("git", *args), cwd=root, capture_output=True, text=True, check=True).stdout


def is_mutable_path(root: Path, path: str) -> bool:
    return (
        path.endswith(".py")
        and path.startswith(MUTATE_ROOTS)
        and not path.startswith(EXCLUDED_PREFIXES)
        and (root / path).is_file()
    )


def is_unit_test_path(root: Path, path: str) -> bool:
    """Only the mirrored unit suite; tests/e2e needs a live proxy and cannot run in mutmut's sandbox."""
    return (
        path.startswith(f"{TEST_MIRROR_ROOT}/")
        and path.endswith(".py")
        and Path(path).name.startswith("test_")
        and (root / path).is_file()
    )


def parse_unified_diff(root: Path, diff: str) -> tuple[ChangedFile, ...]:
    """Map each changed production file to the post-image line numbers the diff touched."""

    def touched_lines() -> Iterator[tuple[str, int]]:
        current = ""
        cursor = 0
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                current = line[len("+++ b/") :]
            elif (header := HUNK_HEADER.match(line)) is not None:
                cursor = int(header.group(1))
            elif line.startswith("+") and not line.startswith("+++") and current:
                yield current, cursor
                cursor += 1

    touched: Final = tuple(touched_lines())
    return tuple(
        ChangedFile(path=path, lines=frozenset(line for other, line in touched if other == path))
        for path in sorted({path for path, _ in touched})
        if is_mutable_path(root, path)
    )


def changed_test_files(root: Path, diff: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            path
            for line in diff.splitlines()
            if line.startswith("+++ b/") and is_unit_test_path(root, path := line[len("+++ b/") :])
        )
    )


def mapped_tests(root: Path, path: str) -> tuple[str, ...]:
    """Find the tests mirroring a production file, per tests/test_litellm/readme.md."""
    relative: Final = Path(path).relative_to("litellm")
    mirror_dir: Final = Path(TEST_MIRROR_ROOT) / relative.parent
    named: Final = tuple(sorted(str(p.relative_to(root)) for p in (root / mirror_dir).glob(f"test_{relative.stem}*.py")))
    if named:
        return named
    return (str(mirror_dir),) if (root / mirror_dir).is_dir() else ()


def trampoline_units(source: str) -> tuple[TrampolineUnit, ...]:
    """Every unit mutmut trampolines: top-level functions and direct class methods."""
    tree: Final = ast.parse(source)

    def unit(node: ast.stmt, class_name: str | None) -> TrampolineUnit | None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        separator = CLASS_NAME_SEPARATOR
        mangled = f"x{separator}{class_name}{separator}{node.name}" if class_name else f"x_{node.name}"
        return TrampolineUnit(
            mangled_name=mangled,
            start_line=min((d.lineno for d in node.decorator_list), default=node.lineno),
            end_line=node.end_lineno or node.lineno,
        )

    def walk() -> Iterator[TrampolineUnit]:
        for statement in tree.body:
            if (function := unit(statement, None)) is not None:
                yield function
            elif isinstance(statement, ast.ClassDef):
                for member in statement.body:
                    if (method := unit(member, statement.name)) is not None:
                        yield method

    return tuple(walk())


def module_name(path: str) -> str:
    return path[: -len(".py")].replace(os.sep, ".").replace("/", ".")


def globs_for(root: Path, changed: ChangedFile) -> tuple[str, ...]:
    try:
        units = trampoline_units((root / changed.path).read_text(encoding="utf-8"))
    except SyntaxError:
        return ()
    module: Final = module_name(changed.path)
    return tuple(
        f"{module}.{unit.mangled_name}__mutmut_*"
        for unit in units
        if any(unit.start_line <= line <= unit.end_line for line in changed.lines)
    )


def build_scope(root: Path, base: str, max_functions: int) -> Scope:
    merge_base: Final = _git(root, "merge-base", base, "HEAD").strip()
    diff: Final = _git(root, "diff", "-U0", merge_base, "--", ".")
    files: Final = parse_unified_diff(root, diff)
    all_globs: Final = tuple(glob for changed in files for glob in globs_for(root, changed))
    tests: Final = tuple(
        sorted({*(t for changed in files for t in mapped_tests(root, changed.path)), *changed_test_files(root, diff)})
    )
    return Scope(
        files=files,
        tests=tests,
        globs=all_globs[:max_functions],
        dropped_globs=all_globs[max_functions:],
    )


def render_config(scope: Scope, pytest_add_cli_args: Sequence[str]) -> str:
    def toml_list(values: Iterable[str]) -> str:
        return "[\n" + "".join(f'    "{value}",\n' for value in values) + "]"

    return (
        "[tool.mutmut]\n"
        f"paths_to_mutate = {toml_list(f.path for f in scope.files)}\n"
        f"tests_dir = {toml_list(scope.tests)}\n"
        f"also_copy = {toml_list(SANDBOX_COPIES)}\n"
        # Gathering coverage costs an extra full run of the selected tests, and mutants on
        # uncovered lines are cheap here: mutmut marks them "no tests" without running any.
        "mutate_only_covered_lines = false\n"
        f"pytest_add_cli_args = {toml_list(pytest_add_cli_args)}\n"
        "\n"
    )


def rewrite_pyproject(root: Path, scope: Scope, pytest_add_cli_args: Sequence[str]) -> None:
    pyproject: Final = root / "pyproject.toml"
    original: Final = pyproject.read_text(encoding="utf-8")
    if CONFIG_SECTION.search(original) is None:
        raise SystemExit("Could not find a [tool.mutmut] section to replace in pyproject.toml")
    pyproject.write_text(
        CONFIG_SECTION.sub(lambda _: render_config(scope, pytest_add_cli_args), original, count=1), encoding="utf-8"
    )


def existing_pytest_cli_args(root: Path) -> tuple[str, ...]:
    with open(root / "pyproject.toml", "rb") as handle:
        section: Mapping[str, object] = tomllib.load(handle)["tool"]["mutmut"]
    args: Final = section.get("pytest_add_cli_args", ())
    return tuple(str(arg) for arg in args) if isinstance(args, list) else ()


def describe(scope: Scope, max_functions: int) -> str:
    def section(title: str, values: Iterable[str]) -> Iterator[str]:
        listed = tuple(values)
        yield f"{title}: {len(listed)}"
        yield from (f"  {value}" for value in listed)

    lines: Final = (
        *section("changed production files", (f"{f.path} ({len(f.lines)} lines)" for f in scope.files)),
        *section("test selection", scope.tests),
        *section("functions to mutate", scope.globs),
        *(
            section(f"NOT mutated (over --max-functions={max_functions})", scope.dropped_globs)
            if scope.dropped_globs
            else ()
        ),
    )
    return "\n".join(lines)


def emit_github_output(scope: Scope) -> None:
    if (output := os.environ.get("GITHUB_OUTPUT")) is None:
        return
    with open(output, "a", encoding="utf-8") as handle:
        handle.write(f"has_scope={'true' if scope.is_runnable else 'false'}\n")
        handle.write(f"function_count={len(scope.globs)}\n")


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/litellm_internal_staging")
    parser.add_argument("--max-functions", type=int, default=40)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--write-pyproject", action="store_true")
    parser.add_argument("--globs-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args: Final = parser.parse_args(argv)

    root: Final = args.root.resolve()
    scope: Final = build_scope(root, args.base, args.max_functions)
    globs_out: Final = args.globs_out or root / "mutmut-scope-globs.txt"
    json_out: Final = args.json_out or root / "mutmut-scope.json"

    globs_out.write_text("".join(f"{glob}\n" for glob in scope.globs), encoding="utf-8")
    json_out.write_text(
        json.dumps(
            {
                "files": [{"path": f.path, "lines": sorted(f.lines)} for f in scope.files],
                "tests": list(scope.tests),
                "globs": list(scope.globs),
                "dropped_globs": list(scope.dropped_globs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(describe(scope, args.max_functions))

    if args.write_pyproject and scope.is_runnable:
        rewrite_pyproject(root, scope, existing_pytest_cli_args(root))
        print("rewrote [tool.mutmut] in pyproject.toml for this diff")

    emit_github_output(scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
